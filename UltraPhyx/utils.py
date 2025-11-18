import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Literal, Dict, Any, Tuple, List, Optional

ProbeType = Literal["linear", "curvilinear", "phased", "unknown"]


# ============================================================
# 0. UTIL: GRAYSCALE + UINT8 + Nakagami
# ============================================================

def sample_param(v):
    """
    Handle UltraPhyx parameter sampling:
    - If v is a scalar (int/float/bool), return it as is
    - If v is a tuple/list (low, high):
        * If both are ints → sample int
        * If either is float → sample float with 0.05 increments
    """
    # Not a range
    if not isinstance(v, (tuple, list)):
        return v

    if len(v) != 2:
        raise ValueError(f"Range parameter must be 2-length tuple, got {v}")

    a, b = v

    # Case 1: both integers → sample integer
    if isinstance(a, int) and isinstance(b, int):
        return np.random.randint(a, b + 1)

    # Case 2: floats or mixed → sample float with 0.05 increments
    a = float(a)
    b = float(b)

    # Step size = 0.05
    step = 0.05
    n_steps = int(round((b - a) / step)) + 1

    # Generate grid of possible values
    grid = a + step * np.arange(n_steps)

    # Select a random value from the grid
    val = float(np.random.choice(grid))

    # Round nicely to 2 decimals for cleanliness
    return round(val, 2)


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert image to uint8 [0,255] if needed.
    Accepts float [0,1] or [0,255], or other integer types.
    """
    if img.dtype == np.uint8:
        return img
    arr = img.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)

def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)

def _clip_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)

def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert an input ultrasound image to uint8 grayscale.
    Supports:
        - HxW (grayscale)
        - HxWx3 (BGR or RGB; treated as BGR here for cv2)
        - 1xHxW or HxWx1 (squeezed)
    """
    if img.ndim == 2:
        gray = img.copy()
    elif img.ndim == 3:
        if img.shape[-1] == 1:
            gray = img[..., 0]
        elif img.shape[0] == 1 and img.shape[1] > 1:
            gray = img[0]
        else:
            # Assume BGR (cv2 style). If your images are RGB, swap channels before calling.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape for grayscale conversion: {img.shape}")

    # Ensure uint8
    if gray.dtype != np.uint8:
        grayf = gray.astype(np.float32)
        if grayf.max() <= 1.0:
            grayf *= 255.0
        grayf = np.clip(grayf, 0, 255)
        gray = grayf.astype(np.uint8)

    return gray

def estimate_from_intensity(intensity: np.ndarray) -> Tuple[float, float]:
    """
    Estimate Nakagami parameters (m, Omega) from intensity image.
    This assumes 'intensity' ~ amplitude^2 (ish), but we just use
    the classic moment-based estimation.

    Returns:
        m      : shape parameter
        Omega  : spread parameter (mean intensity)
    """
    I = intensity.astype(np.float64)
    mean_I = I.mean()
    var_I = I.var() + 1e-8
    m = (mean_I ** 2) / var_I
    Omega = mean_I

    # Clamp to sane ranges
    m = float(max(1e-3, min(m, 50.0)))
    Omega = float(max(1e-3, min(Omega, 1e6)))
    return m, Omega


def sample_nakagami(size, m: float, Omega: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample Nakagami-squared intensity:
        R^2 ~ Gamma(shape=m, scale=Omega/m)
    """
    R2 = rng.gamma(shape=m, scale=(Omega / m), size=size)
    return R2


# ============================================================
# 1. CLEAN MASK EXTRACTION
# ============================================================

def _extract_clean_mask(
        img: np.ndarray,
        blur_ksize: int = 7,
        brightness_factor: float = 0.35,
        morph_ksize: int = 15,
        disconnect_threshold: float = 0.05,
        show_debug: bool = False
    ) -> np.ndarray:
    """
    Creates a clean bright mask:
      - grayscale
      - blur
      - brightness threshold (relative to mean brightness)
      - per-component closing
      - remove small components (< disconnect_threshold * largest area)
    Returns:
        clean_mask (uint8, in {0,1})
    """
    gray = _to_gray_uint8(img)
    h, w = gray.shape

    # Blur
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Threshold relative to mean
    mean_intensity = float(np.mean(blur))
    thr = brightness_factor * mean_intensity
    init_mask = (blur > thr).astype(np.uint8)

    # Connected components BEFORE morphology
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(init_mask, 8)

    kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
    final_mask = np.zeros_like(init_mask)

    # Per-component closing (no merging of separate blobs)
    for comp_id in range(1, num_labels):
        component = (labels == comp_id).astype(np.uint8)

        closed = cv2.morphologyEx(component, cv2.MORPH_CLOSE, kernel)
        local_support = cv2.dilate(component, kernel)
        closed = np.logical_and(closed, (local_support > 0)).astype(np.uint8)

        final_mask[closed == 1] = 1

    # Recompute CC and keep big ones
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(final_mask, 8)

    if num_labels2 <= 1:
        clean_mask = final_mask
    else:
        areas2 = stats2[:, cv2.CC_STAT_AREA]
        largest_area = np.max(areas2[1:])
        clean_mask = np.zeros_like(final_mask)
        for comp_id in range(1, num_labels2):
            if areas2[comp_id] >= disconnect_threshold * largest_area:
                clean_mask[labels2 == comp_id] = 1

    if show_debug:
        fig, axs = plt.subplots(1, 5, figsize=(18, 4))
        axs[0].imshow(gray, cmap="gray"); axs[0].set_title("Gray"); axs[0].axis("off")
        axs[1].imshow(blur, cmap="gray"); axs[1].set_title("Blur"); axs[1].axis("off")
        axs[2].imshow(init_mask, cmap="gray"); axs[2].set_title("Initial Mask"); axs[2].axis("off")
        axs[3].imshow(labels, cmap="tab20"); axs[3].set_title("CC IDs"); axs[3].axis("off")
        axs[4].imshow(clean_mask, cmap="gray"); axs[4].set_title("Clean Mask"); axs[4].axis("off")
        plt.tight_layout(); plt.show()

    return clean_mask


# ============================================================
# 2. LINEAR SHAPE ANALYSIS
# ============================================================

def _analyze_linear_shape(
        clean_mask: np.ndarray,
        band_height: int = 12,
        rect_ratio_threshold = 0.75,
        slope_threshold = 2.0,
        top_edge_flatness_threshold = 0.70,
        side_walls_slope_threshold = 1.0,
        show_debug: bool = False
    ) -> Dict[str, Any]:
    """
    Linear-probe detector:
      - Uses top/bottom bands
      - Measures width profile, edge flatness, side-wall slopes
      - If show_debug=True, prints which rules passed/failed.
    """
    ys, xs = np.where(clean_mask == 1)
    h, w = clean_mask.shape

    if len(xs) == 0:
        if show_debug:
            print("[LinearCheck] Empty mask → cannot be linear.")
        return {
            "is_linear": False,
            "reason": "empty mask",
            "linear_debug_reasons": [],
            "widths": np.array([]),
            "y_min": 0,
            "y_max": 0,
            "bbox": (0, 0, w - 1, h - 1),
        }

    y_min, y_max = ys.min(), ys.max()
    H = y_max - y_min + 1
    bh = min(band_height, max(1, H // 4))

    # Width profile over region
    widths = np.array([(clean_mask[y] > 0).sum() for y in range(y_min, y_max + 1)])
    if len(widths) > 1:
        slope = np.polyfit(np.arange(len(widths)), widths, 1)[0]
    else:
        slope = 0.0

    # Top & bottom bands
    top_band = clean_mask[y_min : y_min + bh]
    bot_band = clean_mask[y_max - bh + 1 : y_max + 1]

    width_top = float(np.mean(np.sum(top_band > 0, axis=1)))
    width_bottom = float(np.mean(np.sum(bot_band > 0, axis=1)))
    rect_ratio = width_top / (width_bottom + 1e-6)

    # Top edge flatness
    left_edges, right_edges = [], []
    for row in top_band:
        xs_row = np.where(row > 0)[0]
        if len(xs_row):
            left_edges.append(xs_row[0])
            right_edges.append(xs_row[-1])

    if len(left_edges) == 0:
        top_edge_flatness = 0.0
    else:
        L = np.mean(left_edges)
        R = np.mean(right_edges)
        top_width = R - L + 1
        top_edge_flatness = top_width / (width_bottom + 1e-6)

    # Side walls
    x_lefts, x_rights = [], []
    for y in range(y_min, y_max + 1):
        xs_row = np.where(clean_mask[y] > 0)[0]
        if len(xs_row):
            x_lefts.append(xs_row[0])
            x_rights.append(xs_row[-1])

    if len(x_lefts) > 1:
        left_slope = np.polyfit(np.arange(len(x_lefts)), x_lefts, 1)[0]
    else:
        left_slope = 0.0

    if len(x_rights) > 1:
        right_slope = np.polyfit(np.arange(len(x_rights)), x_rights, 1)[0]
    else:
        right_slope = 0.0

    # Debug plots
    if show_debug:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        axs[0].plot(widths, marker="o")
        axs[0].set_title(f"Width profile (slope={slope:.2f})")
        axs[0].set_xlabel("Row idx (within mask)")
        axs[0].set_ylabel("Width")
        axs[0].grid(True)

        axs[1].plot(left_edges, marker="o", label="Left edge")
        axs[1].plot(right_edges, marker="o", label="Right edge")
        axs[1].set_title(f"Top edge flatness={top_edge_flatness:.2f}")
        axs[1].set_xlabel("Row idx in top band")
        axs[1].set_ylabel("X pos")
        axs[1].grid(True); axs[1].legend()

        axs[2].plot(x_lefts, marker="o", label="Left wall")
        axs[2].plot(x_rights, marker="o", label="Right wall")
        axs[2].set_title(f"Side walls (L={left_slope:.2f}, R={right_slope:.2f})")
        axs[2].set_xlabel("Row idx in mask")
        axs[2].set_ylabel("X pos")
        axs[2].grid(True); axs[2].legend()

        plt.tight_layout(); plt.show()

    # Decision rules
    is_linear = True
    debug_reasons = []

    # 1) top vs bottom width
    if rect_ratio < rect_ratio_threshold:
        is_linear = False
        debug_reasons.append(
            f"[LinearCheck] rect_ratio={rect_ratio:.3f} < {rect_ratio_threshold:.3f} → rejects linear."
        )

    # 2) width slope
    if slope > slope_threshold:
        is_linear = False
        debug_reasons.append(
            f"[LinearCheck] slope={slope:.2f} > +{slope_threshold:.2f} → widening downward → non-linear."
        )
    if slope < -(slope_threshold):
        is_linear = False
        debug_reasons.append(
            f"[LinearCheck] slope={slope:.2f} < -{slope_threshold:.2f} → narrowing downward → non-linear."
        )

    # 3) top flatness
    if top_edge_flatness < top_edge_flatness_threshold:
        is_linear = False
        debug_reasons.append(
            f"[LinearCheck] top_edge_flatness={top_edge_flatness:.2f} < {top_edge_flatness_threshold} → top not flat."
        )

    # 4) side walls
    if abs(left_slope) > side_walls_slope_threshold:
        is_linear = False
        debug_reasons.append(
            f"[LinearCheck] |left_slope|={abs(left_slope):.2f} > {side_walls_slope_threshold} → left wall tilted."
        )
    if abs(right_slope) > side_walls_slope_threshold:
        is_linear = False
        debug_reasons.append(
            f"[LinearCheck] |right_slope|={abs(right_slope):.2f} > {side_walls_slope_threshold} → right wall tilted."
        )

    if show_debug:
        print("\n=== LINEAR DETECTOR DECISION ===")
        if is_linear:
            print("Result: LINEAR probe ✔")
            print(f"   rect_ratio={rect_ratio:.3f}, slope={slope:.2f}, "
                  f"top_flat={top_edge_flatness:.2f}, "
                  f"L_slope={left_slope:.2f}, R_slope={right_slope:.2f}")
        else:
            print("Result: NOT linear.")
            for r in debug_reasons:
                print("   ", r)

    return dict(
        is_linear=is_linear,
        width_top=width_top,
        width_bottom=width_bottom,
        rect_ratio=rect_ratio,
        slope=slope,
        top_edge_flatness=top_edge_flatness,
        left_slope=left_slope,
        right_slope=right_slope,
        top_band_height=bh,
        y_min=y_min,
        y_max=y_max,
        widths=widths,
        bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        linear_debug_reasons=debug_reasons,
    )


# ============================================================
# 3. TOPOLOGY HELPERS
# ============================================================

def _get_top_boundary_points(clean_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each x, find smallest y where mask==1."""
    h, w = clean_mask.shape
    xs = np.arange(w)
    ys = np.full_like(xs, fill_value=h, dtype=np.int32)

    for x in range(w):
        col = np.where(clean_mask[:, x] > 0)[0]
        if len(col) > 0:
            ys[x] = col[0]

    valid = ys < h
    return xs[valid].astype(np.float32), ys[valid].astype(np.float32)


def _fit_two_lines_and_apex(xs: np.ndarray, ys: np.ndarray, show_debug: bool = False):
    """
    Fit two lines that meet at an apex (V-shape).
    Returns (m1, b1), (m2, b2), (apex_x, apex_y), residual.
    """
    n = len(xs)
    if n < 20:
        if show_debug:
            print("[V-Fit] Not enough points (<20) to fit V-shape.")
        return None, None, None, np.inf

    order = np.argsort(xs)
    xs = xs[order]; ys = ys[order]

    best_resid = np.inf
    best_params = (None, None, None, np.inf)

    k_min = max(3, int(0.2 * n))
    k_max = min(n - 3, int(0.8 * n))

    for k in range(k_min, k_max + 1):
        xs_left, ys_left = xs[:k], ys[:k]
        xs_right, ys_right = xs[k:], ys[k:]

        A_l = np.vstack([xs_left, np.ones_like(xs_left)]).T
        m1, b1 = np.linalg.lstsq(A_l, ys_left, rcond=None)[0]

        A_r = np.vstack([xs_right, np.ones_like(xs_right)]).T
        m2, b2 = np.linalg.lstsq(A_r, ys_right, rcond=None)[0]

        if abs(m1 - m2) < 1e-6:
            continue

        apex_x = (b2 - b1) / (m1 - m2)
        apex_y = m1 * apex_x + b1

        ys_pred = np.where(xs < apex_x, m1 * xs + b1, m2 * xs + b2)
        resid = np.mean(np.abs(ys - ys_pred))

        if resid < best_resid:
            best_resid = resid
            best_params = ((m1, b1), (m2, b2), (apex_x, apex_y), resid)

    (m1, b1), (m2, b2), (apex_x, apex_y), best_resid = best_params

    if show_debug and m1 is not None:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal", adjustable="box")

        ax.scatter(xs, ys, s=20, color="blue", label="Boundary points")

        xl = np.linspace(xs.min(), apex_x, 100)
        yl = m1 * xl + b1
        ax.plot(xl, yl, color="red", label="Left line")

        xr = np.linspace(apex_x, xs.max(), 100)
        yr = m2 * xr + b2
        ax.plot(xr, yr, color="green", label="Right line")

        ax.scatter([apex_x], [apex_y], color="purple", s=80, marker="x", label="Apex")

        for x_i, y_i in zip(xs, ys):
            y_pred = m1 * x_i + b1 if x_i < apex_x else m2 * x_i + b2
            ax.plot([x_i, x_i], [y_i, y_pred], color="gray", alpha=0.2)

        ax.set_title(
            f"V-fit: Apex=({apex_x:.1f},{apex_y:.1f}), resid={best_resid:.2f}\n"
            f"m1={m1:.2f}, m2={m2:.2f}"
        )
        ax.grid(True); ax.legend()
        plt.gca().invert_yaxis()
        plt.tight_layout(); plt.show()

    return best_params


# ============================================================
# 4. NON-LINEAR CLASSIFICATION HELPERS
# ============================================================

def _compute_top_band_distribution(
        clean_mask: np.ndarray,
        y_min: int,
        band_height: int,
        show_debug: bool = False
    ) -> Dict[str, Any]:
    """
    Top band x-distribution metrics:
      - variance
      - kurtosis
      - dip_score (two lobes)
      - peak_score (central peak)
    """
    top_band = clean_mask[y_min : y_min + band_height]
    xs_all = []
    for row in top_band:
        xs_row = np.where(row > 0)[0]
        xs_all.extend(xs_row.tolist())
    xs_all = np.array(xs_all)

    if len(xs_all) == 0:
        if show_debug:
            print("[TopBand] No points in top band; distribution metrics default to zero.")
        return dict(xs_all=xs_all, variance=0.0, kurt=0.0,
                    dip_score=0.0, peak_score=0.0, width=0)

    width = int(xs_all.max() - xs_all.min() + 1)
    var = float(np.var(xs_all))

    if len(xs_all) > 1 and var > 0:
        mu = xs_all.mean()
        kurt = float(np.mean((xs_all - mu)**4) / (var**2 + 1e-6))
    else:
        kurt = 0.0

    nbins = max(20, width // 3)
    hist, _ = np.histogram(xs_all, nbins)
    mid = nbins // 2
    left = hist[:mid]
    right = hist[mid+1:]
    center = hist[mid] if mid < len(hist) else 0

    dip_score = float((left.mean() + right.mean()) / (center + 1e-6))
    peak_score = float(center / (hist.mean() + 1e-6))

    if show_debug:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].hist(xs_all, bins=nbins)
        axs[0].set_title(
            f"Top-band x-dist\nvar={var:.1f}, kurt={kurt:.1f}\n"
            f"dip={dip_score:.2f}, peak={peak_score:.2f}"
        )
        axs[0].set_xlabel("x"); axs[0].set_ylabel("count")
        axs[0].grid(True)

        axs[1].plot(hist, marker="o")
        axs[1].set_title("Histogram bins")
        axs[1].set_xlabel("bin"); axs[1].set_ylabel("count")
        axs[1].grid(True)

        plt.tight_layout(); plt.show()

    return dict(
        xs_all=xs_all,
        variance=var,
        kurt=kurt,
        dip_score=dip_score,
        peak_score=peak_score,
        width=width
    )


def _classify_non_linear(
        dist: Dict[str, Any],
        slope: float,
        left_slope: float,
        right_slope: float,
        var_threshold: float = 500.0,
        dip_threshold: float = 5000.0,
        show_debug: bool = False
    ) -> str:
    """
    Logic (tunable via var_threshold, dip_threshold):

      • If variance < var_threshold → strong PHASED vote
      • If dip_score < dip_threshold → strong PHASED vote
      • inward walls (L>0, R<0) → soft PHASED vote

      • If variance ≥ var_threshold → CURVILINEAR vote
      • If dip_score ≥ dip_threshold → CURVILINEAR vote
      • outward walls (L<0, R>0) → soft CURVILINEAR vote

    Returns: "curvilinear" | "phased" | "unknown"
    """
    var = dist["variance"]
    dip = dist["dip_score"]
    peak = dist["peak_score"]

    reasons = []

    # === PHASED rules ======================
    phased_votes = 0

    if var < var_threshold:
        phased_votes += 1
        reasons.append(f"[Phased] variance={var:.1f} < {var_threshold:.1f} → phased vote.")

    if dip < dip_threshold:
        phased_votes += 1
        reasons.append(f"[Phased] dip_score={dip:.1f} < {dip_threshold:.1f} → phased vote.")

    if left_slope > 0 and right_slope < 0:
        phased_votes += 1
        reasons.append(
            f"[Phased] inward slopes L={left_slope:.2f}, R={right_slope:.2f} → inward V (soft vote)."
        )

    # === CURVILINEAR rules =================
    curvi_votes = 0

    if var >= var_threshold:
        curvi_votes += 1
        reasons.append(f"[Curvilinear] variance={var:.1f} ≥ {var_threshold:.1f} → wide footprint.")

    if dip >= dip_threshold:
        curvi_votes += 1
        reasons.append(
            f"[Curvilinear] dip_score={dip:.1f} ≥ {dip_threshold:.1f} → stronger two-lobe separation."
        )

    if left_slope < 0 and right_slope > 0:
        curvi_votes += 1
        reasons.append(
            f"[Curvilinear] outward slopes L={left_slope:.2f}, R={right_slope:.2f} (soft vote)."
        )

    # === DEBUG PRINT =======================
    if show_debug:
        print("\n=== NON-LINEAR CLASSIFIER ===")
        print(f"variance={var:.1f}, dip={dip:.1f}, peak={peak:.1f}")
        print(f"left_slope={left_slope:.2f}, right_slope={right_slope:.2f}, width_slope={slope:.2f}")
        print(f"phased_votes={phased_votes}, curvi_votes={curvi_votes}")
        for r in reasons:
            print("   ", r)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["phased", "curvilinear"],
               [phased_votes, curvi_votes],
               color=["skyblue", "orange"])
        ax.set_ylim(0, 3)
        ax.set_title("Non-linear votes")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    if phased_votes > curvi_votes:
        return "phased"
    elif curvi_votes > phased_votes:
        return "curvilinear"
    else:
        return "unknown"


# ============================================================
# 5. GEOMETRY FROM APEX
# ============================================================

def _compute_geometry_from_apex(
        apex_x: float,
        apex_y: float,
        xs_b: np.ndarray,
        ys_b: np.ndarray,
        clean_mask: np.ndarray,
        show_debug: bool = False
    ) -> Dict[str, Any]:
    """
    Generic non-linear geometry from apex:
      - theta_min, theta_max, theta_center from top boundary
      - r_max: max distance from apex to any mask pixel
    """
    ang = np.arctan2(ys_b - apex_y, xs_b - apex_x)
    theta_min = float(np.percentile(ang, 2))
    theta_max = float(np.percentile(ang, 98))
    theta_center = 0.5 * (theta_min + theta_max)

    ys_all, xs_all = np.where(clean_mask == 1)
    if len(xs_all) == 0:
        r_max = 0.0
    else:
        r_max = float(np.max(np.sqrt((xs_all - apex_x)**2 + (ys_all - apex_y)**2)))

    if show_debug:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].hist(ang, bins=40)
        ax[0].set_title(
            f"Angles from apex\nθmin={theta_min:.2f}, θmax={theta_max:.2f}, θc={theta_center:.2f}"
        )
        ax[0].set_xlabel("θ (rad)"); ax[0].set_ylabel("count")
        ax[0].grid(True)

        thetas = np.linspace(theta_min, theta_max, 30)
        xs_ray = apex_x + r_max * np.cos(thetas)
        ys_ray = apex_y + r_max * np.sin(thetas)
        ax[1].scatter(xs_b, ys_b, s=10, label="Top boundary")
        ax[1].scatter([apex_x], [apex_y], c="red", marker="x", s=80, label="Apex")
        ax[1].scatter(xs_ray, ys_ray, s=10, c="green", label="Rays end")
        ax[1].invert_yaxis()
        ax[1].set_title(f"r_max={r_max:.1f}")
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout(); plt.show()

    return dict(
        apex=(float(apex_x), float(apex_y)),
        theta_min=theta_min,
        theta_max=theta_max,
        theta_center=theta_center,
        r_max=r_max
    )


def _unknown_geometry_defaults(
        xs_b: np.ndarray,
        ys_b: np.ndarray,
        show_debug: bool = False
    ) -> Dict[str, Any]:
    """Generic plausible aperture for unknown type."""
    if len(xs_b) > 0:
        mid_x = float(np.mean(xs_b))
        width = float(xs_b.max() - xs_b.min() + 1)
    else:
        mid_x = 0.0
        width = 100.0

    if len(ys_b) > 0:
        apex_y = float(ys_b.min() - 20)
    else:
        apex_y = -20.0

    theta_span = np.deg2rad(min(60.0, width / 5.0))
    theta_min = -theta_span / 2.0
    theta_max = +theta_span / 2.0
    theta_center = 0.0
    r_max = float(width * 2.0)

    geom = dict(
        apex=(mid_x, apex_y),
        theta_min=theta_min,
        theta_max=theta_max,
        theta_center=theta_center,
        r_max=r_max
    )

    if show_debug:
        print("\n[UnknownGeom] Using default unknown geometry:")
        print(f"   apex=({mid_x:.1f}, {apex_y:.1f}), width={width:.1f}, "
              f"θ=[{theta_min:.2f},{theta_max:.2f}], r_max={r_max:.1f}")

    return geom


# ============================================================
# 6. BRIGHTEST STRUCTURE MASKS (TOP-K COMPONENTS)
# ============================================================

def _extract_structure_masks(
        img: np.ndarray,
        clean_mask: np.ndarray,
        blur_ksize: int = 3,
        threshold_percentile: float = 90.0,
        max_components: int = 3,
        min_rel_area: float = 0.001,
        show_debug: bool = False
    ) -> Dict[str, Any]:
    """
    Extracts top-K brightest *connected components* inside the clean scan mask.

    Steps:
      - convert to grayscale + blur
      - multiply by clean_mask (only inside scan region)
      - compute intensity threshold from given percentile of in-mask pixels
      - threshold → binary structure mask
      - connected components, sort by area descending
      - keep up to max_components, each as a separate mask
      - ignore components whose area < min_rel_area * total_clean_mask_area

    Returns dict:
      {
        "structure_masks": [mask1, mask2, ...],    # each uint8 {0,1}
        "component_areas": [area1, area2, ...],
        "threshold_value": float,
      }
    """
    gray = _to_gray_uint8(img)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    masked = blur.copy()
    masked[clean_mask == 0] = 0

    valid_pixels = masked[clean_mask > 0].astype(np.float32)
    if valid_pixels.size == 0:
        if show_debug:
            print("[StructureMask] No pixels inside clean_mask; returning empty structure set.")
        return dict(structure_masks=[], component_areas=[], threshold_value=0.0)

    thr = float(np.percentile(valid_pixels, threshold_percentile))
    binary = np.zeros_like(masked, dtype=np.uint8)
    binary[(masked >= thr) & (clean_mask > 0)] = 1

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    structure_masks: List[np.ndarray] = []
    component_areas: List[int] = []

    mask_area = int(clean_mask.sum())
    area_thresh = max(1, int(min_rel_area * mask_area))

    # sort component ids (exclude 0) by area descending
    comp_ids = list(range(1, num_labels))
    comp_ids.sort(key=lambda cid: stats[cid, cv2.CC_STAT_AREA], reverse=True)

    for cid in comp_ids[:max_components]:
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < area_thresh:
            continue
        m = (labels == cid).astype(np.uint8)
        structure_masks.append(m)
        component_areas.append(area)

    if show_debug:
        print("\n[StructureMask] threshold_percentile:", threshold_percentile,
              "→ thr:", thr)
        print("[StructureMask] mask_area=", mask_area,
              "area_thresh=", area_thresh,
              "num_components_kept=", len(structure_masks))

        # Build overlay visualization
        vis = gray.copy()
        vis_color = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        for idx, m in enumerate(structure_masks):
            c = colors[idx % len(colors)]
            ys, xs = np.where(m == 1)
            vis_color[ys, xs] = c

        fig, axs = plt.subplots(1, 4, figsize=(20, 4))
        axs[0].imshow(gray, cmap="gray")
        axs[0].set_title("Gray"); axs[0].axis("off")

        axs[1].imshow(masked, cmap="gray")
        axs[1].set_title("Gray × clean_mask"); axs[1].axis("off")

        axs[2].imshow(binary, cmap="gray")
        axs[2].set_title("Structure threshold mask"); axs[2].axis("off")

        axs[3].imshow(cv2.cvtColor(vis_color, cv2.COLOR_BGR2RGB))
        axs[3].set_title("Top-K structure masks overlaid"); axs[3].axis("off")

        plt.tight_layout(); plt.show()

        # Also plot each mask separately
        if len(structure_masks) > 0:
            n = len(structure_masks)
            fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
            if n == 1:
                axs = [axs]
            for i, m in enumerate(structure_masks):
                axs[i].imshow(m, cmap="gray")
                axs[i].set_title(f"Structure #{i+1}\narea={component_areas[i]}")
                axs[i].axis("off")
            plt.tight_layout(); plt.show()

    return dict(
        structure_masks=structure_masks,
        component_areas=component_areas,
        threshold_value=thr,
    )


# ============================================================
# 7. FULL ANALYSIS
# ============================================================

def analyze_ultrasound_scan(
        img: np.ndarray,
        clean_mask_params: Optional[Dict[str, Any]] = None,
        linear_params: Optional[Dict[str, Any]] = None,
        nonlinear_params: Optional[Dict[str, Any]] = None,
        structure_params: Optional[Dict[str, Any]] = None,
        show_debug: bool = False
    ) -> Dict[str, Any]:
    """
    High-level geometry + structure analysis.

    Parameter style:
      - clean_mask_params: dict for _extract_clean_mask
          {
            "blur_ksize": int,
            "brightness_factor": float,
            "morph_ksize": int,
            "disconnect_threshold": float,
          }

      - linear_params: dict for _analyze_linear_shape
          {
            "band_height": int,
            "rect_ratio_threshold" = 0.75,
            "slope_threshold" = 2.0,
            "top_edge_flatness_threshold" = 0.70,
            "side_walls_slope_threshold" = 1.0,

          }

      - nonlinear_params: dict for _classify_non_linear
          {
            "var_threshold": float,
            "dip_threshold": float,
          }

      - structure_params: dict for _extract_structure_masks
          {
            "blur_ksize": int,
            "threshold_percentile": float,
            "max_components": int,
            "min_rel_area": float,
          }

    Returns dict with keys:
      - "probe_type"      : "linear" | "curvilinear" | "phased" | "unknown"
      - "clean_mask"      : main bright mask for scan region
      - "structure_masks" : list of top-K masks inside scan region
      - "structure_info"  : metadata from structure extraction
      - "geometry"        : type-specific geometry dict
      - "polygon"         : Nx2 contour of scan region
      - "crop_bbox"       : (x_min, y_min, x_max, y_max)
      - "cropped_image"   : cropped scan region
      - "original_image"  : original input image
    """

    # Defaults for parameter dicts
    cm_defaults = dict(
        blur_ksize=7,
        brightness_factor=0.4,
        morph_ksize=15,
        disconnect_threshold=0.05,
    )
    if clean_mask_params is not None:
        cm_defaults.update(clean_mask_params)

    lin_defaults = dict(
        band_height=12,
        rect_ratio_threshold = 0.75,
        slope_threshold = 2.0,
        top_edge_flatness_threshold = 0.70,
        side_walls_slope_threshold = 1.0,
    )
    if linear_params is not None:
        lin_defaults.update(linear_params)

    nl_defaults = dict(
        var_threshold=500.0,
        dip_threshold=5000.0,
    )
    if nonlinear_params is not None:
        nl_defaults.update(nonlinear_params)

    struct_defaults = dict(
        blur_ksize=3,
        threshold_percentile=90.0,
        max_components=3,
        min_rel_area=0.001,
    )
    if structure_params is not None:
        struct_defaults.update(structure_params)

    # 1) Clean mask
    clean_mask = _extract_clean_mask(
        img,
        blur_ksize=cm_defaults["blur_ksize"],
        brightness_factor=cm_defaults["brightness_factor"],
        morph_ksize=cm_defaults["morph_ksize"],
        disconnect_threshold=cm_defaults["disconnect_threshold"],
        show_debug=show_debug
    )

    h, w = clean_mask.shape
    ys_all, xs_all = np.where(clean_mask == 1)

    if len(xs_all) == 0:
        if show_debug:
            print("\n[Main] Clean mask empty → classify as UNKNOWN with default geometry.")
        xs_b = np.arange(w, dtype=np.float32)
        ys_b = np.full_like(xs_b, fill_value=h//2, dtype=np.float32)
        geom_unknown = _unknown_geometry_defaults(xs_b, ys_b, show_debug=show_debug)

        # No structures if no mask
        structure_info = dict(structure_masks=[], component_areas=[], threshold_value=0.0)

        return dict(
            probe_type="unknown",
            clean_mask=clean_mask,
            structure_masks=[],
            structure_info=structure_info,
            geometry=dict(type="unknown", **geom_unknown),
            polygon=None,
            crop_bbox=(0, 0, w-1, h-1),
            cropped_image=img.copy(),
            original_image=img
        )

    # 2) Linear analysis
    lin = _analyze_linear_shape(
        clean_mask,
        band_height=lin_defaults["band_height"],
        rect_ratio_threshold = lin_defaults["rect_ratio_threshold"],
        slope_threshold = lin_defaults["slope_threshold"],
        top_edge_flatness_threshold = lin_defaults["top_edge_flatness_threshold"],
        side_walls_slope_threshold = lin_defaults["side_walls_slope_threshold"],
        show_debug=show_debug
    )

    probe_type: ProbeType = "unknown"
    geometry: Dict[str, Any] = {}

    if lin["is_linear"]:
        probe_type = "linear"
        geometry = {
            "type": "linear",
            "width_top": lin["width_top"],
            "width_bottom": lin["width_bottom"],
            "rect_ratio": lin["rect_ratio"],
            "top_edge_flatness": lin["top_edge_flatness"],
            "slope": lin["slope"],
            "left_slope": lin["left_slope"],
            "right_slope": lin["right_slope"],
            "bbox": lin["bbox"],
        }
        if show_debug:
            print("\n[Main] Final decision: LINEAR (no non-linear analysis needed).")
    else:
        # 3) Non-linear branch (always assume non-linear if not linear)
        if show_debug:
            print("\n[Main] Linear check failed → assuming NON-LINEAR and classifying type.")

        xs_b, ys_b = _get_top_boundary_points(clean_mask)

        dist = _compute_top_band_distribution(
            clean_mask, lin["y_min"], lin["top_band_height"], show_debug=show_debug
        )

        nl_type = _classify_non_linear(
            dist,
            slope=lin["slope"],
            left_slope=lin["left_slope"],
            right_slope=lin["right_slope"],
            var_threshold=nl_defaults["var_threshold"],
            dip_threshold=nl_defaults["dip_threshold"],
            show_debug=show_debug
        )

        if nl_type == "unknown":
            if show_debug:
                print("[Main] Could not confidently pick curvilinear vs phased → UNKNOWN non-linear.")
            geom_unknown = _unknown_geometry_defaults(xs_b, ys_b, show_debug=show_debug)
            probe_type = "unknown"
            geometry = dict(
                type="unknown",
                **geom_unknown,
                top_band_variance=dist["variance"],
                top_band_kurtosis=dist["kurt"],
                top_band_dip_score=dist["dip_score"],
                top_band_peak_score=dist["peak_score"],
            )
        else:
            probe_type = nl_type
            left_line, right_line, apex, v_resid = _fit_two_lines_and_apex(
                xs_b, ys_b, show_debug=show_debug
            )

            if apex is None:
                if show_debug:
                    print("[Main] V-fit failed (no apex) → fallback to UNKNOWN geometry but keep non-linear type label.")
                geom_unknown = _unknown_geometry_defaults(xs_b, ys_b, show_debug=show_debug)
                geometry = dict(
                    type=probe_type,
                    **geom_unknown,
                    vfit_residual=float(v_resid),
                    left_line=None,
                    right_line=None,
                    top_band_variance=dist["variance"],
                    top_band_kurtosis=dist["kurt"],
                    top_band_dip_score=dist["dip_score"],
                    top_band_peak_score=dist["peak_score"],
                )
            else:
                apex_x, apex_y = apex
                geom_nonlin = _compute_geometry_from_apex(
                    apex_x, apex_y, xs_b, ys_b, clean_mask, show_debug=show_debug
                )
                geometry = dict(
                    type=probe_type,
                    **geom_nonlin,
                    vfit_residual=float(v_resid),
                    left_line=left_line,
                    right_line=right_line,
                    top_band_variance=dist["variance"],
                    top_band_kurtosis=dist["kurt"],
                    top_band_dip_score=dist["dip_score"],
                    top_band_peak_score=dist["peak_score"],
                )
            if show_debug:
                print(f"[Main] Final decision: {probe_type.upper()} (non-linear).")

    # 4) Structure masks (top-K brightest blobs inside clean_mask)
    structure_info = _extract_structure_masks(
        img,
        clean_mask=clean_mask,
        blur_ksize=struct_defaults["blur_ksize"],
        threshold_percentile=struct_defaults["threshold_percentile"],
        max_components=struct_defaults["max_components"],
        min_rel_area=struct_defaults["min_rel_area"],
        show_debug=show_debug
    )
    structure_masks = structure_info["structure_masks"]

    # 5) Polygon / contour
    contours, _ = cv2.findContours(
        clean_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    polygon = None
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        polygon = cnt[:, 0, :]

    # 6) Crop bbox + image
    ys, xs = np.where(clean_mask == 1)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    crop_bbox = (x_min, y_min, x_max, y_max)

    if img.ndim == 2:
        cropped_image = img[y_min:y_max+1, x_min:x_max+1]
    else:
        cropped_image = img[y_min:y_max+1, x_min:x_max+1, :]

    # 7) Global debug figure: mask & crop
    if show_debug:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        vis = img.copy()
        if vis.ndim == 2:
            vis_color = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        else:
            vis_color = vis.copy()

        if polygon is not None:
            cv2.polylines(vis_color, [polygon.astype(np.int32)], True, (0, 255, 0), 2)

        axs[0].imshow(
            cv2.cvtColor(vis_color, cv2.COLOR_BGR2RGB)
            if vis_color.ndim == 3 else vis_color,
            cmap="gray"
        )
        axs[0].set_title(f"Original + polygon ({probe_type})"); axs[0].axis("off")

        axs[1].imshow(clean_mask, cmap="gray")
        axs[1].set_title("Clean mask"); axs[1].axis("off")

        axs[2].imshow(
            cropped_image if cropped_image.ndim == 2
            else cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB),
            cmap="gray"
        )
        axs[2].set_title("Cropped image"); axs[2].axis("off")

        plt.tight_layout(); plt.show()

    # 8) FINAL DEBUG OVERLAY: geometry + structure masks
    if show_debug:
        vis = img.copy()
        if vis.ndim == 2:
            vis_color = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        else:
            vis_color = vis.copy()

        # Draw the main mask outline
        contours, _ = cv2.findContours(clean_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(vis_color, contours, -1, (0,255,0), 2)

        # Overlay structure masks as colors
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        for idx, m in enumerate(structure_masks):
            c = colors[idx % len(colors)]
            ys_m, xs_m = np.where(m == 1)
            vis_color[ys_m, xs_m] = c

        # Draw probe geometry
        if probe_type == "linear":
            x_min, y_min, x_max, y_max = geometry["bbox"]
            cv2.rectangle(vis_color, (x_min, y_min), (x_max, y_max), (255,0,0), 2)

            cv2.putText(vis_color, "LINEAR PROBE", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
            cv2.putText(vis_color, f"rect_ratio={geometry['rect_ratio']:.3f}",
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(vis_color, f"top_flat={geometry['top_edge_flatness']:.3f}",
                        (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        elif probe_type in ("curvilinear", "phased"):
            apex_x, apex_y = map(int, geometry["apex"])
            theta_min = geometry["theta_min"]
            theta_max = geometry["theta_max"]
            r_max     = geometry["r_max"]

            label = "CURVILINEAR PROBE" if probe_type == "curvilinear" else "PHASED PROBE"
            cv2.putText(vis_color, label, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
            cv2.circle(vis_color, (apex_x, apex_y), 6, (0,0,255), -1)

            # Rays
            for th in [theta_min, theta_max]:
                x2 = int(apex_x + r_max*np.cos(th))
                y2 = int(apex_y + r_max*np.sin(th))
                cv2.line(vis_color,(apex_x,apex_y),(x2,y2),(0,255,255),2)

            # Arc
            for t in np.linspace(theta_min, theta_max, 80):
                x = int(apex_x + r_max*np.cos(t))
                y = int(apex_y + r_max*np.sin(t))
                if 0 <= y < vis_color.shape[0] and 0 <= x < vis_color.shape[1]:
                    vis_color[y, x] = (0,128,255)

            # If phased and lines exist, draw them too
            if probe_type == "phased" and geometry.get("left_line") is not None:
                (m1, b1) = geometry["left_line"]
                (m2, b2) = geometry["right_line"]
                h_img, w_img = vis_color.shape[:2]
                for m, b, col in [(m1,b1,(255,0,0)), (m2,b2,(255,0,0))]:
                    x1 = 0
                    y1 = int(m*x1 + b)
                    x2 = w_img-1
                    y2 = int(m*x2 + b)
                    cv2.line(vis_color, (x1,y1), (x2,y2), col, 2)

        else:
            cv2.putText(vis_color, "UNKNOWN PROBE", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            apex_x, apex_y = map(int, geometry["apex"])
            theta_min = geometry["theta_min"]
            theta_max = geometry["theta_max"]
            r_max     = geometry["r_max"]
            cv2.circle(vis_color, (apex_x, apex_y), 6, (0,0,255), -1)
            for th in [theta_min, theta_max]:
                x2 = int(apex_x + r_max*np.cos(th))
                y2 = int(apex_y + r_max*np.sin(th))
                cv2.line(vis_color,(apex_x,apex_y),(x2,y2),(0,128,255),2)

        plt.figure(figsize=(8,8))
        plt.imshow(cv2.cvtColor(vis_color, cv2.COLOR_BGR2RGB))
        plt.title(f"Final Decision Overlay: {probe_type.upper()} + Structures")
        plt.axis("off")
        plt.show()

        print("\n==============================")
        print("FINAL PROBE CLASSIFICATION")
        print("==============================")
        print("probe_type:", probe_type)
        print("geometry keys:", list(geometry.keys()))
        print("num_structure_masks:", len(structure_masks))

    return dict(
        probe_type=probe_type,
        clean_mask=clean_mask,
        structure_masks=structure_masks,
        structure_info=structure_info,
        geometry=geometry,
        polygon=polygon,
        crop_bbox=crop_bbox,
        cropped_image=cropped_image,
        original_image=img,
    )

# ============================================================
# 8. LONGEST INTERNAL LINE SEGMENT INSIDE STRUCTURE
# ============================================================

def _longest_internal_line(struct_mask, 
                           max_tilt_deg=45, 
                           num_angles=91, 
                           num_offsets=200,
                           dilate_amount=5):
    """
    Find the longest straight line inside struct_mask.
    First dilate the mask to allow straighter internal lines
    for slightly curved or thin structures.
    """

    # ------------------------------------------------------------
    # 1. Dilate the structure slightly
    # ------------------------------------------------------------
    if dilate_amount > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_amount, dilate_amount)
        )
        struct_mask = cv2.dilate(struct_mask.astype(np.uint8), kernel).astype(bool)

    # ------------------------------------------------------------
    # 2. Original logic begins
    # ------------------------------------------------------------
    H, W = struct_mask.shape
    ys, xs = np.where(struct_mask)
    if len(xs) == 0:
        return None

    cx = xs.mean()
    cy = ys.mean()

    best_len = -1
    best_pts = None
    best_angle = None

    angles = np.linspace(-max_tilt_deg, max_tilt_deg, num_angles)
    angles_rad = np.deg2rad(angles)

    for ang, rad in zip(angles, angles_rad):
        vx = np.cos(rad)
        vy = np.sin(rad)

        nx = -vy
        ny = vx

        offs = np.linspace(-150, 150, num_offsets)

        for o in offs:
            px = cx + o * nx
            py = cy + o * ny

            # forward
            x0, y0 = px, py
            lf = 0
            while True:
                x0 += vx
                y0 += vy
                ix = int(round(x0))
                iy = int(round(y0))
                if ix < 0 or ix >= W or iy < 0 or iy >= H or not struct_mask[iy, ix]:
                    break
                lf += 1

            # backward
            x1, y1 = px, py
            lb = 0
            while True:
                x1 -= vx
                y1 -= vy
                ix = int(round(x1))
                iy = int(round(y1))
                if ix < 0 or ix >= W or iy < 0 or iy >= H or not struct_mask[iy, ix]:
                    break
                lb += 1

            total = lf + lb
            if total > best_len:
                best_len = total
                best_pts = (x0, y0, x1, y1)
                best_angle = ang

    if best_pts is None:
        return None

    return {
        "angle_deg": best_angle,
        "length": best_len,
        "x0": best_pts[0],
        "y0": best_pts[1],
        "x1": best_pts[2],
        "y1": best_pts[3]
    }