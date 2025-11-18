import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from .utils import _ensure_uint8, _clip_uint8, _longest_internal_line


# ============================================================
# MIRROR ARTIFACT GENERATOR
# ============================================================

def add_mirror(
    analysis: Dict[str, Any],
    img: np.ndarray,
    *,
    max_depth_fraction: float = 0.30,
    max_tilt_deg: float = 45.0,
    mirror_strength: float = 0.85,
    decay_rate: float = 0.2,
    lateral_blur_px: int = 18,
    original_remaining_threshold_percentage: float = 0.01,
    opp_angle_thresh_deg: float = 45,
    seed: Optional[int] = None,
    show_debug: bool = True
):

    rng = np.random.default_rng(seed)

    # ======================================================
    # PREP + CLEAN MASKS
    # ======================================================
    img_u8 = _ensure_uint8(img)
    gray = img_u8 if img_u8.ndim == 2 else cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    clean = analysis["clean_mask"].astype(bool)
    structs = analysis["structure_masks"]

    if len(structs) == 0:
        return gray, {"applied": False, "reason": "no structures"}

    # pick random structure
    idx = int(rng.integers(0, len(structs)))
    S = structs[idx].astype(bool) & clean
    if S.sum() < 20:
        return gray, {"applied": False, "reason": "tiny structure"}

    # ======================================================
    # INTERNAL LINE
    # ======================================================
    info_line = _longest_internal_line(S, max_tilt_deg=max_tilt_deg)
    if info_line is None or info_line["length"] < 5:
        return gray, {"applied": False, "reason": "no valid internal line"}

    x0, y0 = info_line["x0"], info_line["y0"]
    x1, y1 = info_line["x1"], info_line["y1"]

    dx = x1 - x0
    dy = y1 - y0
    L = np.hypot(dx, dy)
    if L < 5:
        return gray, {"applied": False, "reason": "short line"}

    # ======================================================
    # LINE NORMAL FORM  (a,b,c: ax + by + c = 0)
    # ======================================================
    a = dy / L
    b = -dx / L
    c = -(a * x0 + b * y0)

    # force "upward-ish" normal
    if b < 0:
        a, b, c = -a, -b, -c

    # ======================================================
    # APEX-OPPOSITE CONSTRAINT
    # ======================================================
    geom = analysis.get("geometry", {})
    if "apex" in geom:
        apex_x, apex_y = geom["apex"]
    else:
        ys_s, xs_s = np.where(S)
        apex_idx = ys_s.argmin()
        apex_x, apex_y = xs_s[apex_idx], ys_s[apex_idx]

    apex_x = float(apex_x)
    apex_y = float(apex_y)

    mid_x = 0.5 * (x0 + x1)
    mid_y = 0.5 * (y0 + y1)

    v_apex = np.array([apex_x - mid_x, apex_y - mid_y])
    v_norm = np.array([a, b])

    dot = np.dot(v_apex, v_norm)
    denom = np.linalg.norm(v_apex) * np.linalg.norm(v_norm) + 1e-8
    angle_deg = np.rad2deg(np.arccos(np.clip(dot / denom, -1, 1)))

    if angle_deg < opp_angle_thresh_deg:
        a, b, c = -a, -b, -c

    # ======================================================
    # GEOMETRY GRID
    # ======================================================
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    D = a * X + b * Y + c     # signed distance from line (since |(a,b)| = 1)

    # Below side of reflector plane
    dist_below = D           # positive below, negative above
    below = dist_below > 0

    # X-range for mirror zone
    xmin = int(np.floor(min(x0, x1)))
    xmax = int(np.ceil(max(x0, x1)))
    xmin = max(0, xmin)
    xmax = min(W - 1, xmax)
    valid_x = (X >= xmin) & (X <= xmax)

    # ======================================================
    # REFLECTION COORDINATES
    # ======================================================
    Xr = X - 2 * a * D
    Yr = Y - 2 * b * D

    Xr_i = np.clip(Xr.astype(int), 0, W - 1)
    Yr_i = np.clip(Yr.astype(int), 0, H - 1)

    # source must be clean
    src_valid = clean[Yr_i, Xr_i]

    # ======================================================
    # NEW PHYSICS LIMIT:
    # Mirror cannot exceed available “real depth above”
    # ======================================================
    depth_available = np.zeros(W, dtype=int)

    for col in range(xmin, xmax + 1):
        above_mask = (dist_below[:, col] < 0) & clean[:, col]
        depth_available[col] = np.count_nonzero(above_mask)

    max_depth_map = depth_available[np.newaxis, :]

    valid_struct_depth = (dist_below >= 0) & (dist_below <= max_depth_map)

    # ======================================================
    # FRACTIONAL DEPTH LIMIT (existing)
    # ======================================================
    if dist_below.max() > 0:
        depth_norm = dist_below / (dist_below.max() + 1e-6)
    else:
        depth_norm = np.zeros_like(dist_below)

    # -------------------------
    # RANDOM DEPTH FRACTION (skewed deeper)
    # -------------------------
    low = 0.1
    high = max_depth_fraction
    p = 0.5   # lower p => stronger skew toward deeper
    
    U = rng.uniform(0.0, 1.0)
    chosen_fraction = low + (high - low) * (U ** p)
    
    valid_fraction_depth = depth_norm < chosen_fraction

    # combine both
    valid_depth = valid_struct_depth & valid_fraction_depth

    # ======================================================
    # FINAL VALID REGION
    # ======================================================
    valid = clean & src_valid & below & valid_x & valid_depth

    # reflection source
    src = gray[Yr_i, Xr_i]

    # fade with depth
    alpha = mirror_strength * np.exp(-decay_rate * depth_norm)
    alpha = np.clip(alpha, 0, 1)

    artifact = np.zeros_like(gray, float)
    artifact[valid] = src[valid] * alpha[valid]

    # ======================================================
    # SIGMOID EDGE-ONLY FEATHERING  (best for ultrasound)
    # ======================================================
    if lateral_blur_px > 0:
    
        # -- 1) Distance to nearest NON-valid pixel
        #     So distance = 0 at the boundary, grows inward.
        dist = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    
        # Clamp distance to the chosen feather range
        d = np.clip(dist, 0, lateral_blur_px)
    
        # Convert 0 → 1 smooth transition using a sigmoid
        # sharpness controls how soft/strong the blend is
        sharpness = 8.0   # increase for sharper edge, decrease for smoother
        t = (d / lateral_blur_px)          # normalized distance [0,1]
        alpha_edge = 1.0 / (1.0 + np.exp(-sharpness * (t - 0.5)))
    
        # alpha goes 0 → 1 at boundary → interior.
        # Now we need to blend artifact with original gray only near edges.
        blended = alpha_edge * artifact + (1 - alpha_edge) * gray
    
        # Only apply blending at locations where boundary is near
        # i.e. where dist < lateral_blur_px
        mask_boundary = (dist < lateral_blur_px)
        artifact[mask_boundary] = blended[mask_boundary]




    artifact_u8 = _clip_uint8(artifact)

    # ======================================================
    # FINAL BLEND
    # ======================================================
    out = gray.astype(float)
    out[valid] = (
        original_remaining_threshold_percentage * out[valid]
        + artifact_u8[valid]
    )
    out = _clip_uint8(out)

    # ======================================================
    # DEBUG
    # ======================================================
    if show_debug:
        line_mask = np.zeros_like(gray)
        ts = np.linspace(-L, L, 500)
        for t in ts:
            xx = int(round(mid_x + t * (dx / L)))
            yy = int(round(mid_y + t * (dy / L)))
            if 0 <= xx < W and 0 <= yy < H:
                line_mask[yy, xx] = 255

        fig, ax = plt.subplots(1, 6, figsize=(34, 6))
        ax[0].imshow(gray, cmap='gray'); ax[0].set_title("Original"); ax[0].axis("off")
        ax[1].imshow(S, cmap='gray'); ax[1].set_title("Structure Mask"); ax[1].axis("off")
        ax[2].imshow(line_mask, cmap='gray'); ax[2].set_title("Internal Line"); ax[2].axis("off")
        ax[3].imshow(valid, cmap='viridis'); ax[3].set_title("Valid Mirror Zone"); ax[3].axis("off")
        ax[4].imshow(artifact_u8, cmap='gray'); ax[4].set_title("Artifact Only"); ax[4].axis("off")
        ax[5].imshow(out, cmap='gray'); ax[5].set_title("Final Output"); ax[5].axis("off")
        plt.tight_layout()
        plt.show()

    return out, {
        "applied": True,
        "structure_index": idx,
        "line_info": info_line,
        "xmin": xmin,
        "xmax": xmax
    }
