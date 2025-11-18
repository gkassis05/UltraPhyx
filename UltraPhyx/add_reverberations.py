import numpy as np
import cv2
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

from .utils import _ensure_uint8, _clip_uint8, estimate_from_intensity, sample_nakagami, _longest_internal_line


def add_reverberations(
    analysis: Dict[str, Any],
    img: np.ndarray,
    *,
    max_depth_fraction: float = 0.35,      # max depth of reverb region
    prefer_thin_prob: float = 0.8,         # probability to choose thinnest structures
    max_reverbs: int = 15,                  # maximum allowed reverberation copies
    decay_rate: float = 0.4,               # how fast echoes darken with depth
    spacing_jitter: float = 0.20,          # random jitter in echo spacing
    base_spacing_px: float = 15.0,         # typical spacing in pixels
    lateral_feather_px: int = 5,          # edge feather
    strength: float = 1.0,                 # overall intensity scaling
    #BRIGHTNESS CONTROL
    intensity_base: float = 0.5,           # baseline scaling (was 0.5)
    intensity_noise_amp: float = 0.8,      # Nakagami modulation amplitude (was 0.8)
    seed: Optional[int] = None,
    show_debug: bool = False
):
    """
    Add reverberation artifacts under a chosen structure.
    """
    rng = np.random.default_rng(seed)

    # ---------------------------
    # Prep
    # ---------------------------
    img_u8 = _ensure_uint8(img)
    gray = img_u8 if img_u8.ndim == 2 else cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    clean = analysis.get("clean_mask", None)
    structs = analysis.get("structure_masks", [])
    if clean is None or len(structs) == 0:
        return gray, {"applied": False, "reason": "no structures"}

    clean_bool = clean.astype(bool)

    # ---------------------------
    # Thinness scoring for each structure
    # ---------------------------
    perp_widths = []
    line_infos = []

    for m in structs:
        mask_bool = m.astype(bool) & clean_bool

        if mask_bool.sum() < 20:
            perp_widths.append(1e9)
            line_infos.append(None)
            continue

        info = _longest_internal_line(mask_bool)
        if info is None or info.get("length", 0) < 5:
            perp_widths.append(1e9)
            line_infos.append(None)
            continue

        x0 = float(info["x0"])
        y0 = float(info["y0"])
        x1 = float(info["x1"])
        y1 = float(info["y1"])
        dx = x1 - x0
        dy = y1 - y0
        L = np.hypot(dx, dy)
        if L < 1e-3:
            perp_widths.append(1e9)
            line_infos.append(None)
            continue

        # unit direction
        ux = dx / L
        uy = dy / L
        # perpendicular
        px = -uy
        py = ux

        ys_s, xs_s = np.where(mask_bool)
        mid_x = 0.5 * (x0 + x1)
        mid_y = 0.5 * (y0 + y1)

        proj = (xs_s - mid_x) * px + (ys_s - mid_y) * py
        w = proj.max() - proj.min()

        perp_widths.append(w)
        line_infos.append(info)

    perp_widths = np.array(perp_widths, float)

    if perp_widths.size == 0 or np.all(perp_widths >= 1e8):
        return gray, {"applied": False, "reason": "no valid structures"}

    thin_idx = int(np.argmin(perp_widths))

    if rng.uniform() < prefer_thin_prob:
        chosen_idx = thin_idx
    else:
        chosen_idx = int(rng.integers(0, len(structs)))

    S = structs[chosen_idx].astype(bool) & clean_bool

    if S.sum() < 20:
        return gray, {"applied": False, "reason": "tiny structure"}

    # ---------------------------
    # Fit line for chosen structure
    # ---------------------------
    info_line = line_infos[chosen_idx]
    if info_line is None or info_line.get("length", 0) < 5:
        info_line = _longest_internal_line(S)

    if info_line is None or info_line.get("length", 0) < 5:
        return gray, {"applied": False, "reason": "no valid line"}

    x0 = float(info_line["x0"])
    y0 = float(info_line["y0"])
    x1 = float(info_line["x1"])
    y1 = float(info_line["y1"])
    dx = x1 - x0
    dy = y1 - y0
    L = np.hypot(dx, dy)

    if L < 5:
        return gray, {"applied": False, "reason": "line too short"}

    ux = dx / L
    uy = dy / L
    nx = -uy
    ny = ux

    # ---------------------------
    # Downward normal
    # ---------------------------
    geom = analysis.get("geometry", {})
    if "apex" in geom:
        apex_x, apex_y = geom["apex"]
    else:
        ys_s_tmp, xs_s_tmp = np.where(S)
        apex_y = ys_s_tmp.min()
        apex_x = xs_s_tmp[ys_s_tmp.argmin()]

    mid_x = 0.5 * (x0 + x1)
    mid_y = 0.5 * (y0 + y1)
    v_apex = np.array([apex_x - mid_x, apex_y - mid_y])
    v_norm = np.array([nx, ny])

    if np.dot(v_apex, v_norm) > 0:
        nx, ny = -nx, -ny

    # ---------------------------
    # Depth constraints
    # ---------------------------
    ys_s, xs_s = np.where(S)
    y_bottom = ys_s.max()

    ys_clean, xs_clean = np.where(clean_bool)
    if ys_clean.size == 0:
        return gray, {"applied": False, "reason": "empty clean"}

    clean_bottom = ys_clean.max()
    room_in_clean = clean_bottom - y_bottom
    room_in_clean = max(0, room_in_clean)

    max_depth_fraction = float(min(max_depth_fraction, 1/3))
    requested_depth = int(max_depth_fraction * H)
    max_depth_px = min(requested_depth, room_in_clean)

    if max_depth_px <= 1:
        return gray, {"applied": False, "reason": "no depth"}

    # ---------------------------
    # Reverberations count & spacing
    # ---------------------------
    n_reverbs = int(rng.integers(1, max_reverbs + 1))

    base = base_spacing_px
    spacing = base + spacing_jitter * base * (rng.uniform(-1, 1))

    # ---------------------------
    # Nakagami modulation
    # ---------------------------
    I_region = gray[clean_bool]
    if I_region.size == 0:
        I_region = gray
    m_est, Omega_est = estimate_from_intensity(I_region)

    nak = sample_nakagami(gray.shape, m_est, Omega_est, rng)
    nak = nak - nak.min()
    nak = nak / (nak.max() + 1e-6)

    # ---------------------------
    # Build artifact
    # ---------------------------
    artifact = np.zeros_like(gray, float)

    ys_s, xs_s = np.where(S)
    intens = gray[ys_s, xs_s]

    for k in range(1, n_reverbs + 1):
        shift = k * spacing
        if shift > max_depth_px:
            break

        sx = int(round(nx * shift))
        sy = int(round(ny * shift))

        xs_r = xs_s + sx
        ys_r = ys_s + sy

        valid = (
            (xs_r >= 0) & (xs_r < W) &
            (ys_r >= 0) & (ys_r < H)
        )
        if not np.any(valid):
            continue

        xs_r = xs_r[valid]
        ys_r = ys_r[valid]

        # constrain inside clean mask
        reverb_mask = np.zeros_like(gray, bool)
        reverb_mask[ys_r, xs_r] = True
        reverb_mask &= clean_bool

        if reverb_mask.sum() == 0:
            continue

        alpha_k = strength * np.exp(-decay_rate * (k / n_reverbs))

        intens_src = intens[valid]
        noise_local = nak[ys_r, xs_r]

        # brightness control
        intens_mod = intens_src * (intensity_base + intensity_noise_amp * noise_local)

        artifact[ys_r, xs_r] = np.maximum(
            artifact[ys_r, xs_r],
            intens_mod * alpha_k
        )

    # ---------------------------
    # Feathering
    # ---------------------------
    if lateral_feather_px > 0:
        mask_art = (artifact > 0).astype(np.uint8)
        if mask_art.any():
            dist = cv2.distanceTransform(mask_art, cv2.DIST_L2, 5)
            d = np.clip(dist, 0, lateral_feather_px)
            t = d / lateral_feather_px
            sharp = 8.0
            alpha_edge = 1.0 / (1.0 + np.exp(-sharp * (t - 0.5)))

            blended = alpha_edge * artifact + (1 - alpha_edge) * gray
            artifact[dist < lateral_feather_px] = blended[dist < lateral_feather_px]

    out = gray.astype(float)
    mask_total = artifact > 0
    out[mask_total] = artifact[mask_total]
    out = _clip_uint8(out)

    # ---------------------------
    # Debug
    # ---------------------------
    if show_debug:
        fig, ax = plt.subplots(1, 4, figsize=(22, 6))
        ax[0].imshow(gray, cmap='gray'); ax[0].set_title("Original"); ax[0].axis('off')
        ax[1].imshow(S, cmap='gray'); ax[1].set_title("Chosen Structure"); ax[1].axis('off')

        # draw internal line
        line_vis = gray.copy()
        for t in np.linspace(-L, L, 400):
            xx = int(round(mid_x + t * ux))
            yy = int(round(mid_y + t * uy))
            if 0 <= xx < W and 0 <= yy < H:
                line_vis[yy, xx] = 255
        ax[2].imshow(line_vis, cmap='gray'); ax[2].set_title("Internal Line"); ax[2].axis('off')

        ax[3].imshow(out, cmap='gray'); ax[3].set_title("With Reverberations"); ax[3].axis('off')
        plt.tight_layout()
        plt.show()

    return out, {
        "applied": True,
        "structure_idx": chosen_idx,
        "line_info": info_line,
        "n_reverbs": n_reverbs,
        "spacing": spacing,
        "max_depth_px": max_depth_px,
    }
