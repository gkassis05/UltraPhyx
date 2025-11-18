import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from .utils import _ensure_uint8, _clip_uint8


def adjust_gain(
    analysis: Dict[str, Any],
    img: np.ndarray,
    *,
    gain: float = 1.1,
    depth_gain: Optional[float] = 0.2,   # None = disabled, otherwise exponential deep boost
    feather_px: int = 14,                 # soften edges of clean_mask region
    seed: Optional[int] = None,
    show_debug: bool = False
):
    """
    Apply multiplicative gain to the ultrasound image,
    restricted strictly to the clean_mask region.

    Parameters
    ----------
    analysis : dict
        Output of analyze_ultrasound_scan (must include "clean_mask").
    img : ndarray
        Grayscale or BGR ultrasound image.
    gain : float
        Uniform gain multiplier (e.g., 1.2 = 20% brighter).
    depth_gain : float or None
        If provided, applies depth-dependent gain:
            multiplier(y) = gain * exp(depth_gain * norm_depth)
        where norm_depth ∈ [0,1] inside clean_mask.
    feather_px : int
        Feather border inside clean_mask for natural transitions.
    seed : int or None
    show_debug : bool

    Returns
    -------
    out_u8 : uint8 image
    debug_info : dict
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # Prep
    # ---------------------------------------------------------
    img_u8 = _ensure_uint8(img)
    if img_u8.ndim == 2:
        gray = img_u8.copy()
    else:
        gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape

    clean_mask = analysis.get("clean_mask", None)
    if clean_mask is None:
        return gray, {"applied": False, "reason": "no clean_mask"}

    clean_bool = clean_mask.astype(bool)

    # ---------------------------------------------------------
    # Depth normalization inside clean_mask
    # ---------------------------------------------------------
    ys, xs = np.where(clean_bool)
    if ys.size == 0:
        return gray, {"applied": False, "reason": "empty clean_mask"}

    y_min, y_max = ys.min(), ys.max()
    height = float(y_max - y_min + 1)

    # per-pixel depth in [0,1]
    norm_depth = (np.arange(H)[:, None] - y_min) / (height + 1e-6)
    norm_depth = np.clip(norm_depth, 0, 1)

    # ---------------------------------------------------------
    # Gain curve
    # ---------------------------------------------------------
    if depth_gain is None:
        gain_map = np.full_like(gray, gain, dtype=np.float32)
    else:
        # exponential deep compensation / boost
        depth_curve = np.exp(norm_depth * depth_gain)
        gain_map = gain * depth_curve

    # ---------------------------------------------------------
    # Feathering at clean-mask boundary
    # ---------------------------------------------------------
    if feather_px > 0:
        mask_uint = clean_bool.astype(np.uint8)
        dist = cv2.distanceTransform(mask_uint, cv2.DIST_L2, 5)
        # Clamp to feather region
        d = np.clip(dist, 0, feather_px)
        t = d / feather_px

        # Sigmoid edge blend (matches your style in mirror/shadow)
        sharpness = 8.0
        alpha = 1.0 / (1.0 + np.exp(-sharpness * (t - 0.5)))

        # gain_map only active inside clean_mask
        gain_map = clean_bool * (alpha * gain_map + (1 - alpha) * 1.0) + (~clean_bool) * 1.0
    else:
        gain_map = clean_bool * gain_map + (~clean_bool) * 1.0

    # ---------------------------------------------------------
    # Apply gain
    # ---------------------------------------------------------
    out = gray.astype(np.float32) * gain_map
    out_u8 = _clip_uint8(out)

    # ---------------------------------------------------------
    # Debug
    # ---------------------------------------------------------
    dbg = dict(
        applied=True,
        gain=gain,
        depth_gain=depth_gain,
        gain_map=gain_map,
        clean_mask=clean_bool
    )

    if show_debug:
        fig, ax = plt.subplots(1, 3, figsize=(18,6))
        ax[0].imshow(gray, cmap='gray'); ax[0].set_title("Original"); ax[0].axis("off")
        im1 = ax[1].imshow(gain_map, cmap='viridis')
        ax[1].set_title("Gain Map"); ax[1].axis("off")
        fig.colorbar(im1, ax=ax[1])
        ax[2].imshow(out_u8, cmap='gray'); ax[2].set_title("With Gain"); ax[2].axis("off")
        plt.tight_layout()
        plt.show()

    return out_u8, dbg
