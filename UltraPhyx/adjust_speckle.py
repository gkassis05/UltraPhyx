import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from .utils import (
    _ensure_uint8,
    _clip_uint8,
    estimate_from_intensity,
    sample_nakagami,
)


def adjust_speckle(
    analysis: Dict[str, Any],
    img: np.ndarray,
    *,
    target_m: Optional[float] = 0.5,         # override shape parameter (e.g., 0.6–1.2)
    target_Omega: Optional[float] = None,     # override scale parameter
    strength: float = 0.7,                    # blending with original speckle
    feather_px: int = 12,                     # blending edge around mask
    seed: Optional[int] = None,
    show_debug: bool = False
):
    """
    Adjust speckle characteristics within clean_mask by modifying the 
    underlying Nakagami distribution.

    Parameters
    ----------
    target_m : float or None
        Desired Nakagami shape parameter (controls speckle granularity).
        - lower m (~0.3–0.6) → coarse, grainy speckle (older machines)
        - higher m (~1.0–1.5) → finer speckle (modern machines)
    target_Omega : float or None
        Scale parameter (controls brightness envelope)
    strength : float in [0,1]
        0   = no change
        1   = full replacement of speckle in clean_mask
    feather_px : int
        Smooth blending around clean_mask edges.
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------------------
    # Preprocessing
    # ---------------------------------------------
    img_u8 = _ensure_uint8(img)
    if img_u8.ndim == 2:
        gray = img_u8.copy()
    else:
        gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    clean = analysis.get("clean_mask", None)
    if clean is None:
        return gray, {"applied": False, "reason": "no clean_mask"}
    clean_bool = clean.astype(bool)

    # ---------------------------------------------
    # Estimate original speckle parameters inside mask
    # ---------------------------------------------
    region = gray[clean_bool]
    if region.size == 0:
        region = gray

    m_orig, Omega_orig = estimate_from_intensity(region)

    # If user does not specify target values → random small perturbation
    if target_m is None:
        target_m = float(m_orig * rng.uniform(0.7, 1.4))
    if target_Omega is None:
        target_Omega = float(Omega_orig * rng.uniform(0.8, 1.2))

    # ---------------------------------------------
    # Sample new speckle field
    # ---------------------------------------------
    speck = sample_nakagami((H, W), m=target_m, Omega=target_Omega, rng=rng)
    speck -= speck.min()
    speck /= (speck.max() + 1e-8)

    # ---------------------------------------------
    # Normalize original intensities into [0,1]
    # ---------------------------------------------
    gray_norm = gray.astype(np.float32) / 255.0

    # ---------------------------------------------
    # Replace speckle texture while preserving anatomy envelope
    # ---------------------------------------------
    # Anatomy envelope (slow spatial variation)
    envelope = cv2.GaussianBlur(gray_norm, (13,13), 4)

    # Blend between original and generated speckle
    new_texture = (1 - strength) * gray_norm + strength * (envelope * speck)

    # Clip & convert
    new_texture_u8 = _clip_uint8(new_texture * 255.0)

    # ---------------------------------------------
    # Feather mask boundary
    # ---------------------------------------------
    if feather_px > 0:
        mask_uint = clean_bool.astype(np.uint8)
        dist = cv2.distanceTransform(mask_uint, cv2.DIST_L2, 5)
        d = np.clip(dist, 0, feather_px)
        t = d / feather_px
        sharp = 7.0
        alpha = 1.0 / (1.0 + np.exp(-sharp * (t - 0.5)))

        # Only blend inside mask
        out = gray.astype(np.float32)
        out = alpha * new_texture_u8 + (1 - alpha) * out
        out = _clip_uint8(out)
    else:
        out = gray.copy()
        out[clean_bool] = new_texture_u8[clean_bool]

    # ---------------------------------------------
    # Debug
    # ---------------------------------------------
    dbg = dict(
        applied=True,
        original_m=m_orig,
        original_Omega=Omega_orig,
        target_m=target_m,
        target_Omega=target_Omega,
        strength=strength,
    )

    if show_debug:
        fig, ax = plt.subplots(1, 4, figsize=(22,6))
        ax[0].imshow(gray, cmap='gray'); ax[0].set_title("Original"); ax[0].axis("off")
        ax[1].imshow(speck, cmap='gray'); ax[1].set_title("Generated Speckle"); ax[1].axis("off")
        ax[2].imshow(new_texture_u8, cmap='gray'); ax[2].set_title("New Texture (mask only)"); ax[2].axis("off")
        ax[3].imshow(out, cmap='gray'); ax[3].set_title("Final Output"); ax[3].axis("off")
        plt.tight_layout()
        plt.show()

    return out, dbg
