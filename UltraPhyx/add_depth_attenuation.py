import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple

from .utils import _ensure_uint8, _clip_uint8, _rng, estimate_from_intensity, sample_nakagami


def add_depth_attenuation(
    analysis: Dict[str, Any],
    img: np.ndarray,
    *,
    # --- body-habitus / physical attenuation ---
    strength: float = 0.75,          # how much to apply attenuation [0..1]
    curve_shape: float = 2.5,       # depth^curve_shape; >1 = steeper
    habitus_alpha: float = 1.8,     # exponential decay rate (physical attenuation)
    # --- TGC-like compensation ---
    enable_tgc: bool = True,
    tgc_strength: float = 0.5,      # how wiggly / strong the TGC curve is
    tgc_rescue: float = 0.3,        # upward bias in far field (simulated TGC "rescuing" depth)
    tgc_num_knots: int = 8,         # number of depth "sliders" for TGC curve
    # --- noise / SNR modulation ---
    noise_amplitude: float = 0.85,   # how much Nakagami noise modulates multiplier
    seed: Optional[int] = None,
    show_debug: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Add depth-based attenuation (body habitus) + optional TGC-like compensation
    ONLY inside the clean_mask from `analysis`.

    Conceptually:
        1) Apply physical exponential attenuation with depth:
               I -> I * exp(-habitus_alpha * depth^curve_shape)
        2) Optionally apply a TGC curve (piecewise, slider-style) along depth:
               I -> I * TGC(depth)
        3) Modulate with Nakagami noise to capture SNR variability with depth.

    Parameters
    ----------
    analysis : dict
        Output of analyze_ultrasound_scan, must contain "clean_mask".
    img : ndarray
        Input ultrasound image (2D or 3D; color will be converted to gray internally).
    strength : float
        0 → no attenuation, 1 → full habitus attenuation curve applied.
    curve_shape : float
        Exponent shaping depth-normalization: depth_norm ** curve_shape.
    habitus_alpha : float
        Exponential decay coefficient controlling how strongly intensity falls with depth.
    enable_tgc : bool
        If True, apply a random TGC-like compensation curve along depth.
    tgc_strength : float
        Amplitude of TGC deviations from 1.0 (higher → more aggressive TGC).
    tgc_rescue : float
        Bias to increase gain in the far field (simulating user pulling bottom sliders up).
    tgc_num_knots : int
        Number of slider-like knots for the TGC curve along depth.
    noise_amplitude : float
        Nakagami noise modulation strength on the final multiplier.
    seed : int or None
        RNG seed.
    show_debug : bool
        If True, shows debug plots of curves and results.

    Returns
    -------
    out : uint8 ndarray
        Attenuated (and possibly TGC-compensated) image.
    debug_info : dict
        Contains:
          - "multiplier"
          - "depth_norm"
          - "habitus_curve"
          - "tgc_curve"
          - "noise"
    """

    rng = _rng(seed)

    # -----------------------------------------
    # PREP
    # -----------------------------------------
    img_u8 = _ensure_uint8(img)
    if img_u8.ndim == 2:
        gray = img_u8
    else:
        gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape

    clean_mask = analysis.get("clean_mask", None)
    if clean_mask is None or clean_mask.sum() == 0:
        return img_u8.copy(), {
            "applied": False,
            "reason": "no clean_mask",
        }

    clean = clean_mask.astype(bool)

    # -----------------------------------------
    # DEPTH NORMALIZATION 0→1 (row-based)
    # -----------------------------------------
    depth_norm = np.linspace(0.0, 1.0, H, dtype=np.float32).reshape(H, 1)
    depth_norm = np.repeat(depth_norm, W, axis=1)

    # -----------------------------------------
    # BODY-HABITUS ATTENUATION (physical)
    # -----------------------------------------
    # base depth curve in [0,1], shaped by exponent
    base_depth = np.clip(depth_norm ** curve_shape, 0.0, 1.0)
    # physical-like exponential attenuation
    habitus_curve = np.exp(-habitus_alpha * base_depth)

    # Mix between no attenuation (1.0) and full habitus curve
    # strength=0 → multiplier 1 everywhere
    # strength=1 → full habitus_curve
    habitus_mult = (1.0 - strength) + strength * habitus_curve
    habitus_mult = np.clip(habitus_mult, 0.0, 1.5)

    # -----------------------------------------
    # TGC-LIKE COMPENSATION
    # -----------------------------------------
    if enable_tgc and tgc_num_knots >= 2:
        # depth positions of "sliders"
        knot_depths = np.linspace(0.0, 1.0, tgc_num_knots, dtype=np.float32)

        # random base gains around 1.0
        base_gains = 1.0 + tgc_strength * (rng.uniform(-1.0, 1.0, size=tgc_num_knots))

        # add an upward bias for deep region if tgc_rescue > 0
        if tgc_rescue > 0.0:
            bias = np.linspace(0.0, 1.0, tgc_num_knots)
            base_gains += tgc_rescue * bias

        # clamp gains
        base_gains = np.clip(base_gains, 0.3, 3.0)

        # interpolate per-row TGC gain
        depth_1d = depth_norm[:, 0]
        tgc_curve_1d = np.interp(depth_1d, knot_depths, base_gains)
        tgc_curve = tgc_curve_1d.reshape(H, 1)
        tgc_curve = np.repeat(tgc_curve, W, axis=1)
    else:
        tgc_curve = np.ones_like(gray, dtype=np.float32)

    # -----------------------------------------
    # NAKAGAMI NOISE MODULATION (SNR-with-depth)
    # -----------------------------------------
    inside_vals = gray[clean]
    if inside_vals.size == 0:
        inside_vals = gray

    m_est, Omega_est = estimate_from_intensity(inside_vals)
    noise = sample_nakagami(gray.shape, m_est, Omega_est, rng)
    noise = noise - noise.min()
    noise = noise / (noise.max() + 1e-6)

    # center to [-0.5, 0.5] then scale
    noise_centered = (noise - 0.5) * 2.0
    noise_mult = 1.0 + noise_amplitude * noise_centered
    noise_mult = np.clip(noise_mult, 0.5, 1.5)

    # -----------------------------------------
    # FINAL MULTIPLIER (only inside clean_mask)
    # -----------------------------------------
    total_mult = habitus_mult * tgc_curve * noise_mult

    multiplier = np.ones_like(gray, dtype=np.float32)
    multiplier[clean] = total_mult[clean]
    multiplier = np.clip(multiplier, 0.0, 3.0)

    # -----------------------------------------
    # APPLY
    # -----------------------------------------
    out_f = gray.astype(np.float32) * multiplier
    out_u8 = _clip_uint8(out_f)

    debug = {
        "applied": True,
        "multiplier": multiplier,
        "depth_norm": depth_norm,
        "habitus_curve": habitus_mult,
        "tgc_curve": tgc_curve,
        "noise": noise,
    }

    # -----------------------------------------
    # DEBUG PLOTS
    # -----------------------------------------
    if show_debug:
        fig, ax = plt.subplots(2, 3, figsize=(18, 10))

        ax[0, 0].imshow(gray, cmap="gray")
        ax[0, 0].set_title("Original")
        ax[0, 0].axis("off")

        ax[0, 1].imshow(habitus_mult, cmap="viridis")
        ax[0, 1].set_title("Habitus Multiplier")
        ax[0, 1].axis("off")

        ax[0, 2].imshow(tgc_curve, cmap="plasma")
        ax[0, 2].set_title("TGC Curve (2D)")
        ax[0, 2].axis("off")

        ax[1, 0].imshow(noise_mult, cmap="magma")
        ax[1, 0].set_title("Noise Multiplier")
        ax[1, 0].axis("off")

        ax[1, 1].imshow(multiplier, cmap="cividis")
        ax[1, 1].set_title("Total Multiplier")
        ax[1, 1].axis("off")

        ax[1, 2].imshow(out_u8, cmap="gray")
        ax[1, 2].set_title("Output (Attenuation + TGC)")
        ax[1, 2].axis("off")

        plt.tight_layout()
        plt.show()

        # Also plot 1D curves vs depth (center column)
        center_x = W // 2
        depth_axis = np.arange(H)
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
        ax2.plot(depth_axis, habitus_mult[:, center_x], label="Habitus")
        ax2.plot(depth_axis, tgc_curve[:, center_x], label="TGC")
        ax2.plot(depth_axis, multiplier[:, center_x], label="Total")
        ax2.set_xlabel("Row (depth)")
        ax2.set_ylabel("Gain")
        ax2.set_title("Depth Profiles @ center column")
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.show()

    return out_u8, debug
