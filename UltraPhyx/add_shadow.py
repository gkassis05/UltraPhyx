import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
from .utils import _rng, _ensure_uint8, _clip_uint8, estimate_from_intensity, sample_nakagami


def add_shadow(
    analysis: Dict[str, Any],
    img: np.ndarray,
    *,
    # shadow geometry / extent
    max_depth_fraction: float = 0.33,   # ≤ 1/3 of image height, as requested
    structure_dilate_iters: int = 2,    # thicken bright structure before shadowing
    fan_spread_factor: float = 0.6,     # how much shadow widens with depth for fan probes
    # shadow intensity control
    shadow_strength: float = 0.65,       # 0 = no change, 1 = can reach very dark
    decay_rate: float = 2.0,            # how fast shadow fades with depth
    edge_blur_ksize: int = 9,           # Gaussian blur for soft edges (odd number)
    # Nakagami control (if None, auto-estimate from image under clean_mask)
    m: Optional[float] = None,
    Omega: Optional[float] = None,
    seed: Optional[int] = None,
    show_debug: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Add a physically-plausible acoustic shadow to `img`, using:

      - The probe geometry + clean_mask + structure_masks
        from `analysis` (output of analyze_ultrasound_scan).
      - A randomly chosen bright structure mask.
      - Shadow shape that follows the chosen structure downward.
      - Depth-limited & attenuated with depth.
      - Darkness controlled by `shadow_strength`.
      - Nakagami-distributed texture inside the shadow.

    Parameters
    ----------
    analysis : dict
        Output from analyze_ultrasound_scan. Must contain:
          - "clean_mask"       : 2D mask of scan region
          - "structure_masks"  : list of 2D masks of bright structures
          - "geometry"         : probe geometry dict (type, apex, etc.)
          - "probe_type"       : "linear" | "curvilinear" | "phased" | "unknown"
    img : ndarray
        Original ultrasound image (2D or 3D).
    max_depth_fraction : float
        Maximum shadow depth as fraction of image height (capped at 1/3).
    structure_dilate_iters : int
        How many dilation iterations to thicken the chosen structure.
    fan_spread_factor : float
        For non-linear probes: how much wider the shadow gets with depth.
    shadow_strength : float in [0,1]
        Overall darkness strength (multiplicative attenuation).
    decay_rate : float
        Exponential decay of shadow strength with depth.
    edge_blur_ksize : int
        Gaussian blur kernel for soft edges (odd number; 0 or 1 disables blur).
    m, Omega : float or None
        Nakagami parameters. If None, estimated from intensities inside clean_mask.
    seed : int or None
        RNG seed for reproducibility.
    show_debug : bool
        If True, shows debug figures and returns more debug metadata.

    Returns
    -------
    shadowed_img : ndarray
        Image with one shadow added.
    debug_info : dict
        Contains:
          - "chosen_index"
          - "structure_mask_raw"
          - "structure_mask_dilated"
          - "shadow_weight"
          - "nakagami_norm"
          - "multiplier"
    """
    rng = _rng(seed)

    # ------------------------------------------------------------
    # Basic setup
    # ------------------------------------------------------------
    img_u8 = _ensure_uint8(img)
    if img_u8.ndim == 2:
        base_gray = img_u8
    else:
        base_gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)

    h, w = base_gray.shape

    clean_mask = analysis.get("clean_mask", None)
    structure_masks = analysis.get("structure_masks", None)
    geometry = analysis.get("geometry", {})
    probe_type = analysis.get("probe_type", "unknown")

    if clean_mask is None or structure_masks is None or len(structure_masks) == 0:
        if show_debug:
            print("[add_shadow_from_analysis] No clean_mask or structure_masks → no shadow added.")
        return img_u8.copy(), {
            "chosen_index": None,
            "structure_mask_raw": None,
            "structure_mask_dilated": None,
            "shadow_weight": np.zeros_like(base_gray, dtype=np.float32),
            "nakagami_norm": None,
            "multiplier": np.ones_like(base_gray, dtype=np.float32),
        }

    clean_mask_bool = (clean_mask.astype(bool))

    # ------------------------------------------------------------
    # 1) Choose a random structure mask (uniform)
    # ------------------------------------------------------------
    idx = int(rng.integers(0, len(structure_masks)))
    struct_raw = structure_masks[idx].astype(bool) & clean_mask_bool

    # If the chosen structure is tiny, loop a bit to find something non-degenerate
    tries = 0
    while struct_raw.sum() < 50 and tries < len(structure_masks):
        idx = int(rng.integers(0, len(structure_masks)))
        struct_raw = structure_masks[idx].astype(bool) & clean_mask_bool
        tries += 1

    if struct_raw.sum() == 0:
        if show_debug:
            print("[add_shadow_from_analysis] All structure masks are tiny or empty → no shadow.")
        return img_u8.copy(), {
            "chosen_index": None,
            "structure_mask_raw": None,
            "structure_mask_dilated": None,
            "shadow_weight": np.zeros_like(base_gray, dtype=np.float32),
            "nakagami_norm": None,
            "multiplier": np.ones_like(base_gray, dtype=np.float32),
        }

    # ------------------------------------------------------------
    # 2) Thicken the structure mask a bit
    # ------------------------------------------------------------
    struct_mask = struct_raw.astype(np.uint8)
    if structure_dilate_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        struct_mask = cv2.dilate(struct_mask, kernel, iterations=structure_dilate_iters)
    struct_mask = (struct_mask > 0) & clean_mask_bool

    ys_struct, xs_struct = np.where(struct_mask)
    y_bottom_struct = ys_struct.max()

    # ------------------------------------------------------------
    # 3) Decide shadow depth (≤ 1/3 of image height, also limited by bottom)
    # ------------------------------------------------------------
    max_depth_fraction = float(min(max_depth_fraction, 1.0 / 3.0))
    max_depth_px = int(max_depth_fraction * h)

    # cannot go past the bottom of the scan
    max_depth_px = min(max_depth_px, h - 1 - y_bottom_struct)
    if max_depth_px <= 1:
        if show_debug:
            print("[add_shadow_from_analysis] max_depth_px too small → no shadow.")
        return img_u8.copy(), {
            "chosen_index": idx,
            "structure_mask_raw": struct_raw,
            "structure_mask_dilated": struct_mask,
            "shadow_weight": np.zeros_like(base_gray, dtype=np.float32),
            "nakagami_norm": None,
            "multiplier": np.ones_like(base_gray, dtype=np.float32),
        }

    # ------------------------------------------------------------
    # 4) Build a shadow weight map by sliding the structure down
    #    and optionally widening (fan) for non-linear probes.
    # ------------------------------------------------------------
    shadow_weight = np.zeros_like(base_gray, dtype=np.float32)

    is_fan = (probe_type in ("curvilinear", "phased")) or ("apex" in geometry)

    # We'll shift the structure downward and accumulate a depth-based weight.
    struct_base = struct_mask.astype(np.uint8)

    for d in range(1, max_depth_px + 1):
        depth_ratio = d / max_depth_px  # 0→1

        shifted = np.zeros_like(struct_base)
        shifted[d:, :] = struct_base[:-d, :]

        # Fan widening: for non-linear probes, widen with depth
        if is_fan and fan_spread_factor > 0:
            # kernel width grows with depth_ratio
            extra = int(1 + fan_spread_factor * depth_ratio * 4)
            if extra > 1:
                k = np.ones((1, extra), np.uint8)
                shifted = cv2.dilate(shifted, k, iterations=1)

        # Constrain shadow inside clean_mask
        shifted = shifted.astype(bool) & clean_mask_bool

        # Depth-dependent weight (stronger near structure, decays with depth)
        # e^{-decay_rate * depth_ratio} in [e^{-decay_rate}, 1]
        w = float(np.exp(-decay_rate * depth_ratio))

        shadow_weight[shifted] = np.maximum(shadow_weight[shifted], w)

    # Optional edge blur to soften borders
    if edge_blur_ksize >= 3 and edge_blur_ksize % 2 == 1:
        shadow_weight = cv2.GaussianBlur(shadow_weight, (edge_blur_ksize, edge_blur_ksize), 0)

    # Normalize to [0,1] (in case some overlapping caused >1, should not but safe)
    if shadow_weight.max() > 0:
        shadow_weight = shadow_weight / shadow_weight.max()

    # ------------------------------------------------------------
    # 5) Nakagami texture inside clean_mask (or structure neighborhood)
    # ------------------------------------------------------------
    if m is None or Omega is None:
        # estimate parameters from the region under clean_mask
        I_region = base_gray[clean_mask_bool]
        if I_region.size == 0:
            I_region = base_gray
        m_est, Omega_est = estimate_from_intensity(I_region)
        if m is None:
            m = m_est
        if Omega is None:
            Omega = Omega_est

    nak = sample_nakagami(size=base_gray.shape, m=m, Omega=Omega, rng=rng)
    # Normalize Nakagami draws to [0,1]
    nak = nak - nak.min()
    if nak.max() > 0:
        nak_norm = nak / nak.max()
    else:
        nak_norm = np.zeros_like(nak, dtype=np.float32)

    # ------------------------------------------------------------
    # 6) Compute multiplicative attenuation map
    # ------------------------------------------------------------
    # Final shadow multiplier in [1 - shadow_strength, 1]:
    #   multiplier = 1 - shadow_strength * shadow_weight * (0.5 + 0.5*nak_norm)
    # So:
    #   - highest weight & highest nak → darkest
    #   - outside shadow_weight == 0 → multiplier = 1 (no change)
    core = 0.5 + 0.5 * nak_norm   # in [0.5,1]
    multiplier = 1.0 - shadow_strength * shadow_weight * core
    multiplier = np.clip(multiplier, 0.0, 1.0)

    # Only apply inside clean_mask
    mask_float = clean_mask_bool.astype(np.float32)
    # outside clean_mask: multiplier = 1
    multiplier = 1.0 * (1.0 - mask_float) + multiplier * mask_float

    # ------------------------------------------------------------
    # 7) Apply to image (all channels)
    # ------------------------------------------------------------
    img_f = img_u8.astype(np.float32)
    if img_f.ndim == 2:
        out_f = img_f * multiplier
    else:
        # broadcast 2D multiplier to channels
        out_f = img_f * multiplier[..., None]

    out_u8 = _clip_uint8(out_f)

    # ------------------------------------------------------------
    # 8) Debug plots
    # ------------------------------------------------------------
    debug_info = {
        "chosen_index": idx,
        "structure_mask_raw": struct_raw,
        "structure_mask_dilated": struct_mask,
        "shadow_weight": shadow_weight,
        "nakagami_norm": nak_norm,
        "multiplier": multiplier,
    }

    if show_debug:
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))

        axs[0,0].imshow(base_gray, cmap="gray")
        axs[0,0].set_title("Original (gray)")
        axs[0,0].axis("off")

        axs[0,1].imshow(clean_mask_bool, cmap="gray")
        axs[0,1].set_title("Clean Mask")
        axs[0,1].axis("off")

        axs[0,2].imshow(struct_mask, cmap="gray")
        axs[0,2].set_title(f"Chosen Structure (idx={idx})")
        axs[0,2].axis("off")

        im3 = axs[1,0].imshow(shadow_weight, cmap="hot")
        axs[1,0].set_title("Shadow Weight (0–1)")
        axs[1,0].axis("off")
        fig.colorbar(im3, ax=axs[1,0], fraction=0.046)

        im4 = axs[1,1].imshow(multiplier, cmap="viridis")
        axs[1,1].set_title("Multiplier (Intensity Scale)")
        axs[1,1].axis("off")
        fig.colorbar(im4, ax=axs[1,1], fraction=0.046)

        axs[1,2].imshow(out_u8 if out_u8.ndim == 2 else cv2.cvtColor(out_u8, cv2.COLOR_BGR2RGB),
                        cmap="gray")
        axs[1,2].set_title("Shadowed Image")
        axs[1,2].axis("off")

        plt.tight_layout()
        plt.show()

    return out_u8, debug_info
