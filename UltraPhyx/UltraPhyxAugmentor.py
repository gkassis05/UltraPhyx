import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Callable

from .add_mirror import add_mirror
from .add_shadow import add_shadow
from .add_reverberations import add_reverberations
from .adjust_gain import adjust_gain
from .adjust_speckle import adjust_speckle
from .add_depth_attenuation import add_depth_attenuation
from .utils import _ensure_uint8, sample_param

from .config import UltraPhyxConfig

# ============================================================
# AUGMENTOR (supports modes)
# ============================================================
class UltraPhyxAugmentor:
    def __init__(self, config: UltraPhyxConfig):
        self.cfg = config
        np.random.seed(config.seed)

        self.ops = {
            "mirror": add_mirror,
            "shadow": add_shadow,
            "reverb": add_reverberations,
            "gain": adjust_gain,
            "speckle": adjust_speckle,
            "depth_atten": add_depth_attenuation,
        }

    # -----------------------------------------
    # tensor/np helpers
    # -----------------------------------------
    def _to_numpy(self, img):
        if isinstance(img, torch.Tensor):
            if img.ndim == 3:
                return np.ascontiguousarray(img.permute(1,2,0).cpu().numpy())
            return img.cpu().numpy()
        return img

    def _to_tensor_like(self, np_img, template):
        if isinstance(template, torch.Tensor):
            out = torch.from_numpy(np_img)
            if out.ndim == 2:
                out = out.unsqueeze(-1)
            return out.permute(2,0,1).float() / 255.0
        return np_img

    # -----------------------------------------
    # SAMPLE PARAMETERS FOR ONE ARTIFACT
    # -----------------------------------------
    def sample_artifact_parameters(self, artifact_name):
        cfg = self.cfg.artifact_configs.get(artifact_name)
        if cfg is None:
            return {}
        return {k: sample_param(v) for k, v in cfg.items()}

    # -----------------------------------------
    # CHOOSE ARTIFACTS BASED ON MODE
    # -----------------------------------------
    def choose_artifacts(self) -> List[str]:
        names = [k for k, v in self.cfg.artifact_configs.items() if v is not None]
        if len(names) == 0:
            return []

        probs = np.array([self.cfg.artifact_probs.get(n, 0.1667) for n in names])
        probs = probs / probs.sum()
        mode = self.cfg.mode

        # SINGLE
        if mode == "single":
            return [np.random.choice(names, p=probs)]

        # ANY
        if mode == "any":
            return [n for n, p in zip(names, probs) if np.random.rand() < p]

        # RANDOM_K
        if mode == "random_k":
            k = min(self.cfg.random_k, len(names))  # FIXED BUG
            return list(np.random.choice(names, size=k, replace=False, p=probs))

        return []

    # -----------------------------------------
    # MAIN AUGMENTATION CALL
    # -----------------------------------------
    def __call__(self, img, analysis, show_choices: bool = False):
        """
        show_choices=True, prints which artifacts are applied
        """
        img_np = self._to_numpy(img)
        img_np = _ensure_uint8(img_np)
        out = img_np.copy()

        # global probability
        if np.random.rand() > self.cfg.p_global:
            if show_choices:
                print("No augmentation applied (p_global check failed).")
            return self._to_tensor_like(out, img)

        selected = self.choose_artifacts()

        if show_choices:
            if len(selected) == 0:
                print("No artifacts selected.")
            else:
                print(f"Applying artifacts: {', '.join(selected)}")

        if len(selected) == 0:
            return self._to_tensor_like(out, img)

        # apply artifacts
        for name in selected:
            op = self.ops[name]
            params = self.sample_artifact_parameters(name)
            out, _ = op(analysis, out, **params)

        return self._to_tensor_like(out, img)
