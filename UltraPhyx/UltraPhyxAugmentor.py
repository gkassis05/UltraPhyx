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
        
        # 5B: Use a local, modern random generator
        self.rng = np.random.default_rng(config.seed)

        self.ops = {
            "mirror": add_mirror,
            "shadow": add_shadow,
            "reverb": add_reverberations,
            "gain": adjust_gain,
            "speckle": adjust_speckle,
            "depth_atten": add_depth_attenuation,
        }

        # -----------------------------------------
        # INITIALIZATION VALIDATION
        # -----------------------------------------
        valid_modes = ["single", "any", "random_k"]
        if self.cfg.mode not in valid_modes:
            raise ValueError(f"Unknown mode '{self.cfg.mode}'. Supported modes are: {valid_modes}")

        if not (0.0 <= self.cfg.p_global <= 1.0):
            raise ValueError(f"p_global must be between 0.0 and 1.0, got {self.cfg.p_global}")

        if not isinstance(self.cfg.k, int) or self.cfg.k < 0:
            raise ValueError(f"k must be a non-negative integer, got {self.cfg.k}")

        for name in (self.cfg.artifact_configs or {}).keys():
            if name not in self.ops:
                raise ValueError(f"Unknown artifact '{name}'. Supported artifacts: {list(self.ops.keys())}")

    # -----------------------------------------
    # tensor/np helpers
    # -----------------------------------------
    def _to_numpy(self, img):
        if isinstance(img, torch.Tensor):
            if img.min() < 0:
                raise ValueError("Input tensor has negative values. UltraPhyx must run before normalization.")
            if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[0] != 1):
                raise ValueError(f"Unsupported tensor shape {img.shape}. Expected HxW or 1xHxW.")
            
            arr = img.detach().cpu().numpy()
            return arr[0] if img.ndim == 3 else arr
        else:
            if img.min() < 0:
                raise ValueError("Input array has negative values. UltraPhyx must run before normalization.")
            if img.ndim != 2:
                raise ValueError(f"Unsupported numpy shape {img.shape}. Expected HxW.")
            return img

    def _to_tensor_like(self, np_img, template):
        if isinstance(template, torch.Tensor):
            out = torch.from_numpy(np_img).to(device=template.device)
            if template.is_floating_point():
                out = out.to(template.dtype) / 255.0
            else:
                out = out.to(template.dtype)
            
            if template.ndim == 3:
                out = out.unsqueeze(0)
            return out
        else:
            if np.issubdtype(template.dtype, np.floating):
                return (np_img.astype(template.dtype) / 255.0)
            return np_img.astype(template.dtype)

    # -----------------------------------------
    # SAMPLE PARAMETERS FOR ONE ARTIFACT
    # -----------------------------------------
    def sample_artifact_parameters(self, artifact_name):
        cfg = self.cfg.artifact_configs.get(artifact_name)
        if cfg is None:
            return {}
        # 5B: Pass the local rng to sample_param
        return {k: sample_param(v, rng=self.rng) for k, v in cfg.items()}

    # -----------------------------------------
    # CHOOSE ARTIFACTS BASED ON MODE
    # -----------------------------------------
    def choose_artifacts(self) -> List[str]:
        names = [k for k, v in self.cfg.artifact_configs.items() if v is not None]
        if len(names) == 0:
            return []

        weights = np.asarray([
            self.cfg.artifact_probs.get(name, 0.1667)
            for name in names
        ], dtype=float)

        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("Artifact probabilities/weights must be finite and nonnegative.")

        # ANY (Independent sampling)
        if self.cfg.mode == "any":
            if np.any(weights > 1):
                raise ValueError("'any' mode requires probabilities between 0 and 1.")
            return [
                name for name, probability in zip(names, weights)
                if self.rng.random() < probability
            ]

        # Only categorical selection modes normalize weights.
        positive = weights > 0
        if not positive.any():
            return []

        names = np.asarray(names, dtype=object)[positive]
        weights = weights[positive]
        probabilities = weights / weights.sum()

        # SINGLE
        if self.cfg.mode == "single":
            return [self.rng.choice(names, p=probabilities)]

        # RANDOM_K
        if self.cfg.mode == "random_k":
            k = min(self.cfg.k, len(names))
            return list(self.rng.choice(names, size=k, replace=False, p=probabilities))

        raise ValueError(f"Mode '{self.cfg.mode}' is valid but not handled in selection.")

    # -----------------------------------------
    # 5C: SAMPLE AN AUGMENTATION PLAN
    # -----------------------------------------
    def sample_plan(self):
        if self.rng.random() >= self.cfg.p_global:
            return []
        
        plan = []
        for name in self.choose_artifacts():
            params = self.sample_artifact_parameters(name)
            
            # 5B: Set specific seeds for the operator to handle None properly
            if params.get("seed") is None:
                params["seed"] = int(self.rng.integers(0, 2**32))
            params.setdefault("show_debug", False)
            
            plan.append((name, params))
            
        return plan

    # -----------------------------------------
    # MAIN AUGMENTATION CALL
    # -----------------------------------------
    def __call__(self, img, analysis, show_choices: bool = False, plan: Optional[List] = None):
        """
        show_choices=True, prints which artifacts are applied.
        plan lets you pass a pre-sampled augmentation plan (for sequences).
        """
        if plan is None:
            plan = self.sample_plan()

        if not plan:
            if show_choices:
                print("No artifacts selected/applied.")
            return img.clone() if isinstance(img, torch.Tensor) else img.copy()

        if show_choices:
            print(f"Selected artifacts: {', '.join([n for n, p in plan])}")

        img_np = self._to_numpy(img)
        img_np = _ensure_uint8(img_np)
        out = img_np.copy()

        applied_success = []
        applied_failed = []

        # apply artifacts loop with passed-down parameters
        for name, params in plan:
            before = out.copy()
            out, info = self.ops[name](analysis, out, **params)
            
            changed = not np.array_equal(before, out)
            if changed:
                applied_success.append(name)
            else:
                reason = info.get("reason", "no visible change")
                applied_failed.append(f"{name} ({reason})")

        if show_choices:
            if applied_success:
                print(f"Successfully applied: {', '.join(applied_success)}")
            if applied_failed:
                print(f"Failed to apply: {', '.join(applied_failed)}")

        # Fast exit identity if no pixels were actually changed
        if not applied_success:
            return img.clone() if isinstance(img, torch.Tensor) else img.copy()

        return self._to_tensor_like(out, img)
