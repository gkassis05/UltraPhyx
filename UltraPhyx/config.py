# ============================================================
# CONFIG
# ============================================================
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal

@dataclass
class UltraPhyxConfig:
    """
    mode:
        "single"   -> choose exactly ONE artifact per image
        "random_k" -> choose k artifacts (k given below)
        "any"      -> each artifact independently sampled by its probability

    p_global:
        probability that ANY augmentation is applied at all
        (if fails → return original image)

    artifact_probs:
        weights or probabilities for each artifact
        (used for "single" or when selecting k in "random_k")

    artifact_configs:
        parameter ranges for each artifact type
    """

    # How many artifacts to apply per sample?
    mode: str = "single"                 # "single", "any", or "random_k"
    k: int = 2                           # only used if mode == "random_k"

    # Overall chance of applying any augmentation
    p_global: float = 1.0

    # Probability weights for choosing artifacts
    artifact_probs: Dict[str, float] = field(default_factory=lambda: {
        "mirror": 0.1667,
        "shadow": 0.1667,
        "reverb": 0.1667,
        "gain": 0.1667,
        "speckle": 0.1667,
        "depth_atten": 0.1667,
    })

    # Configs for each artifact
    artifact_configs: Dict[str, Optional[Dict[str, Any]]] = field(default_factory=lambda: {
        "mirror": None,
        "shadow": None,
        "reverb": None,
        "gain": None,
        "speckle": None,
        "depth_atten": None,
    })

    seed: Optional[int] = None

