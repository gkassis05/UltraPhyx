# UltraPhyx  
*A physics-grounded ultrasound augmentation engine for ML + robotics.*

UltraPhyx detects the imaging region in an image and simulates real ultrasound artifacts including reverberations, mirror-image artifacts, depth-dependent attenuation, acoustic shadows, speckle variation, gain changes, and more, in a way that is physically meaningful, probe-aware, and compatible with PyTorch pipelines.

The goal is to create robust ultrasound deep learning models that generalize across machines, probes, scanning techniques, and patient body habitus. UltraPhyx aims to provide better data augmentation techniques during the development of deep learning models related to medical imaging.

UltraPhyx is modular, differentiable-compatible, and built with physics + anatomy in mind.

---

## Features

- Fast mask extraction + structure analysis  
- Physics-grounded attenuation + TGC simulation  
- Mirror artifacts along structure axis  
- Complex reverberation patterns with Nakagami modulation  
- Depth-dependent shadowing based on probe geometry  
- Speckle simulation using adjustable Nakagami parameters  
- Gain modifications with realistic feathering  
- Accurate ultrasound probe classification  
- PyTorch integration  
- Configurable augmentation modes:  
  - `"single"` — apply one artifact  
  - `"any"` — probabilistic sampling  
  - `"random_k"` — apply k artifacts  

---

## Installation

```bash
git clone https://github.com/gkassis05/UltraPhyx
cd UltraPhyx
pip install -e .
```

---

## How UltraPhyx Works

### Augmentation Modes

The augmentation logic is controlled by `UltraPhyxConfig`.

**Mode: "single"**  
Select one artifact based on probability weights.

**Mode: "any"**  
Each artifact is independently applied with probability `artifact_probs[name]`.

**Mode: "random_k"**  
Select **k** artifacts without replacement.

---

## How Each Artifact Works

### a. Mirror Artifact (`add_mirror.py`)
Simulates structure reflection when sound bounces between two reflectors.

- Finds longest internal line inside structure  
- Computes reflection plane  
- Applies reflection with depth-limited constraints  
- Uses sigmoid feathering to hide seams  

### b. Reverberations (`add_reverberations.py`)
Generates repeated echoes below structures.

- Computes downward normal  
- Places echoes spaced in depth  
- Modulates with Nakagami speckle  
- Exponential decay per order  

### c. Acoustic Shadow (`add_shadow.py`)
Produces darkening behind strongly attenuating structures.

- Uses probe geometry (fan widening for curvilinear/phased)  
- Depth-based decay  
- Soft Gaussian feathering  
- Nakagami-dependent texture inside shadow  

### d. Depth Attenuation (`add_depth_attenuation.py`)
Models patient body habitus attenuation + TGC sliders.

- Exponential attenuation  
- Randomized slider-like TGC curve  
- Nakagami SNR modulation  

### e. Gain Adjustment (`adjust_gain.py`)
Applies depth-aware gain with soft mask boundaries.

### f. Speckle Modification (`adjust_speckle.py`)
Replaces speckle envelope using Nakagami texture sampling.

---

## Using UltraPhyx in PyTorch Datasets

```python
from torch.utils.data import Dataset
from UltraPhyx.UltraPhyxAugmentor import UltraPhyxAugmentor
from UltraPhyx.config import UltraPhyxConfig
from UltraPhyx.utils import analyze_ultrasound_scan

class USDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs
        self.cfg = UltraPhyxConfig(mode="any", p_global=0.8)  # See Examples folder
        self.augment = UltraPhyxAugmentor(self.cfg)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        analysis = analyze_ultrasound_scan(img)
        return self.augment(img, analysis)

    def __len__(self):
        return len(self.imgs)
```

---
