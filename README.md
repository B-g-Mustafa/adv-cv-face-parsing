# Face Semantic Parsing — Lightweight 19-Class Pipeline

A hybrid classical-preprocessing + lightweight-DL + classical-postprocessing pipeline for parsing faces into 19 semantic classes under a strict **<1.7M parameter budget**, trained from scratch on only 1,000 image–mask pairs.

---

## Table of Contents
- [Pipeline Overview](#pipeline-overview)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Post-Processing](#post-processing)
- [Quick Start](#quick-start)
- [Ablation Toggles](#ablation-toggles)

---

## Pipeline Overview

```
Input RGB (512×512)
    │
    ▼
┌───────────────────────────────────┐
│  CLASSICAL PREPROCESSING          │
│  CLAHE → Bilateral → Sobel Edge  │
│  → Normalize → 4-channel tensor  │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  LIGHTWEIGHT DL MODEL             │
│  AttentionLiteUNet (~1.68M params)│
│  DS-ResBlocks + ASPP + Att. Gates │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  INFERENCE BOOSTING               │
│  TTA (flip/rotation averaging)    │
│  + Checkpoint Ensemble (top-K)    │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  CLASSICAL POST-PROCESSING        │
│  Morph. Open/Close → Hole Fill    │
│  → CC Filtering → (Optional CRF) │
└───────────────┬───────────────────┘
                │
                ▼
     19-class mask (512×512)
```

---

## Preprocessing

> **Philosophy**: Delegate low-level vision tasks (edge detection, noise removal, contrast normalization) to deterministic classical algorithms so the shallow neural network can focus entirely on high-level semantic understanding.

All preprocessing is implemented in `preprocessing/transforms.py` and each step is **independently togglable** via `configs/default.yaml` for ablation studies.

### 1. CLAHE — Contrast Limited Adaptive Histogram Equalization

**What**: Enhances local contrast by dividing the image into tiles and equalizing the histogram within each tile, with a clip limit to prevent noise amplification.

**Why**: Face images suffer from uneven illumination — shadows under the chin, overexposed foreheads, poorly lit ears. CLAHE normalizes these intensity variations without the global flattening problem of regular histogram equalization.

**How it works in the pipeline**:
1. Convert RGB → LAB color space (separates luminance from color)
2. Apply CLAHE only to the L (lightness) channel
3. Convert back to RGB

This preserves natural color relationships while normalizing brightness.

```yaml
# configs/default.yaml
preprocessing:
  use_clahe: true
  clahe_clip_limit: 2.0    # higher = more contrast enhancement
  clahe_grid_size: 8        # tile grid, 8×8 = 64 local regions
```

### 2. Bilateral Filtering — Edge-Preserving Denoising

**What**: Smooths flat regions while preserving sharp edges by considering both spatial distance *and* intensity similarity when averaging pixels.

**Why**: Unlike Gaussian blur which blurs everything uniformly (destroying hair/skin boundaries), bilateral filtering only smooths regions of similar intensity. This removes camera sensor noise and fine texture while keeping the exact boundaries the model needs to segment.

**How it works**: Each pixel's new value is a weighted average of neighbors, where the weight depends on:
- **Spatial closeness** (`sigma_space`): nearby pixels contribute more
- **Intensity similarity** (`sigma_color`): similar-looking pixels contribute more

Pixels on the other side of an edge (different intensity) contribute almost nothing, preserving the edge.

```yaml
preprocessing:
  use_bilateral: true
  bilateral_d: 5                # neighborhood diameter
  bilateral_sigma_color: 50     # color similarity range
  bilateral_sigma_space: 50     # spatial proximity range
```

### 3. Sobel Edge Injection — Explicit Boundary Channel

**What**: Computes the image gradient magnitude using a Sobel operator and appends it as a 4th input channel alongside RGB.

**Why**: This is directly inspired by the **ARU-Net** research, which showed that feeding explicit edge maps to a shallow network spares it from wasting its limited convolutional layers learning basic Gabor-like edge filters. For faces, this is critical: eyebrow boundaries, lip contours, ear outlines, and hair silhouettes all have strong gradient signatures that Sobel captures perfectly with zero learnable parameters.

**How it works**:
1. Convert to grayscale
2. Compute horizontal gradient `Gx` and vertical gradient `Gy` using 3×3 Sobel kernels
3. Magnitude = `√(Gx² + Gy²)`, normalized to [0, 1]
4. Concatenated as the 4th channel → model input becomes `(4, 512, 512)` instead of `(3, 512, 512)`

```yaml
preprocessing:
  use_sobel_edge: true
# When Sobel is enabled, set model input channels to 4:
model:
  in_channels: 4
```

### 4. ImageNet-Style Normalization

**What**: Scales pixel values to [0, 1] then applies per-channel mean subtraction and standard deviation division using ImageNet statistics.

**Why**: Even though the model is trained from scratch (not pretrained on ImageNet), these statistics provide a reasonable prior for centering and scaling natural images. They prevent initial gradient magnitudes from being dominated by raw pixel scale (0–255).

```yaml
preprocessing:
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std:  [0.229, 0.224, 0.225]
```

### Preprocessing Order

The order matters:
1. **CLAHE first** — corrects illumination before any filtering
2. **Bilateral next** — denoises the contrast-normalized image
3. **Sobel computed** — extracts edges from the cleaned image (better edges after denoising)
4. **Normalize last** — scales the final RGB channels
5. **Concatenate** — RGB (3ch) + Sobel edge (1ch) = 4-channel input

---

## Training

### Model: AttentionLiteUNet

A 4-stage encoder-decoder with three key parameter-saving strategies:

| Strategy | Description | Param Savings |
|---|---|---|
| **Depthwise Separable Convolutions** | Split spatial and channel mixing into two steps (depthwise + 1×1 pointwise) | ~8× fewer params than standard conv |
| **Attention Gates** | Learned spatial masks on skip connections that suppress irrelevant background regions | Focus limited capacity on face features |
| **ASPP Bottleneck** | Parallel dilated convolutions at rates 6, 12, 18 + global average pooling | Multi-scale context without deeper layers |

Channel progression: `32 → 64 → 128 → 256` (encoder) → `256` (ASPP) → `128 → 64 → 32` (decoder) → `19` (head)

### Loss Function

Three complementary terms handle different aspects of the 19-class imbalance:

| Term | Formula | Purpose |
|---|---|---|
| **Weighted Cross-Entropy** | `−wc · log(pc)` | Per-class weights inversely proportional to pixel frequency. Upweights rare classes like earrings and necklaces. |
| **Dice Loss** | `1 − 2|P∩G| / (|P|+|G|)` | Directly optimizes overlap, inherently handles imbalance since each class contributes equally regardless of size. |
| **Boundary Loss** | `∑ p(x) · φ(x)` | Uses distance transforms: penalizes predictions that are spatially far from true boundaries. Helps with fine structures (inner mouth, eyebrows). |

Combined: `Total = α·CE + β·Dice + γ·Boundary`

```yaml
loss:
  ce_weight: 1.0
  dice_weight: 1.0
  boundary_weight: 0.5        # set to 0 to disable
  label_smoothing: 0.05       # regularization for small dataset
  use_class_weights: true      # auto-computed from training set
```

### Class Weight Computation

Uses **median frequency balancing**: `weight_c = median(freq) / freq_c`

This means if the background has frequency 0.40 and earrings have frequency 0.0001, earrings get a weight ~4000× higher than background in the CE loss.

### Training Strategy

| Setting | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Weight decay decoupled from gradient updates; better generalization |
| Scheduler | Cosine Annealing Warm Restarts (T₀=50) | Periodic LR resets escape local minima; restart provides fresh exploration |
| Mixed Precision | AMP (float16 forward, float32 gradients) | 2× faster training, lower VRAM, no accuracy loss |
| Gradient Clipping | max_norm=1.0 | Prevents exploding gradients from rare-class samples |
| Early Stopping | patience=30 | Stop if val F1 doesn't improve for 30 epochs |
| Augmentation | Flip, rotate±15°, scale, color jitter, gaussian noise, coarse dropout, elastic transform | Maximizes effective dataset size from only 850 training samples |

---

## Post-Processing

> **Philosophy**: The shallow network produces coarse, blob-like predictions. Classical algorithms enforce spatial logic, remove artifacts, and sharpen boundaries with zero additional parameters.

All steps are in `postprocessing/` and individually togglable.

### 1. Morphological Opening & Closing

**Opening** (erosion → dilation): Removes small false-positive blobs — e.g., isolated pixels incorrectly labeled as "earring" in a hair region.

**Closing** (dilation → erosion): Fills small holes and gaps — e.g., a few pixels inside a correctly predicted "skin" region that were incorrectly labeled as "background".

Applied **per-class** with an elliptical kernel, so each class is cleaned independently.

### 2. Hole Filling

Uses flood fill from the image border to identify internal holes in each class region. Any pixel surrounded entirely by a class but labeled differently gets reassigned. Ensures topological integrity — the skin region won't have stray background holes.

### 3. Connected Component Filtering

For each class, finds all spatially connected groups of pixels. Components with area smaller than `cc_min_area` (default: 100 pixels) are removed (reassigned to background). Eliminates tiny, implausible detections — there shouldn't be a 15-pixel "hat" island floating in the middle of the forehead.

### 4. Dense CRF Refinement (Optional)

The Conditional Random Field models the segmentation as a probabilistic graphical model:
- **Unary potentials**: the raw softmax probabilities from the network
- **Pairwise potentials**: penalize label disagreements between nearby pixels that share similar colors

Effect: Snaps blurry model boundaries to the true image edges. If two adjacent pixels have the same RGB color but different labels, the CRF nudges them toward agreement.

> CRF is **disabled by default** (`use_crf: false`) because it adds ~200ms per image. Enable it for final competition submissions.

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Verify preprocessing works (no training needed)
python verify_preprocessing.py --image path/to/any/face.jpg

# 3. Update data paths in configs/default.yaml, then train
python train.py

# 4. Run inference
python inference.py --input data/test/images --output outputs/predictions

# 5. Evaluate
python evaluate.py --preds outputs/predictions --gt data/val/masks
```

---

## Ablation Toggles

Run ablation studies by changing booleans in `configs/default.yaml`:

```yaml
# Disable all preprocessing (raw RGB baseline)
preprocessing:
  use_clahe: false
  use_bilateral: false
  use_sobel_edge: false
model:
  in_channels: 3   # no Sobel = 3 channels

# Disable all post-processing
postprocessing:
  use_morphological: false
  use_hole_filling: false
  use_connected_components: false
  use_crf: false

# Disable inference boosting
tta:
  enabled: false
ensemble:
  enabled: false
```
