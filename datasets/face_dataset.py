"""
Face Parsing Dataset — supports RGB-colored masks.

Loads (image, mask) pairs where masks use distinct RGB colors for each of
the 19 CelebAMask-HQ classes. Converts RGB masks → integer class labels.
Applies augmentation then classical preprocessing.
Computes class-frequency weights for loss rebalancing.
"""
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.transforms import FacePreprocessor
from preprocessing.augmentation import get_train_augmentation, get_val_augmentation


# =====================================================================
# CelebAMask-HQ RGB color → class index mapping
# =====================================================================
# fmt: off
# Each entry: (R, G, B) → class_id
# Derived empirically from the training masks by pixel frequency analysis:
#   background(0)=8.0M, skin(1)=6.7M, hair(17)=7.6M, neck(14)=1.1M, cloth(16)=901K
#   l_brow(2)=533K, u_lip(12)=181K, r_ear(8)=174K, r_brow(3)=123K, eye_g(6)=121K
#   mouth(11)=109K, l_ear(7)=109K, ear_r(9)=93K, nose(10)=83K, hat(18)=75K
#   neck_l(15)=60K, l_eye(4)=57K, r_eye(5)=55K, necklace=1.7K
CELEBAMASK_COLOR_MAP = {
    (0,     0,   0): 0,   # background     — 8.0M px, black
    (204,   0,   0): 1,   # skin           — 6.7M px, dark red
    (76,  153,   0): 2,   # l_brow         — 533K px, olive green
    (204, 204,   0): 3,   # r_brow         — 123K px, dark yellow
    (51,   51, 255): 4,   # l_eye          — 57K px, blue
    (204,   0, 204): 5,   # r_eye          — 55K px, magenta
    (0,   255, 255): 6,   # eye_g          — 121K px, cyan
    (255, 204, 204): 7,   # l_ear          — 109K px, light pink
    (102,  51,   0): 8,   # r_ear          — 174K px, brown
    (255,   0,   0): 9,   # ear_r          — 93K px, bright red
    (102, 204,   0): 10,  # nose           — 83K px, lime green
    (255, 255,   0): 11,  # mouth          — 109K px, yellow
    (0,     0, 153): 12,  # u_lip          — 181K px, dark blue
    (0,   204, 204): 13,  # l_lip          — 60K px, teal
    (255, 153,  51): 14,  # neck           — 1.1M px, orange
    (0,    51,   0): 15,  # neck_l         — 1.7K px, dark green (rare)
    (0,   204,   0): 16,  # cloth          — 901K px, green
    (0,     0, 204): 17,  # hair           — 7.6M px, blue
    (255,  51, 153): 18,  # hat            — 75K px, hot pink
}
# fmt: on

# Build fast lookup table: shape (256, 256, 256) would be too big,
# so we use a hash-based dict with packed keys instead.
def _build_color_lut() -> Dict[int, int]:
    """Build R*65536 + G*256 + B → class_id lookup."""
    lut = {}
    for (r, g, b), cls_id in CELEBAMASK_COLOR_MAP.items():
        key = r * 65536 + g * 256 + b
        lut[key] = cls_id
    return lut

_COLOR_LUT = _build_color_lut()


def rgb_mask_to_class_ids(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB mask to a single-channel class-index mask.

    Args:
        mask_rgb: (H, W, 3) uint8 array in RGB order.

    Returns:
        (H, W) int32 array with class indices 0..18.
    """
    H, W, _ = mask_rgb.shape
    # Pack RGB into single int for fast lookup
    packed = (
        mask_rgb[:, :, 0].astype(np.int32) * 65536
        + mask_rgb[:, :, 1].astype(np.int32) * 256
        + mask_rgb[:, :, 2].astype(np.int32)
    )
    result = np.zeros((H, W), dtype=np.int32)
    for key, cls_id in _COLOR_LUT.items():
        result[packed == key] = cls_id
    return result


def class_ids_to_rgb_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert a class-index mask back to an RGB visualization.

    Args:
        mask: (H, W) integer array with class indices 0..18.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    # Build reverse map: class_id → (R, G, B)
    id_to_color = {v: k for k, v in CELEBAMASK_COLOR_MAP.items()}
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, (r, g, b) in id_to_color.items():
        rgb[mask == cls_id] = [r, g, b]
    return rgb


# =====================================================================
# Dataset class
# =====================================================================
class FaceParsingDataset(Dataset):
    """
    Dataset for 19-class face semantic segmentation.

    Supports both:
    - RGB-colored masks (auto-detected, converted to class indices)
    - Grayscale masks (class indices directly)

    Directory layout expected:
        images_dir/
            0001.png / 0001.jpg
        masks_dir/
            0001.png
    """

    IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        preprocessor: FacePreprocessor,
        augmentation: Optional[Any] = None,
        copy_paste: Optional[Any] = None,
        image_size: int = 512,
        num_classes: int = 19,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.copy_paste = copy_paste
        self.image_size = image_size
        self.num_classes = num_classes

        # Gather sorted file list (match by stem)
        self.samples: List[Tuple[Path, Path]] = self._gather_samples()

        # Auto-detect mask format from first sample
        self._rgb_masks = self._detect_mask_format()

    def _gather_samples(self) -> List[Tuple[Path, Path]]:
        img_stems = {}
        for f in sorted(self.images_dir.iterdir()):
            if f.suffix.lower() in self.IMG_EXTENSIONS:
                img_stems[f.stem] = f

        mask_stems = {}
        for f in sorted(self.masks_dir.iterdir()):
            if f.suffix.lower() in self.IMG_EXTENSIONS:
                mask_stems[f.stem] = f

        common = sorted(set(img_stems.keys()) & set(mask_stems.keys()))
        if len(common) == 0:
            raise FileNotFoundError(
                f"No matching image-mask pairs found in {self.images_dir} / {self.masks_dir}"
            )
        return [(img_stems[s], mask_stems[s]) for s in common]

    def _detect_mask_format(self) -> bool:
        """Check first mask to determine if RGB or grayscale."""
        _, mask_path = self.samples[0]
        test = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        is_rgb = test is not None and test.ndim == 3
        fmt = "RGB-colored" if is_rgb else "grayscale (class indices)"
        print(f"  Mask format detected: {fmt}")
        return is_rgb

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """Load mask and convert to class indices (H, W) int."""
        if self._rgb_masks:
            mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
            mask = rgb_mask_to_class_ids(mask_rgb)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        # Load image (BGR → RGB) and mask (→ class indices)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = self._load_mask(mask_path)

        # Resize to target size
        if img.shape[:2] != (self.image_size, self.image_size):
            img = cv2.resize(
                img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )
        if mask.shape[:2] != (self.image_size, self.image_size):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

        # Clip mask to valid class range
        mask = np.clip(mask, 0, self.num_classes - 1).astype(np.uint8)

        # Copy-Paste augmentation (before geometric augmentation)
        if self.copy_paste is not None:
            img, mask = self.copy_paste(img, mask)

        # Augmentation (operates on uint8 image + mask)
        if self.augmentation is not None:
            augmented = self.augmentation(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Classical preprocessing (CLAHE, bilateral, Sobel, normalize)
        img = self.preprocessor(img)  # float32 (H, W, C)

        # To tensors: (C, H, W) float32 image, (H, W) int64 mask
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64)).long()

        return {"image": img_tensor, "mask": mask_tensor}

    # -----------------------------------------------------------------
    # Class frequency computation for loss weighting
    # -----------------------------------------------------------------
    def compute_class_weights(self, method: str = "median_freq") -> torch.Tensor:
        """
        Compute class weights based on pixel frequency across the dataset.

        Args:
            method: 'median_freq' or 'inverse_freq'.

        Returns:
            Tensor of shape (num_classes,) with per-class weights.
        """
        print("  Computing class weights...")
        counts = np.zeros(self.num_classes, dtype=np.float64)
        for _, mask_path in self.samples:
            mask = self._load_mask(mask_path)
            if mask.shape[:2] != (self.image_size, self.image_size):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)
            mask = np.clip(mask, 0, self.num_classes - 1)
            for c in range(self.num_classes):
                counts[c] += (mask == c).sum()

        # Avoid division by zero
        counts = np.maximum(counts, 1.0)

        if method == "inverse_freq":
            weights = 1.0 / counts
            weights = weights / weights.sum() * self.num_classes
        else:  # median_freq
            freq = counts / counts.sum()
            median_freq = np.median(freq[freq > 0])
            weights = median_freq / freq

        print(f"  Class weights computed. Range: [{weights.min():.4f}, {weights.max():.4f}]")
        return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    cfg: Dict[str, Any],
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.Tensor]:
    """
    Build train and val DataLoaders from config.

    Returns:
        (train_loader, val_loader, class_weights)
    """
    preproc_cfg = cfg.get("preprocessing", {})
    aug_cfg = cfg.get("augmentation", {})
    train_cfg = cfg.get("training", {})

    preprocessor = FacePreprocessor(preproc_cfg)
    train_aug = get_train_augmentation(aug_cfg)
    val_aug = get_val_augmentation()

    # Copy-Paste augmentation (training only)
    copy_paste = None
    cp_cfg = cfg.get("copy_paste", {})
    if cp_cfg.get("enabled", False):
        from preprocessing.copy_paste import CopyPasteAugmentation
        copy_paste = CopyPasteAugmentation(
            masks_dir=cfg["data"]["train_masks"],
            images_dir=cfg["data"]["train_images"],
            rgb_masks=True,
            target_classes=cp_cfg.get("target_classes", None),
            p=cp_cfg.get("probability", 0.5),
            max_paste_per_image=cp_cfg.get("max_paste_per_image", 3),
            blend_sigma=cp_cfg.get("blend_sigma", 3),
            image_size=cfg.get("image_size", 512),
        )
        print(f"  Copy-Paste augmentation: ON (p={cp_cfg.get('probability', 0.5)})")

    train_ds = FaceParsingDataset(
        images_dir=cfg["data"]["train_images"],
        masks_dir=cfg["data"]["train_masks"],
        preprocessor=preprocessor,
        augmentation=train_aug,
        copy_paste=copy_paste,
        image_size=cfg.get("image_size", 512),
        num_classes=cfg.get("num_classes", 19),
    )
    val_ds = FaceParsingDataset(
        images_dir=cfg["data"]["val_images"],
        masks_dir=cfg["data"]["val_masks"],
        preprocessor=preprocessor,
        augmentation=val_aug,
        image_size=cfg.get("image_size", 512),
        num_classes=cfg.get("num_classes", 19),
    )

    class_weights = train_ds.compute_class_weights()

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        drop_last=False,
    )

    return train_loader, val_loader, class_weights
