"""
Face Parsing Dataset.

Loads (image, mask) pairs, applies augmentation then classical preprocessing.
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


class FaceParsingDataset(Dataset):
    """
    Dataset for 19-class face semantic segmentation.

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
        image_size: int = 512,
        num_classes: int = 19,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.image_size = image_size
        self.num_classes = num_classes

        # Gather sorted file list (match by stem)
        self.samples: List[Tuple[Path, Path]] = self._gather_samples()

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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        # Load image (BGR → RGB) and mask (grayscale)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Resize to target size
        if img.shape[:2] != (self.image_size, self.image_size):
            img = cv2.resize(
                img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )
        if mask.shape[:2] != (self.image_size, self.image_size):
            mask = cv2.resize(
                mask,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )

        # Clip mask to valid class range
        mask = np.clip(mask, 0, self.num_classes - 1)

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
        counts = np.zeros(self.num_classes, dtype=np.float64)
        for _, mask_path in self.samples:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask.shape[:2] != (self.image_size, self.image_size):
                mask = cv2.resize(
                    mask,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_NEAREST,
                )
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

    train_ds = FaceParsingDataset(
        images_dir=cfg["data"]["train_images"],
        masks_dir=cfg["data"]["train_masks"],
        preprocessor=preprocessor,
        augmentation=train_aug,
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
