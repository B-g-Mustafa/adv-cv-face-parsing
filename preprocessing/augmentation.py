"""
Training-time augmentations using albumentations.

Returns an albumentations Compose pipeline that operates on both
image and mask simultaneously (ensures spatial consistency).
"""
import albumentations as A
from typing import Any, Dict, Optional


def get_train_augmentation(cfg: Optional[Dict[str, Any]] = None) -> A.Compose:
    """Build training augmentation pipeline from config."""
    if cfg is None:
        cfg = {}

    if not cfg.get("enabled", True):
        return A.Compose([])

    transforms = [
        A.HorizontalFlip(p=cfg.get("horizontal_flip_p", 0.5)),
        A.Affine(
            translate_percent=cfg.get("shift_limit", 0.05),
            scale=(1 - cfg.get("scale_limit", 0.1), 1 + cfg.get("scale_limit", 0.1)),
            rotate=(-cfg.get("rotate_limit", 15), cfg.get("rotate_limit", 15)),
            border_mode=0,  # constant border
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=cfg.get("brightness_limit", 0.2),
            contrast_limit=cfg.get("contrast_limit", 0.2),
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=cfg.get("hue_shift_limit", 10),
            sat_shift_limit=15,
            val_shift_limit=15,
            p=0.3,
        ),
        A.GaussNoise(
            std_range=(0, cfg.get("gaussian_noise_std", 0.05)),
            p=0.2,
        ),
        A.GaussianBlur(
            blur_limit=cfg.get("gaussian_blur_limit", 3),
            p=0.2,
        ),
        A.CoarseDropout(
            num_holes_range=(1, cfg.get("coarse_dropout_max_holes", 8)),
            hole_height_range=(8, cfg.get("coarse_dropout_max_height", 32)),
            hole_width_range=(8, cfg.get("coarse_dropout_max_width", 32)),
            p=0.3,
        ),
        A.ElasticTransform(
            alpha=cfg.get("elastic_alpha", 30),
            sigma=cfg.get("elastic_sigma", 5),
            p=0.2,
        ),
    ]

    return A.Compose(transforms)


def get_val_augmentation() -> A.Compose:
    """Validation pipeline — no augmentation, just identity."""
    return A.Compose([])
