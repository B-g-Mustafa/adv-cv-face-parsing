"""
Copy-Paste Augmentation for rare face classes.

During training, randomly selects a "donor" image from the dataset,
crops regions belonging to rare classes (eyes, eyebrows, earrings, nose, etc.),
and pastes them onto the current training image at the correct location.

This dramatically increases the effective pixel count for small classes,
directly addressing the imbalance that causes poor segmentation of
eyes, eyebrows, earrings, and nose.
"""
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


class CopyPasteAugmentation:
    """
    Copy-Paste augmentation for rare face segmentation classes.

    Strategy:
    - For each training sample, with probability `p`, pick a random donor image
    - Extract connected regions of rare classes from the donor
    - Paste them onto the current image at the donor's original location
      (faces are roughly aligned in CelebAMask-HQ, so spatial location is meaningful)
    - Blend edges with Gaussian feathering so pastes don't look cut-out

    Args:
        masks_dir: Path to training masks directory (for donor lookup).
        images_dir: Path to training images directory.
        rgb_masks: Whether masks are RGB-encoded (True) or grayscale class IDs (False).
        target_classes: Class IDs to copy-paste (rare classes).
        p: Probability of applying copy-paste per sample.
        max_paste_per_image: Maximum number of class regions to paste per sample.
        blend_sigma: Gaussian blur sigma for edge feathering.
        image_size: Target resize dimension.
    """

    # Default rare classes: l_brow(2), r_brow(3), l_eye(4), r_eye(5),
    # eye_g(6), l_ear(7), r_ear(8), ear_r(9), nose(10), u_lip(12), l_lip(13)
    DEFAULT_RARE_CLASSES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]

    def __init__(
        self,
        masks_dir: str,
        images_dir: str,
        rgb_masks: bool = True,
        target_classes: Optional[List[int]] = None,
        p: float = 0.5,
        max_paste_per_image: int = 3,
        blend_sigma: int = 3,
        image_size: int = 512,
    ):
        self.masks_dir = Path(masks_dir)
        self.images_dir = Path(images_dir)
        self.rgb_masks = rgb_masks
        self.target_classes = set(target_classes or self.DEFAULT_RARE_CLASSES)
        self.p = p
        self.max_paste = max_paste_per_image
        self.blend_sigma = blend_sigma
        self.image_size = image_size

        # Build donor file index
        self._donor_files = self._index_donors()

    def _index_donors(self) -> List[Tuple[Path, Path]]:
        """Index all image-mask pairs available as donors."""
        extensions = {".png", ".jpg", ".jpeg"}
        img_stems = {}
        for f in sorted(self.images_dir.iterdir()):
            if f.suffix.lower() in extensions:
                img_stems[f.stem] = f
        mask_stems = {}
        for f in sorted(self.masks_dir.iterdir()):
            if f.suffix.lower() in extensions:
                mask_stems[f.stem] = f

        common = sorted(set(img_stems.keys()) & set(mask_stems.keys()))
        return [(img_stems[s], mask_stems[s]) for s in common]

    def _load_donor_mask(self, mask_path: Path) -> np.ndarray:
        """Load a donor mask as class-index array."""
        if self.rgb_masks:
            from datasets.face_dataset import rgb_mask_to_class_ids
            mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
            return rgb_mask_to_class_ids(mask_rgb)
        else:
            return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.int32)

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply copy-paste augmentation.

        Args:
            image: (H, W, 3) uint8 RGB image.
            mask: (H, W) int class-index mask.

        Returns:
            Augmented (image, mask) tuple.
        """
        if random.random() > self.p:
            return image, mask

        # Pick a random donor
        donor_img_path, donor_mask_path = random.choice(self._donor_files)

        # Load donor image + mask
        donor_img = cv2.imread(str(donor_img_path), cv2.IMREAD_COLOR)
        donor_img = cv2.cvtColor(donor_img, cv2.COLOR_BGR2RGB)
        donor_mask = self._load_donor_mask(donor_mask_path)

        # Resize to match
        h, w = image.shape[:2]
        if donor_img.shape[:2] != (h, w):
            donor_img = cv2.resize(donor_img, (w, h), interpolation=cv2.INTER_LINEAR)
            donor_mask = cv2.resize(
                donor_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

        # Find which rare classes are present in the donor
        donor_classes = set(np.unique(donor_mask).tolist())
        available = list(donor_classes & self.target_classes)

        if not available:
            return image, mask

        # Randomly pick up to max_paste classes to copy
        n_paste = min(self.max_paste, len(available))
        selected = random.sample(available, n_paste)

        # Build combined paste mask for all selected classes
        paste_mask = np.zeros((h, w), dtype=bool)
        for cls_id in selected:
            paste_mask |= (donor_mask == cls_id)

        if paste_mask.sum() < 10:
            return image, mask

        # Feather the edges with Gaussian blur for natural blending
        paste_float = paste_mask.astype(np.float32)
        if self.blend_sigma > 0:
            paste_float = cv2.GaussianBlur(
                paste_float, (0, 0), sigmaX=self.blend_sigma
            )
            # Re-threshold to keep hard class boundaries in the mask
            paste_hard = paste_float > 0.5
        else:
            paste_hard = paste_mask

        # Blend image: use soft mask for smooth edges
        alpha = paste_float[:, :, np.newaxis]  # (H, W, 1)
        image_out = (
            image.astype(np.float32) * (1 - alpha)
            + donor_img.astype(np.float32) * alpha
        ).astype(np.uint8)

        # Paste mask: hard replacement (no blending for labels)
        mask_out = mask.copy()
        mask_out[paste_hard] = donor_mask[paste_hard]

        return image_out, mask_out
