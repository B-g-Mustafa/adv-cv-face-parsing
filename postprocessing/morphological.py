"""
Morphological post-processing for segmentation masks.

- Per-class morphological opening/closing
- Hole filling
- Connected-component filtering (remove small islands)
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional


class MorphologicalPostProcessor:
    """Apply morphological cleanup to integer segmentation masks."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        if cfg is None:
            cfg = {}

        self.use_morphological = cfg.get("use_morphological", True)
        self.kernel_size = cfg.get("morph_kernel_size", 3)
        self.iterations = cfg.get("morph_iterations", 1)
        self.use_hole_filling = cfg.get("use_hole_filling", True)
        self.use_cc = cfg.get("use_connected_components", True)
        self.cc_min_area = cfg.get("cc_min_area", 100)

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )

    def _morph_per_class(
        self, mask: np.ndarray, num_classes: int
    ) -> np.ndarray:
        """Apply opening then closing per class to remove noise and fill gaps."""
        H, W = mask.shape
        result = np.zeros_like(mask)
        # Process classes in order; later classes overwrite earlier (priority)
        for c in range(num_classes):
            binary = (mask == c).astype(np.uint8)
            if binary.sum() == 0:
                continue
            # Opening (remove small spurious blobs)
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN, self.kernel, iterations=self.iterations
            )
            # Closing (fill small holes / gaps)
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE, self.kernel, iterations=self.iterations
            )
            result[binary > 0] = c
        return result

    def _fill_holes(self, mask: np.ndarray, num_classes: int) -> np.ndarray:
        """Fill internal holes in each class region."""
        result = mask.copy()
        for c in range(num_classes):
            binary = (mask == c).astype(np.uint8)
            if binary.sum() == 0:
                continue
            # Flood fill from corner
            h, w = binary.shape
            flood = binary.copy()
            fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(flood, fill_mask, (0, 0), 1)
            # Holes are where original is 0 but flood didn't reach
            holes = (1 - flood) & (1 - binary)
            result[holes > 0] = c
        return result

    def _filter_connected_components(
        self, mask: np.ndarray, num_classes: int
    ) -> np.ndarray:
        """Remove connected components smaller than min_area."""
        result = mask.copy()
        for c in range(num_classes):
            binary = (mask == c).astype(np.uint8)
            if binary.sum() == 0:
                continue
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < self.cc_min_area:
                    result[labels == label_id] = 0  # reassign to background
        return result

    def __call__(self, mask: np.ndarray, num_classes: int = 19) -> np.ndarray:
        """
        Apply full morphological post-processing pipeline.

        Args:
            mask: (H, W) integer label mask.
            num_classes: Total number of classes.

        Returns:
            Cleaned (H, W) integer label mask.
        """
        if self.use_morphological:
            mask = self._morph_per_class(mask, num_classes)
        if self.use_hole_filling:
            mask = self._fill_holes(mask, num_classes)
        if self.use_cc:
            mask = self._filter_connected_components(mask, num_classes)
        return mask
