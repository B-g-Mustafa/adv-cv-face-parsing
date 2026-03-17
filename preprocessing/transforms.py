"""
Classical preprocessing transforms for face images.

Each transform is individually togglable via config to support ablation studies.
Pipeline: CLAHE → Bilateral Filter → (optional Sobel edge channel) → Normalize.
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple


class FacePreprocessor:
    """Configurable classical image preprocessing pipeline."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        # Defaults (overridden by cfg)
        self.use_clahe = True
        self.clahe_clip = 2.0
        self.clahe_grid = 8
        self.use_bilateral = True
        self.bilateral_d = 5
        self.bilateral_sc = 50
        self.bilateral_ss = 50
        self.use_sobel = True
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if cfg is not None:
            self.use_clahe = cfg.get("use_clahe", self.use_clahe)
            self.clahe_clip = cfg.get("clahe_clip_limit", self.clahe_clip)
            self.clahe_grid = cfg.get("clahe_grid_size", self.clahe_grid)
            self.use_bilateral = cfg.get("use_bilateral", self.use_bilateral)
            self.bilateral_d = cfg.get("bilateral_d", self.bilateral_d)
            self.bilateral_sc = cfg.get("bilateral_sigma_color", self.bilateral_sc)
            self.bilateral_ss = cfg.get("bilateral_sigma_space", self.bilateral_ss)
            self.use_sobel = cfg.get("use_sobel_edge", self.use_sobel)
            self.mean = np.array(cfg.get("normalize_mean", self.mean), dtype=np.float32)
            self.std = np.array(cfg.get("normalize_std", self.std), dtype=np.float32)

    # ----------------------------------------------------------
    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the L channel of a LAB-converted image."""
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_grid, self.clahe_grid),
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def apply_bilateral(self, img: np.ndarray) -> np.ndarray:
        """Edge-preserving bilateral filter."""
        return cv2.bilateralFilter(
            img, self.bilateral_d, self.bilateral_sc, self.bilateral_ss
        )

    def compute_sobel_edge(self, img: np.ndarray) -> np.ndarray:
        """Compute Sobel gradient magnitude on grayscale, normalized to [0, 1]."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx ** 2 + sy ** 2)
        mag = np.clip(mag / (mag.max() + 1e-8), 0, 1).astype(np.float32)
        return mag

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Scale to [0,1] then apply ImageNet-style normalization."""
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return img

    # ----------------------------------------------------------
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline.

        Args:
            img: uint8 RGB image, shape (H, W, 3).

        Returns:
            float32 array of shape (H, W, C) where C is 3 or 4.
        """
        if self.use_clahe:
            img = self.apply_clahe(img)

        if self.use_bilateral:
            img = self.apply_bilateral(img)

        # Compute Sobel edge map *before* normalization (needs uint8 / pixel range)
        edge = None
        if self.use_sobel:
            edge = self.compute_sobel_edge(img)  # shape (H, W), float32 [0,1]

        # Normalize RGB channels
        img = self.normalize(img)  # (H, W, 3) float32

        # Concatenate edge channel if requested
        if edge is not None:
            img = np.concatenate([img, edge[:, :, None]], axis=-1)  # (H, W, 4)

        return img

    @property
    def out_channels(self) -> int:
        """Number of output channels (3 RGB or 4 RGB+edge)."""
        return 4 if self.use_sobel else 3
