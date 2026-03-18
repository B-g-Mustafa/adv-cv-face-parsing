"""
Post-processing pipeline orchestrator.

Chains: morphological cleanup → CRF refinement (if enabled).
Each step is individually togglable via config.
"""
import numpy as np
from typing import Dict, Any, Optional

from .morphological import MorphologicalPostProcessor
from .crf_refine import apply_crf
from .heuristic import SpatialHeuristicFixer


class PostProcessor:
    """Orchestrates all post-processing steps."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        if cfg is None:
            cfg = {}
        self.cfg = cfg
        self.use_crf = cfg.get("use_crf", False)
        self.use_heuristic = cfg.get("use_heuristic", True)  # Hack for Left/Right
        self.morph = MorphologicalPostProcessor(cfg)
        self.heuristic = SpatialHeuristicFixer()

    def __call__(
        self,
        probs: np.ndarray,
        image: Optional[np.ndarray] = None,
        num_classes: int = 19,
    ) -> np.ndarray:
        """
        Apply post-processing pipeline.

        Args:
            probs: Softmax probabilities (C, H, W) float32, or
                   integer mask (H, W) if CRF is disabled.
            image: Original RGB image (H, W, 3) uint8 — needed for CRF.
            num_classes: Number of classes.

        Returns:
            Refined integer label mask (H, W).
        """
        # Step 1: CRF refinement (operates on probabilities)
        if self.use_crf and image is not None and probs.ndim == 3:
            mask = apply_crf(image, probs, self.cfg)
        elif probs.ndim == 3:
            mask = probs.argmax(axis=0).astype(np.int32)
        else:
            mask = probs.astype(np.int32)

        # Step 2: Morphological cleanup (operates on integer mask)
        mask = self.morph(mask, num_classes)

        # Step 3: Spatial Heuristic Fixer (Left/Right hack)
        if self.use_heuristic:
            mask = self.heuristic(mask)

        return mask
