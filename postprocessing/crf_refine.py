"""
Dense CRF-based segmentation refinement.

Uses pydensecrf to snap blurry neural network boundaries to true image edges
via pairwise Gaussian and bilateral potentials.
"""
import numpy as np
from typing import Dict, Any, Optional


def apply_crf(
    image: np.ndarray,
    probs: np.ndarray,
    cfg: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Refine segmentation probabilities using a fully-connected CRF.

    Args:
        image: Original RGB image, uint8, shape (H, W, 3).
        probs: Softmax probabilities, float32, shape (C, H, W).
        cfg: CRF configuration parameters.

    Returns:
        Refined integer label mask of shape (H, W).
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        # Fallback: just argmax if pydensecrf not installed
        return probs.argmax(axis=0).astype(np.int32)

    if cfg is None:
        cfg = {}

    n_classes, H, W = probs.shape

    # Unary potentials from softmax
    unary = unary_from_softmax(probs)

    d = dcrf.DenseCRF2D(W, H, n_classes)
    d.setUnaryEnergy(unary)

    # Pairwise Gaussian (spatial smoothness)
    d.addPairwiseGaussian(
        sxy=cfg.get("crf_gaussian_sxy", 3),
        compat=cfg.get("crf_gaussian_compat", 3),
    )

    # Pairwise bilateral (appearance-based; snaps to image edges)
    d.addPairwiseBilateral(
        sxy=cfg.get("crf_bilateral_sxy", 60),
        srgb=cfg.get("crf_bilateral_srgb", 10),
        rgbim=image.copy(order="C"),
        compat=cfg.get("crf_bilateral_compat", 5),
    )

    # Inference
    n_iters = cfg.get("crf_iterations", 5)
    Q = d.inference(n_iters)
    result = np.argmax(Q, axis=0).reshape(H, W).astype(np.int32)
    return result
