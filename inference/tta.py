"""
Test-Time Augmentation (TTA) for segmentation.

Averages softmax probabilities across augmented views
(horizontal flip, small rotations, scale variations).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional


class TTAPredictor:
    """
    Test-Time Augmentation.

    Creates multiple augmented views of each input, runs forward passes,
    and averages softmax probabilities across all views.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        if cfg is None:
            cfg = {}
        self.enabled = cfg.get("enabled", True)
        self.hflip = cfg.get("horizontal_flip", True)
        self.rotations = cfg.get("rotations", [-5, 0, 5])
        self.scales = cfg.get("scales", [1.0])

    @torch.no_grad()
    def predict(
        self, model: nn.Module, image: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Run TTA prediction on a single image.

        Args:
            model: Segmentation model in eval mode.
            image: (1, C, H, W) preprocessed input tensor.
            device: Compute device.

        Returns:
            Averaged softmax probabilities (1, num_classes, H, W).
        """
        model.eval()
        image = image.to(device)
        _, _, H, W = image.shape

        all_probs = []

        for scale in self.scales:
            for angle in self.rotations:
                for flip in ([False, True] if self.hflip else [False]):
                    augmented = image.clone()

                    # Scale
                    if scale != 1.0:
                        sH, sW = int(H * scale), int(W * scale)
                        augmented = F.interpolate(
                            augmented, size=(sH, sW),
                            mode="bilinear", align_corners=False
                        )

                    # Flip
                    if flip:
                        augmented = torch.flip(augmented, dims=[3])

                    # Rotation
                    if angle != 0:
                        theta = torch.tensor(
                            [[np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                             [np.sin(np.radians(angle)),  np.cos(np.radians(angle)), 0]],
                            dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        grid = F.affine_grid(theta, augmented.shape, align_corners=False)
                        augmented = F.grid_sample(
                            augmented, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False
                        )

                    # Forward pass
                    logits = model(augmented)

                    # Undo scale
                    if scale != 1.0:
                        logits = F.interpolate(
                            logits, size=(H, W),
                            mode="bilinear", align_corners=False
                        )

                    # Undo rotation
                    if angle != 0:
                        theta_inv = torch.tensor(
                            [[np.cos(np.radians(-angle)), -np.sin(np.radians(-angle)), 0],
                             [np.sin(np.radians(-angle)),  np.cos(np.radians(-angle)), 0]],
                            dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        grid_inv = F.affine_grid(theta_inv, logits.shape, align_corners=False)
                        logits = F.grid_sample(
                            logits, grid_inv, mode="bilinear",
                            padding_mode="zeros", align_corners=False
                        )

                    # Undo flip
                    if flip:
                        logits = torch.flip(logits, dims=[3])

                    probs = F.softmax(logits, dim=1)
                    all_probs.append(probs)

        # Average all views
        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        return avg_probs
