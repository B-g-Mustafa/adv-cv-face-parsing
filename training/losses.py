"""
Loss functions for face semantic segmentation.

Combined loss: weighted CE + Dice + Boundary + Edge auxiliary.
Handles 19-class imbalance via per-class weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Dict, Optional


# =====================================================================
# Edge map generation from ground truth masks
# =====================================================================
def compute_edge_maps(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute binary edge maps from class-index masks.

    A pixel is on an edge if any of its 4-connected neighbors has a
    different class label. This produces 1-pixel-wide boundaries between
    all class regions.

    Args:
        masks: (B, H, W) integer class labels.

    Returns:
        (B, 1, H, W) float32 edge maps (1.0 = edge, 0.0 = non-edge).
    """
    masks_float = masks.float().unsqueeze(1)  # (B, 1, H, W)

    # Shift in 4 directions and compare
    pad = F.pad(masks_float, (1, 1, 1, 1), mode="replicate")
    # up, down, left, right
    diff_u = (masks_float != pad[:, :, :-2, 1:-1]).float()
    diff_d = (masks_float != pad[:, :, 2:, 1:-1]).float()
    diff_l = (masks_float != pad[:, :, 1:-1, :-2]).float()
    diff_r = (masks_float != pad[:, :, 1:-1, 2:]).float()

    edges = (diff_u + diff_d + diff_l + diff_r).clamp(0, 1)
    return edges  # (B, 1, H, W)


# =====================================================================
# Loss components
# =====================================================================
class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, num_classes: int = 19, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw predictions.
            targets: (B, H, W) integer class labels.
        """
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        targets_oh = F.one_hot(targets, self.num_classes)  # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        dims = (0, 2, 3)  # reduce over batch and spatial
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_oh.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss using distance transforms.

    Penalizes predictions far from true boundaries, helping with
    fine structures like earrings and eyebrows.
    """

    def __init__(self, num_classes: int = 19):
        super().__init__()
        self.num_classes = num_classes

    @torch.no_grad()
    def _compute_dist_maps(self, targets: torch.Tensor) -> torch.Tensor:
        """Compute distance transform for each class per batch sample."""
        B, H, W = targets.shape
        dist_maps = torch.zeros(B, self.num_classes, H, W, device=targets.device)

        targets_np = targets.cpu().numpy()
        for b in range(B):
            for c in range(self.num_classes):
                mask = (targets_np[b] == c).astype(np.uint8)
                if mask.sum() == 0:
                    continue
                # Distance transform from boundary
                pos_dist = distance_transform_edt(mask)
                neg_dist = distance_transform_edt(1 - mask)
                # Signed distance: negative inside, positive outside
                dist = neg_dist - pos_dist
                # Normalize to [-1, 1] range
                max_val = max(abs(dist.min()), abs(dist.max()), 1e-8)
                dist = dist / max_val
                dist_maps[b, c] = torch.from_numpy(dist.astype(np.float32))

        return dist_maps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        dist_maps = self._compute_dist_maps(targets)
        return (probs * dist_maps).mean()


class EdgeLoss(nn.Module):
    """
    Binary cross-entropy loss on predicted edge maps vs GT edges.

    Uses class-balanced weighting: edge pixels are rare (~5% of image),
    so they get upweighted to avoid the model predicting all-zeros.
    """

    def __init__(self, pos_weight: float = 5.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(
        self, edge_logits: torch.Tensor, edge_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            edge_logits: (B, 1, H, W) raw edge predictions.
            edge_targets: (B, 1, H, W) binary edge ground truth.
        """
        pw = self.pos_weight.to(edge_logits.device)
        return F.binary_cross_entropy_with_logits(
            edge_logits, edge_targets, pos_weight=pw
        )


# =====================================================================
# Combined loss
# =====================================================================
class CombinedLoss(nn.Module):
    """
    Combined loss: α·CE + β·Dice + γ·Boundary + δ·Edge.

    Args:
        num_classes: Number of segmentation classes.
        class_weights: Optional per-class weights for CE loss.
        ce_weight: Coefficient for cross-entropy term.
        dice_weight: Coefficient for Dice term.
        boundary_weight: Coefficient for boundary term (0 to disable).
        edge_weight: Coefficient for edge auxiliary term (0 to disable).
        label_smoothing: Label smoothing for CE loss.
    """

    def __init__(
        self,
        num_classes: int = 19,
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 0.5,
        edge_weight: float = 1.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.edge_weight = edge_weight

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=label_smoothing
        )
        self.dice_loss = DiceLoss(num_classes=num_classes)
        self.boundary_loss = BoundaryLoss(num_classes=num_classes) if boundary_weight > 0 else None
        self.edge_loss = EdgeLoss(pos_weight=5.0) if edge_weight > 0 else None

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            model_output: dict with 'seg' (B, C, H, W) and optional 'edge' (B, 1, H, W),
                          OR just a tensor (B, C, H, W) during inference.
            targets: (B, H, W) integer class labels.
        """
        # Handle both dict (training) and tensor (inference) outputs
        if isinstance(model_output, dict):
            logits = model_output["seg"]
            edge_logits = model_output.get("edge", None)
        else:
            logits = model_output
            edge_logits = None

        losses = {}
        total = torch.tensor(0.0, device=logits.device)

        ce = self.ce_loss(logits, targets)
        losses["ce"] = ce
        total = total + self.ce_weight * ce

        dice = self.dice_loss(logits, targets)
        losses["dice"] = dice
        total = total + self.dice_weight * dice

        if self.boundary_loss is not None and self.boundary_weight > 0:
            boundary = self.boundary_loss(logits, targets)
            losses["boundary"] = boundary
            total = total + self.boundary_weight * boundary

        # Edge auxiliary loss
        if (
            self.edge_loss is not None
            and self.edge_weight > 0
            and edge_logits is not None
        ):
            edge_targets = compute_edge_maps(targets)
            edge = self.edge_loss(edge_logits, edge_targets)
            losses["edge"] = edge
            total = total + self.edge_weight * edge

        losses["total"] = total
        return losses
