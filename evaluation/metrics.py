"""
Segmentation evaluation metrics.

Per-class and mean: Dice, IoU, F1, Precision, Recall, Pixel Accuracy.
"""
import numpy as np
from typing import Dict


def compute_metrics_batch(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 19,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute segmentation metrics over a batch of predictions.

    Args:
        preds:   (N, H, W) integer class predictions.
        targets: (N, H, W) integer ground truth labels.
        num_classes: Number of classes.

    Returns:
        Dictionary with per-class and mean metrics.
    """
    # Flatten
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for c_true in range(num_classes):
        t_mask = targets_flat == c_true
        for c_pred in range(num_classes):
            cm[c_true, c_pred] = np.sum(preds_flat[t_mask] == c_pred)

    # Per-class metrics
    per_class_iou = np.zeros(num_classes)
    per_class_dice = np.zeros(num_classes)
    per_class_precision = np.zeros(num_classes)
    per_class_recall = np.zeros(num_classes)
    per_class_f1 = np.zeros(num_classes)

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)

        per_class_precision[c] = precision
        per_class_recall[c] = recall
        per_class_f1[c] = f1
        per_class_iou[c] = iou
        per_class_dice[c] = dice

    # Pixel accuracy
    correct = np.diag(cm).sum()
    total = cm.sum()
    pixel_acc = correct / (total + eps)

    # Mean metrics (across all classes, including background)
    result = {
        "pixel_accuracy": float(pixel_acc),
        "mean_iou": float(per_class_iou.mean()),
        "mean_dice": float(per_class_dice.mean()),
        "mean_f1": float(per_class_f1.mean()),
        "mean_precision": float(per_class_precision.mean()),
        "mean_recall": float(per_class_recall.mean()),
    }

    # Per-class detail
    for c in range(num_classes):
        result[f"class_{c}_iou"] = float(per_class_iou[c])
        result[f"class_{c}_f1"] = float(per_class_f1[c])
        result[f"class_{c}_dice"] = float(per_class_dice[c])

    return result


def compute_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 19,
) -> np.ndarray:
    """Return (num_classes, num_classes) confusion matrix."""
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for c_true in range(num_classes):
        t_mask = targets_flat == c_true
        for c_pred in range(num_classes):
            cm[c_true, c_pred] = np.sum(preds_flat[t_mask] == c_pred)
    return cm


def print_metrics(metrics: Dict[str, float], num_classes: int = 19) -> None:
    """Pretty-print metrics to console."""
    print(f"\n{'='*50}")
    print(f"Pixel Accuracy:  {metrics['pixel_accuracy']:.4f}")
    print(f"Mean IoU:        {metrics['mean_iou']:.4f}")
    print(f"Mean Dice:       {metrics['mean_dice']:.4f}")
    print(f"Mean F1:         {metrics['mean_f1']:.4f}")
    print(f"Mean Precision:  {metrics['mean_precision']:.4f}")
    print(f"Mean Recall:     {metrics['mean_recall']:.4f}")
    print(f"\n{'Per-Class F1:':<20}")
    for c in range(num_classes):
        key = f"class_{c}_f1"
        if key in metrics:
            print(f"  Class {c:>2d}: {metrics[key]:.4f}")
    print(f"{'='*50}\n")
