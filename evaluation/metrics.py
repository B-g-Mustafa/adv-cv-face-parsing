"""
Segmentation evaluation metrics.

Includes:
- compute_multiclass_fscore: EXACT match to the CodaBench submission metric
- compute_metrics_batch: Full metrics suite (Dice, IoU, F1, Precision, Recall)
"""
import numpy as np
from typing import Dict


# =====================================================================
# OFFICIAL SUBMISSION METRIC — DO NOT MODIFY
# =====================================================================
def compute_multiclass_fscore(
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    beta: float = 1.0,
) -> float:
    """
    Exact replica of the CodaBench submission evaluation function.

    Computes per-class F-score ONLY for classes present in the ground truth,
    then returns the mean. This is the metric the model is ranked on.

    Args:
        mask_gt:   (H, W) or (N, H, W) integer ground truth labels.
        mask_pred: (H, W) or (N, H, W) integer predicted labels.
        beta: F-beta parameter (1.0 = standard F1).

    Returns:
        Mean F-score across classes present in ground truth.
    """
    f_scores = []

    for class_id in np.unique(mask_gt):
        tp = np.sum((mask_gt == class_id) & (mask_pred == class_id))
        fp = np.sum((mask_gt != class_id) & (mask_pred == class_id))
        fn = np.sum((mask_gt == class_id) & (mask_pred != class_id))

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f_score = (
            (1 + beta**2)
            * (precision * recall)
            / ((beta**2 * precision) + recall + 1e-7)
        )

        f_scores.append(f_score)

    return float(np.mean(f_scores))


# =====================================================================
# FULL METRICS SUITE (for detailed per-class analysis)
# =====================================================================
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
        Dictionary with per-class and mean metrics,
        including 'submission_f1' matching the exact CodaBench metric.
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

    # Submission F1 (EXACT match to CodaBench grading)
    submission_f1 = compute_multiclass_fscore(targets, preds)

    # Mean metrics (across all classes)
    result = {
        "submission_f1": submission_f1,
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


def print_metrics(metrics: Dict[str, float], num_classes: int = 19) -> None:
    """Pretty-print metrics to console."""
    print(f"\n{'='*55}")
    print(f"  SUBMISSION F1 (CodaBench): {metrics['submission_f1']:.4f}")
    print(f"{'='*55}")
    print(f"  Pixel Accuracy:   {metrics['pixel_accuracy']:.4f}")
    print(f"  Mean IoU:         {metrics['mean_iou']:.4f}")
    print(f"  Mean Dice:        {metrics['mean_dice']:.4f}")
    print(f"  Mean F1 (all 19): {metrics['mean_f1']:.4f}")
    print(f"  Mean Precision:   {metrics['mean_precision']:.4f}")
    print(f"  Mean Recall:      {metrics['mean_recall']:.4f}")

    CLASS_NAMES = [
        "background", "skin", "l_brow", "r_brow", "l_eye", "r_eye",
        "eye_g", "l_ear", "r_ear", "ear_r", "nose", "mouth",
        "u_lip", "l_lip", "neck", "neck_l", "cloth", "hair", "hat",
    ]

    print(f"\n  {'Class':<15} {'F1':>8} {'IoU':>8} {'Dice':>8}")
    print(f"  {'-'*43}")
    for c in range(num_classes):
        f1 = metrics.get(f"class_{c}_f1", 0)
        iou = metrics.get(f"class_{c}_iou", 0)
        dice = metrics.get(f"class_{c}_dice", 0)
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        print(f"  {name:<15} {f1:>8.4f} {iou:>8.4f} {dice:>8.4f}")
    print(f"{'='*55}\n")
