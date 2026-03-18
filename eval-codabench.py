"""
Evaluation script — computes mean F1 between predicted and ground truth masks.

Usage:
    python evaluate.py --pred-dir /path/to/predicted/masks \
                       --gt-dir   /path/to/ground/truth/masks
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def compute_multiclass_fscore(mask_gt: np.ndarray, mask_pred: np.ndarray, beta: int = 1) -> float:
    f_scores = []
    for class_id in np.unique(mask_gt):
        tp = np.sum((mask_gt == class_id) & (mask_pred == class_id))
        fp = np.sum((mask_gt != class_id) & (mask_pred == class_id))
        fn = np.sum((mask_gt == class_id) & (mask_pred != class_id))

        precision = tp / (tp + fp + 1e-7)
        recall    = tp / (tp + fn + 1e-7)
        f_score   = (
            (1 + beta ** 2)
            * (precision * recall)
            / ((beta ** 2 * precision) + recall + 1e-7)
        )
        f_scores.append(f_score)

    return float(np.mean(f_scores))


def main():
    parser = argparse.ArgumentParser(description="F1 evaluation: predicted vs ground truth masks")
    parser.add_argument("--pred-dir", required=True, help="Directory of predicted masks (.png)")
    parser.add_argument("--gt-dir",   required=True, help="Directory of ground truth masks (.png)")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir   = Path(args.gt_dir)

    # Match predicted masks to ground truth by filename stem
    pred_files = {p.stem: p for p in pred_dir.glob("*.png")}
    gt_files   = {p.stem: p for p in gt_dir.glob("*.png")}

    common = sorted(set(pred_files.keys()) & set(gt_files.keys()))

    if not common:
        print("[Error] No matching filenames found between the two directories.")
        print(f"        Pred dir has {len(pred_files)} files, GT dir has {len(gt_files)} files.")
        return

    missing_pred = set(gt_files.keys()) - set(pred_files.keys())
    if missing_pred:
        print(f"[Warning] {len(missing_pred)} GT files have no matching prediction — skipped.")

    per_image_f1s = []
    for stem in common:
        mask_pred = np.array(Image.open(pred_files[stem]).convert("L"), dtype=np.uint8)
        mask_gt   = np.array(Image.open(gt_files[stem]).convert("L"),   dtype=np.uint8)

        if mask_pred.shape != mask_gt.shape:
            # Resize prediction to GT size if they differ
            pred_img  = Image.open(pred_files[stem]).convert("L").resize(
                (mask_gt.shape[1], mask_gt.shape[0]), Image.NEAREST
            )
            mask_pred = np.array(pred_img, dtype=np.uint8)

        f1 = compute_multiclass_fscore(mask_gt, mask_pred, beta=1)
        per_image_f1s.append(f1)

    mean_f1 = float(np.mean(per_image_f1s))
    print(f"{mean_f1:.4f}")


if __name__ == "__main__":
    main()