"""
evaluate.py — Standalone evaluation script.

Loads predicted masks and ground truth masks from directories,
computes all segmentation metrics, and prints a detailed report.

Usage:
    python evaluate.py --preds outputs/predictions --gt data/val/masks
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from evaluation.metrics import compute_metrics_batch, print_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Face Parsing Results")
    parser.add_argument(
        "--preds", type=str, required=True, help="Directory of predicted masks (PNG)."
    )
    parser.add_argument(
        "--gt", type=str, required=True, help="Directory of ground truth masks (PNG)."
    )
    parser.add_argument(
        "--num-classes", type=int, default=19, help="Number of classes."
    )
    parser.add_argument(
        "--image-size", type=int, default=512,
        help="Resize masks to this size before evaluation (0=no resize).",
    )
    args = parser.parse_args()

    pred_dir = Path(args.preds)
    gt_dir = Path(args.gt)

    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    pred_files = {f.stem: f for f in pred_dir.iterdir() if f.suffix.lower() in extensions}
    gt_files = {f.stem: f for f in gt_dir.iterdir() if f.suffix.lower() in extensions}

    common = sorted(set(pred_files.keys()) & set(gt_files.keys()))
    if not common:
        print("No matching prediction-ground truth pairs found.")
        return

    print(f"Evaluating {len(common)} samples...")

    all_preds = []
    all_gts = []
    for stem in common:
        pred = cv2.imread(str(pred_files[stem]), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gt_files[stem]), cv2.IMREAD_GRAYSCALE)

        if args.image_size > 0:
            pred = cv2.resize(pred, (args.image_size, args.image_size),
                              interpolation=cv2.INTER_NEAREST)
            gt = cv2.resize(gt, (args.image_size, args.image_size),
                            interpolation=cv2.INTER_NEAREST)

        pred = np.clip(pred, 0, args.num_classes - 1)
        gt = np.clip(gt, 0, args.num_classes - 1)

        all_preds.append(pred)
        all_gts.append(gt)

    preds_arr = np.stack(all_preds, axis=0)
    gts_arr = np.stack(all_gts, axis=0)

    metrics = compute_metrics_batch(preds_arr, gts_arr, args.num_classes)
    print_metrics(metrics, args.num_classes)


if __name__ == "__main__":
    main()
