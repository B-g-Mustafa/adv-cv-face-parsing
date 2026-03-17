"""
verify_dataloader.py — Verify the dataset loader with a real image.

Checks that:
1. RGB mask → class indices conversion works correctly
2. Preprocessing produces correct shape/dtype
3. Class weights are computed
4. Saves visual overlay of mask on image for inspection

Usage:
    python verify_dataloader.py
    python verify_dataloader.py --config configs/default.yaml --num-samples 3
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

from utils.helpers import load_config
from preprocessing.transforms import FacePreprocessor
from preprocessing.augmentation import get_val_augmentation
from datasets.face_dataset import (
    FaceParsingDataset,
    CELEBAMASK_COLOR_MAP,
    rgb_mask_to_class_ids,
    class_ids_to_rgb_mask,
)


# Class names for display
CLASS_NAMES = [
    "background", "skin", "l_brow", "r_brow", "l_eye", "r_eye",
    "eye_g", "l_ear", "r_ear", "ear_r", "nose", "mouth",
    "u_lip", "l_lip", "neck", "neck_l", "cloth", "hair", "hat",
]


def verify(cfg: dict, num_samples: int = 3, output_dir: str = "outputs/dataloader_verify"):
    os.makedirs(output_dir, exist_ok=True)

    preproc_cfg = cfg.get("preprocessing", {})
    preprocessor = FacePreprocessor(preproc_cfg)
    val_aug = get_val_augmentation()

    # Build dataset (no augmentation for clean verification)
    ds = FaceParsingDataset(
        images_dir=cfg["data"]["val_images"],
        masks_dir=cfg["data"]["val_masks"],
        preprocessor=preprocessor,
        augmentation=val_aug,
        image_size=cfg.get("image_size", 512),
        num_classes=cfg.get("num_classes", 19),
    )

    print(f"\n{'='*60}")
    print(f"  Dataset Loader Verification")
    print(f"{'='*60}")
    print(f"  Samples found: {len(ds)}")
    print(f"  Mask format:   {'RGB-colored → class indices' if ds._rgb_masks else 'grayscale (class indices)'}")

    for i in range(min(num_samples, len(ds))):
        sample = ds[i]
        img_tensor = sample["image"]
        mask_tensor = sample["mask"]

        img_path = ds.samples[i][0]
        mask_path = ds.samples[i][1]
        stem = img_path.stem

        print(f"\n  --- Sample {i+1}: {stem} ---")
        print(f"  Image tensor shape: {img_tensor.shape}  dtype: {img_tensor.dtype}")
        print(f"  Mask tensor shape:  {mask_tensor.shape}  dtype: {mask_tensor.dtype}")
        print(f"  Mask unique values: {torch.unique(mask_tensor).tolist()}")
        print(f"  Mask value range:   [{mask_tensor.min().item()}, {mask_tensor.max().item()}]")

        # Verify all mask values are valid class IDs
        assert mask_tensor.min() >= 0, f"Negative class ID found: {mask_tensor.min()}"
        assert mask_tensor.max() < cfg.get("num_classes", 19), (
            f"Class ID {mask_tensor.max()} exceeds num_classes={cfg.get('num_classes', 19)}"
        )
        print(f"  ✅ All class IDs valid (0-{cfg.get('num_classes', 19)-1})")

        # Per-class pixel counts
        print(f"\n  {'Class':<15} {'ID':>3} {'Pixels':>10} {'%':>8}")
        print(f"  {'-'*40}")
        total_px = mask_tensor.numel()
        for c in range(cfg.get("num_classes", 19)):
            count = (mask_tensor == c).sum().item()
            if count > 0:
                pct = 100 * count / total_px
                print(f"  {CLASS_NAMES[c]:<15} {c:>3} {count:>10,} {pct:>7.2f}%")

        # Save visual: original image + class mask overlay + reconstructed RGB mask
        # Load original image for display
        orig = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, (cfg.get("image_size", 512), cfg.get("image_size", 512)))

        # Reconstruct RGB mask from class indices
        mask_np = mask_tensor.numpy()
        rgb_mask = class_ids_to_rgb_mask(mask_np)

        # Overlay: blend image + mask
        overlay = cv2.addWeighted(orig, 0.5, rgb_mask, 0.5, 0)

        # Save side-by-side: original | RGB mask | overlay
        combined = np.concatenate([orig, rgb_mask, overlay], axis=1)
        save_path = os.path.join(output_dir, f"{stem}_dataloader_verify.png")
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"\n  Saved: {save_path}")
        print(f"         (left=original | center=class mask | right=overlay)")

    # Compute class weights
    print(f"\n  Computing class weights across dataset...")
    weights = ds.compute_class_weights()
    print(f"\n  {'Class':<15} {'Weight':>10}")
    print(f"  {'-'*28}")
    for c in range(cfg.get("num_classes", 19)):
        print(f"  {CLASS_NAMES[c]:<15} {weights[c].item():>10.4f}")

    print(f"\n{'='*60}")
    print(f"  ✅ ALL CHECKS PASSED")
    print(f"  Outputs saved to: {output_dir}/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Verify dataset loader")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output", type=str, default="outputs/dataloader_verify")
    args = parser.parse_args()

    cfg = load_config(args.config)
    verify(cfg, args.num_samples, args.output)


if __name__ == "__main__":
    main()
