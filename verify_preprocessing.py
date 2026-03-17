"""
verify_preprocessing.py — Visual preprocessing verification.

Generates a side-by-side comparison showing each preprocessing stage
applied to your image. Run this BEFORE training to confirm the pipeline
is working correctly.

Usage:
    python verify_preprocessing.py --image path/to/face.jpg
    python verify_preprocessing.py --image path/to/face.jpg --config configs/default.yaml
    python verify_preprocessing.py --image-dir data/train/images --num-samples 5
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def verify_single_image(image_path: str, cfg: dict, output_dir: str) -> None:
    """Run each preprocessing step and save visual comparison."""
    from preprocessing.transforms import FacePreprocessor

    # Load image
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"ERROR: Could not load image: {image_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image_size = cfg.get("image_size", 512)
    img_rgb = cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    stem = Path(image_path).stem
    pcfg = cfg.get("preprocessing", {})

    stages = {}
    stages["01_original"] = img_rgb.copy()

    # --- Stage 1: CLAHE ---
    proc = FacePreprocessor({"use_clahe": True, "use_bilateral": False, "use_sobel_edge": False,
                             "clahe_clip_limit": pcfg.get("clahe_clip_limit", 2.0),
                             "clahe_grid_size": pcfg.get("clahe_grid_size", 8),
                             "normalize_mean": [0, 0, 0], "normalize_std": [1, 1, 1]})
    clahe_out = proc.apply_clahe(img_rgb.copy())
    stages["02_clahe"] = clahe_out

    # --- Stage 2: Bilateral ---
    proc2 = FacePreprocessor({"use_clahe": False, "use_bilateral": True, "use_sobel_edge": False,
                              "bilateral_d": pcfg.get("bilateral_d", 5),
                              "bilateral_sigma_color": pcfg.get("bilateral_sigma_color", 50),
                              "bilateral_sigma_space": pcfg.get("bilateral_sigma_space", 50),
                              "normalize_mean": [0, 0, 0], "normalize_std": [1, 1, 1]})
    bilateral_out = proc2.apply_bilateral(clahe_out.copy())
    stages["03_bilateral"] = bilateral_out

    # --- Stage 3: Sobel Edge Map ---
    proc3 = FacePreprocessor({"use_clahe": False, "use_bilateral": False, "use_sobel_edge": True,
                              "normalize_mean": [0, 0, 0], "normalize_std": [1, 1, 1]})
    edge = proc3.compute_sobel_edge(bilateral_out.copy())
    # Convert to viewable image (0-255 grayscale → colormap)
    edge_vis = (edge * 255).astype(np.uint8)
    edge_colored = cv2.applyColorMap(edge_vis, cv2.COLORMAP_INFERNO)
    edge_colored = cv2.cvtColor(edge_colored, cv2.COLOR_BGR2RGB)
    stages["04_sobel_edge"] = edge_colored

    # --- Stage 4: Full pipeline output (normalized, with edge channel) ---
    full_proc = FacePreprocessor(pcfg)
    full_out = full_proc(img_rgb.copy())  # (H, W, C), float32 normalized
    # Undo normalization for visualization of RGB channels
    mean = np.array(pcfg.get("normalize_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(pcfg.get("normalize_std", [0.229, 0.224, 0.225]), dtype=np.float32)
    rgb_unnorm = full_out[:, :, :3] * std + mean
    rgb_unnorm = np.clip(rgb_unnorm * 255, 0, 255).astype(np.uint8)
    stages["05_final_rgb_channels"] = rgb_unnorm

    if full_out.shape[2] == 4:
        edge_ch = full_out[:, :, 3]
        edge_ch_vis = (np.clip(edge_ch, 0, 1) * 255).astype(np.uint8)
        edge_ch_colored = cv2.applyColorMap(edge_ch_vis, cv2.COLORMAP_INFERNO)
        edge_ch_colored = cv2.cvtColor(edge_ch_colored, cv2.COLOR_BGR2RGB)
        stages["06_final_edge_channel"] = edge_ch_colored

    # --- Save individual stages ---
    os.makedirs(output_dir, exist_ok=True)
    for name, img in stages.items():
        save_path = os.path.join(output_dir, f"{stem}_{name}.png")
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # --- Create comparison grid ---
    imgs = list(stages.values())
    labels = list(stages.keys())

    # Arrange in 2 rows × 3 cols
    h, w = image_size, image_size
    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    grid = np.ones((rows * (h + 40) + 10, cols * (w + 10) + 10, 3), dtype=np.uint8) * 255

    for i, (img, label) in enumerate(zip(imgs, labels)):
        r, c = divmod(i, cols)
        y0 = r * (h + 40) + 40
        x0 = c * (w + 10) + 5
        # Resize if needed
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        grid[y0:y0 + h, x0:x0 + w] = img
        # Label
        label_clean = label[3:].replace("_", " ").title()
        cv2.putText(
            grid, label_clean,
            (x0 + 5, y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
        )

    grid_path = os.path.join(output_dir, f"{stem}_comparison_grid.png")
    cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"  Preprocessing Verification — {stem}")
    print(f"{'='*60}")
    print(f"  Input shape:          {img_rgb.shape}")
    print(f"  Output shape:         {full_out.shape}")
    print(f"  Output channels:      {full_out.shape[2]} ({'RGB + Sobel' if full_out.shape[2] == 4 else 'RGB'})")
    print(f"  Output dtype:         {full_out.dtype}")
    print(f"  Output range (RGB):   [{full_out[:,:,:3].min():.3f}, {full_out[:,:,:3].max():.3f}]")
    if full_out.shape[2] == 4:
        print(f"  Output range (Edge):  [{full_out[:,:,3].min():.3f}, {full_out[:,:,3].max():.3f}]")
    print(f"\n  Steps applied:")
    print(f"    CLAHE:      {'✓ ON' if pcfg.get('use_clahe', True) else '✗ OFF'}")
    print(f"    Bilateral:  {'✓ ON' if pcfg.get('use_bilateral', True) else '✗ OFF'}")
    print(f"    Sobel Edge: {'✓ ON' if pcfg.get('use_sobel_edge', True) else '✗ OFF'}")
    print(f"\n  Saved {len(stages)} stage images + comparison grid to:")
    print(f"    {output_dir}/")
    for name in stages:
        print(f"      {stem}_{name}.png")
    print(f"      {stem}_comparison_grid.png  ← open this!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verify preprocessing pipeline visually."
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single face image to preprocess.",
    )
    parser.add_argument(
        "--image-dir", type=str, default=None,
        help="Directory of face images. Processes --num-samples random images.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=3,
        help="Number of random samples to process from --image-dir.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/preprocessing_verify",
        help="Directory to save verification images.",
    )
    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        print("ERROR: Provide either --image or --image-dir")
        parser.print_help()
        sys.exit(1)

    # Load config
    from utils.helpers import load_config
    cfg = load_config(args.config)

    if args.image:
        verify_single_image(args.image, cfg, args.output)
    else:
        # Process N random samples from directory
        img_dir = Path(args.image_dir)
        extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        all_imgs = sorted(
            f for f in img_dir.iterdir() if f.suffix.lower() in extensions
        )
        if not all_imgs:
            print(f"No images found in {args.image_dir}")
            sys.exit(1)

        import random
        random.seed(42)
        samples = random.sample(all_imgs, min(args.num_samples, len(all_imgs)))
        print(f"Processing {len(samples)} sample images from {args.image_dir}...\n")

        for img_path in samples:
            verify_single_image(str(img_path), cfg, args.output)

    print("Verification complete. Open the comparison grid images to inspect.")


if __name__ == "__main__":
    main()
