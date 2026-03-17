"""
train.py — Main training entry point.

Usage:
    python train.py                                    # train from scratch
    python train.py --config configs/custom.yaml       # custom config
    python train.py --resume                           # resume from last.pth
    python train.py --resume outputs/checkpoints/epoch_050.pth  # resume from specific checkpoint
"""
import argparse
import sys
from pathlib import Path

import torch

from utils.helpers import load_config, set_seed, get_device, model_summary
from datasets.face_dataset import build_dataloaders
from models.attention_lite_unet import build_model
from training.losses import CombinedLoss
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Face Parsing Model")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume", type=str, nargs="?", const="auto", default=None,
        help="Resume training. Pass a checkpoint path, or just --resume to auto-load last.pth.",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device()
    print(f"Device: {device}")

    # Build data loaders
    print("Building datasets...")
    train_loader, val_loader, class_weights = build_dataloaders(cfg)
    print(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")

    # Build model
    print("\nBuilding model...")
    model = build_model(cfg)
    print(model_summary(model))

    # Verify parameter budget
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 1_700_000, (
        f"Model exceeds 1.7M parameter budget: {total_params:,}"
    )
    print(f"✓ Parameter budget OK: {total_params:,} < 1,700,000\n")

    # Build loss
    loss_cfg = cfg.get("loss", {})
    use_cw = loss_cfg.get("use_class_weights", True)
    loss_fn = CombinedLoss(
        num_classes=cfg.get("num_classes", 19),
        class_weights=class_weights.to(device) if use_cw else None,
        ce_weight=loss_cfg.get("ce_weight", 1.0),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        boundary_weight=loss_cfg.get("boundary_weight", 0.5),
        label_smoothing=loss_cfg.get("label_smoothing", 0.05),
    )

    # Build trainer
    trainer = Trainer(model, loss_fn, device, cfg)

    # Resume from checkpoint if requested
    if args.resume is not None:
        if args.resume == "auto":
            # Auto-detect: try last.pth
            ckpt_dir = Path(cfg["data"].get("checkpoint_dir", "outputs/checkpoints"))
            last_ckpt = ckpt_dir / "last.pth"
            if last_ckpt.exists():
                trainer.resume(str(last_ckpt))
            else:
                print(f"No last.pth found in {ckpt_dir}. Starting from scratch.")
        else:
            # Explicit path
            if not Path(args.resume).exists():
                print(f"ERROR: Checkpoint not found: {args.resume}")
                sys.exit(1)
            trainer.resume(args.resume)

    # Train
    trainer.fit(train_loader, val_loader)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
