"""
train.py — Main training entry point.

Usage:
    python train.py                                    # train from scratch
    python train.py --config configs/custom.yaml       # custom config
    python train.py --resume                           # resume from last.pth
    python train.py --resume outputs/checkpoints/epoch_050.pth  # resume from specific checkpoint
    python train.py --wandb                            # force-enable W&B logging
    python train.py --no-wandb                         # force-disable W&B logging
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Any

import torch

from utils.helpers import load_config, set_seed, get_device, model_summary
from datasets.face_dataset import build_dataloaders
from models.attention_lite_unet import build_model
from training.losses import CombinedLoss
from training.trainer import Trainer


def _init_wandb(cfg: dict) -> Optional[Any]:
    """Initialize W&B run from config if enabled."""
    wb_cfg = cfg.get("wandb", {})
    if not wb_cfg.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb is enabled in config but package is not installed.")
        print("Install it with: pip install wandb")
        sys.exit(1)

    run = wandb.init(
        project=wb_cfg.get("project", "face-semantic-parsing"),
        entity=wb_cfg.get("entity", None),
        name=wb_cfg.get("run_name", None),
        tags=wb_cfg.get("tags", []),
        notes=wb_cfg.get("notes", None),
        config=cfg,
    )
    print(f"W&B enabled: run={run.name}")
    return run


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
    parser.add_argument(
        "--wandb", dest="wandb", action="store_true",
        help="Enable Weights & Biases logging for this run.",
    )
    parser.add_argument(
        "--no-wandb", dest="wandb", action="store_false",
        help="Disable Weights & Biases logging for this run.",
    )
    parser.set_defaults(wandb=None)
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.wandb is not None:
        cfg.setdefault("wandb", {})["enabled"] = args.wandb
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
    wandb_run = _init_wandb(cfg)
    trainer = Trainer(model, loss_fn, device, cfg, wandb_run=wandb_run)

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
    try:
        trainer.fit(train_loader, val_loader)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
