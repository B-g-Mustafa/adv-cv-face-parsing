"""
inference.py — Main inference entry point.

Usage:
    python inference.py --input data/test/images --output outputs/predictions
    python inference.py --input path/to/single_image.png --output outputs/predictions
    python inference.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pth
"""
import argparse
from pathlib import Path

from utils.helpers import load_config, set_seed, get_device, colorize_mask
from models.attention_lite_unet import build_model
from inference.predict import InferencePipeline


def main():
    parser = argparse.ArgumentParser(description="Run Face Parsing Inference")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint. Defaults to outputs/checkpoints/best.pth.",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input image or directory of images.",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/predictions",
        help="Directory to save output masks.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = str(
            Path(cfg["data"].get("checkpoint_dir", "outputs/checkpoints")) / "best.pth"
        )

    # Model factory (for ensemble, each call creates a fresh model)
    model_fn = lambda: build_model(cfg)

    pipeline = InferencePipeline(
        model_fn=model_fn,
        cfg=cfg,
        checkpoint_path=checkpoint,
        device=get_device(),
    )

    input_path = Path(args.input)
    if input_path.is_dir():
        pipeline.predict_batch(str(input_path), args.output)
    elif input_path.is_file():
        mask = pipeline.predict_single(str(input_path))
        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        save_file = out_path / f"{input_path.stem}.png"
        colorized = colorize_mask(mask)
        colorized.save(str(save_file))
        print(f"Saved mask to {save_file}")
    else:
        print(f"Error: {args.input} not found.")


if __name__ == "__main__":
    main()
