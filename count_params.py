"""
count_params.py — Model parameter count verification.

Prints a layer-by-layer breakdown and verifies the total
stays under the 1.7M parameter budget.

Usage:
    python count_params.py
    python count_params.py --config configs/default.yaml
"""
import argparse
import torch
from utils.helpers import load_config
from models.attention_lite_unet import build_model


def count_params(model: torch.nn.Module, budget: int = 1_700_000) -> None:
    total = 0
    trainable = 0

    print(f"\n{'Module':<55} {'Shape':<25} {'Params':>12}")
    print("─" * 95)

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        print(f"  {name:<53} {str(list(param.shape)):<25} {n:>12,}")

    print("─" * 95)
    print(f"  {'Total parameters':<53} {'':<25} {total:>12,}")
    print(f"  {'Trainable parameters':<53} {'':<25} {trainable:>12,}")
    print(f"  {'Non-trainable':<53} {'':<25} {total - trainable:>12,}")
    print(f"  {'Budget':<53} {'':<25} {budget:>12,}")
    print(f"  {'Remaining headroom':<53} {'':<25} {budget - total:>12,}")
    print("─" * 95)

    if total < budget:
        print(f"\n  ✅ PASS — {total:,} params < {budget:,} budget\n")
    else:
        print(f"\n  ❌ FAIL — {total:,} params EXCEEDS {budget:,} budget by {total - budget:,}\n")

    # Breakdown by top-level module
    print(f"\n{'Component':<35} {'Params':>12} {'%':>8}")
    print("─" * 58)
    component_params = {}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        component_params[top] = component_params.get(top, 0) + param.numel()
    for comp, n in sorted(component_params.items(), key=lambda x: -x[1]):
        pct = 100 * n / total
        bar = "█" * int(pct / 2)
        print(f"  {comp:<33} {n:>12,} {pct:>7.1f}%  {bar}")
    print("─" * 58)
    print()


def main():
    parser = argparse.ArgumentParser(description="Count model parameters")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    count_params(model)


if __name__ == "__main__":
    main()
