"""
Utility helpers: seeding, device selection, param counting, checkpoint I/O.
"""
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------
def load_config(path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------
def get_device() -> torch.device:
    """Select best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------
# Parameter counting
# ------------------------------------------------------------------
def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Return total parameter count."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_summary(model: torch.nn.Module) -> str:
    """Return a compact summary string."""
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    return (
        f"Total params:     {total:>12,}\n"
        f"Trainable params: {trainable:>12,}\n"
        f"Non-trainable:    {total - trainable:>12,}"
    )


# ------------------------------------------------------------------
# Checkpoint I/O
# ------------------------------------------------------------------
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    best_f1: float = 0.0,
    epochs_no_improve: int = 0,
) -> None:
    """Save a training checkpoint with full resume state."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "best_f1": best_f1,
        "epochs_no_improve": epochs_no_improve,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint."""
    map_location = device or get_device()
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


# ------------------------------------------------------------------
# Timing
# ------------------------------------------------------------------
class Timer:
    """Simple context-manager timer."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


# ------------------------------------------------------------------
# Misc
# ------------------------------------------------------------------
def ensure_dir(path: str) -> Path:
    """Create directory if it does not exist; return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
