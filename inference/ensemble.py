"""
Checkpoint Ensembling.

Loads top-K checkpoints (by validation F1) and averages their
softmax predictions for improved robust outputs.
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.helpers import load_checkpoint, get_device


class CheckpointEnsemble:
    """
    Ensemble predictions from multiple checkpoint snapshots.

    Loads top-K best checkpoints, runs each model, and averages
    the softmax probability maps.
    """

    def __init__(
        self,
        model_fn,
        cfg: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model_fn: Callable that returns a fresh model instance.
            cfg: Full config dict.
            device: Compute device.
        """
        self.model_fn = model_fn
        self.cfg = cfg
        self.device = device or get_device()
        ens_cfg = cfg.get("ensemble", {})
        self.top_k = ens_cfg.get("top_k", 3)
        self.ckpt_dir = Path(cfg["data"].get("checkpoint_dir", "outputs/checkpoints"))

    def _find_top_k_checkpoints(self) -> List[str]:
        """Find top-K checkpoints sorted by saved F1 metric."""
        ckpt_files = sorted(self.ckpt_dir.glob("*.pth"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found in {self.ckpt_dir}")

        # Load each and extract F1
        scored = []
        for cp in ckpt_files:
            try:
                state = torch.load(str(cp), map_location="cpu", weights_only=False)
                f1 = state.get("metrics", {}).get("f1", 0.0)
                scored.append((f1, str(cp)))
            except Exception:
                continue

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [path for _, path in scored[: self.top_k]]
        return top

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run ensemble prediction.

        Args:
            image: (1, C, H, W) preprocessed input tensor.

        Returns:
            Averaged softmax probs (1, num_classes, H, W).
        """
        ckpt_paths = self._find_top_k_checkpoints()
        all_probs = []

        for path in ckpt_paths:
            model = self.model_fn()
            load_checkpoint(path, model, device=self.device)
            model = model.to(self.device).eval()

            logits = model(image.to(self.device))
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)

        return torch.stack(all_probs, dim=0).mean(dim=0)
