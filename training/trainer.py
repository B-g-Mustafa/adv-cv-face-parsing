"""
Training engine: train loop, validation, checkpointing, early stopping, logging.

Supports full resume from checkpoint — restores model, optimizer, scheduler,
scaler, epoch counter, best F1, and early stopping state.
"""
import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from evaluation.metrics import compute_metrics_batch, compute_multiclass_fscore
from utils.helpers import save_checkpoint, load_checkpoint, ensure_dir


class Trainer:
    """
    Full training pipeline with:
    - AdamW + cosine annealing warm restarts
    - Mixed precision (AMP)
    - Gradient clipping
    - Best + periodic checkpointing
    - Early stopping
    - CSV logging
    - Full resume from any checkpoint
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
        cfg: Dict[str, Any],
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.cfg = cfg
        tcfg = cfg.get("training", {})

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tcfg.get("learning_rate", 1e-3),
            weight_decay=tcfg.get("weight_decay", 1e-4),
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=tcfg.get("T_0", 50),
            T_mult=tcfg.get("T_mult", 1),
            eta_min=tcfg.get("eta_min", 1e-6),
        )

        # AMP
        self.use_amp = tcfg.get("use_amp", True) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.grad_clip = tcfg.get("grad_clip_max_norm", 1.0)

        # Training params
        self.epochs = tcfg.get("epochs", 200)
        self.patience = tcfg.get("early_stopping_patience", 30)

        # Tracking
        self.best_f1 = 0.0
        self.epochs_no_improve = 0
        self.start_epoch = 1
        self.history: List[Dict[str, float]] = []

        # Dirs
        self.ckpt_dir = ensure_dir(cfg["data"].get("checkpoint_dir", "outputs/checkpoints"))
        self.log_dir = ensure_dir(cfg["data"].get("log_dir", "outputs/logs"))

        self.num_classes = cfg.get("num_classes", 19)
        self.ckpt_every = tcfg.get("checkpoint_every", 10)

    # ---------------------------------------------------------
    def resume(self, checkpoint_path: str) -> None:
        """
        Resume training from a checkpoint.

        Restores: model weights, optimizer state, scheduler state,
        AMP scaler state, epoch number, best F1, early stopping counter.
        """
        print(f"\n{'='*60}")
        print(f"  Resuming from: {checkpoint_path}")
        print(f"{'='*60}")

        ckpt = load_checkpoint(
            checkpoint_path,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
        )

        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_f1 = ckpt.get("best_f1", ckpt.get("metrics", {}).get("f1", 0.0))
        self.epochs_no_improve = ckpt.get("epochs_no_improve", 0)

        print(f"  Restored epoch:            {ckpt.get('epoch', '?')}")
        print(f"  Resuming from epoch:       {self.start_epoch}")
        print(f"  Best F1 so far:            {self.best_f1:.4f}")
        print(f"  Epochs without improve:    {self.epochs_no_improve}")
        print(f"  Learning rate:             {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}\n")

    # ---------------------------------------------------------
    def _save(self, epoch: int, metrics: Dict[str, float], filename: str) -> None:
        """Save checkpoint with full resume state."""
        save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            metrics,
            str(self.ckpt_dir / filename),
            self.scheduler,
            self.scaler,
            best_f1=self.best_f1,
            epochs_no_improve=self.epochs_no_improve,
        )

    # ---------------------------------------------------------
    def train_one_epoch(
        self, loader: torch.utils.data.DataLoader, epoch: int
    ) -> Dict[str, float]:
        self.model.train()
        running = {"total": 0.0, "ce": 0.0, "dice": 0.0}
        n_batches = 0

        pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                losses = self.loss_fn(logits, masks)

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k in running:
                if k in losses:
                    running[k] += losses[k].item()
            n_batches += 1
            pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # ---------------------------------------------------------
    @torch.no_grad()
    def validate(
        self, loader: torch.utils.data.DataLoader
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        self.model.eval()
        running_loss = {"total": 0.0, "ce": 0.0, "dice": 0.0}
        all_preds = []
        all_targets = []
        n_batches = 0

        for batch in tqdm(loader, desc="Validating", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                losses = self.loss_fn(logits, masks)

            for k in running_loss:
                if k in losses:
                    running_loss[k] += losses[k].item()
            n_batches += 1

            preds = logits.argmax(dim=1)  # (B, H, W)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

        avg_loss = {k: v / max(n_batches, 1) for k, v in running_loss.items()}

        # Compute metrics
        preds_cat = torch.cat(all_preds, dim=0).numpy()
        targets_cat = torch.cat(all_targets, dim=0).numpy()
        metrics = compute_metrics_batch(preds_cat, targets_cat, self.num_classes)

        return avg_loss, metrics

    # ---------------------------------------------------------
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> List[Dict[str, float]]:
        """Run full training loop (supports resume via self.start_epoch)."""
        csv_path = self.log_dir / "training_log.csv"

        # If resuming and CSV exists, append; otherwise write fresh
        csv_mode = "a" if self.start_epoch > 1 and csv_path.exists() else "w"
        csv_file = open(csv_path, csv_mode, newline="")
        csv_writer = None

        print(f"\n{'='*60}")
        print(f"Training epochs {self.start_epoch}–{self.epochs}")
        print(f"Device: {self.device} | AMP: {self.use_amp}")
        if self.start_epoch > 1:
            print(f"Resumed | Best F1: {self.best_f1:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, val_metrics = self.validate(val_loader)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            # Record
            record = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss["total"],
                "val_loss": val_loss["total"],
                "val_submission_f1": val_metrics["submission_f1"],
                "val_f1": val_metrics["mean_f1"],
                "val_iou": val_metrics["mean_iou"],
                "val_dice": val_metrics["mean_dice"],
                "val_pixel_acc": val_metrics["pixel_accuracy"],
                "time_s": elapsed,
            }
            self.history.append(record)

            # CSV
            if csv_writer is None:
                fieldnames = list(record.keys())
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                if csv_mode == "w":
                    csv_writer.writeheader()
            csv_writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in record.items()})
            csv_file.flush()

            # Console log
            print(
                f"Epoch {epoch:>3d}/{self.epochs} | "
                f"lr={lr:.2e} | "
                f"train_loss={train_loss['total']:.4f} | "
                f"val_loss={val_loss['total']:.4f} | "
                f"SubF1={val_metrics['submission_f1']:.4f} | "
                f"F1={val_metrics['mean_f1']:.4f} | "
                f"IoU={val_metrics['mean_iou']:.4f} | "
                f"Dice={val_metrics['mean_dice']:.4f} | "
                f"PixAcc={val_metrics['pixel_accuracy']:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Checkpoint metrics stored in checkpoint file
            ckpt_metrics = {"f1": val_metrics["submission_f1"], "iou": val_metrics["mean_iou"]}

            # Save periodic checkpoint
            if epoch % self.ckpt_every == 0:
                self._save(epoch, ckpt_metrics, f"epoch_{epoch:03d}.pth")

            # Save best (based on submission F1 — the real grading metric)
            if val_metrics["submission_f1"] > self.best_f1:
                self.best_f1 = val_metrics["submission_f1"]
                self.epochs_no_improve = 0
                self._save(epoch, ckpt_metrics, "best.pth")
                print(f"  ↑ New best F1: {self.best_f1:.4f} — saved best.pth")
            else:
                self.epochs_no_improve += 1

            # Save last (always — this is what you resume from)
            self._save(epoch, ckpt_metrics, "last.pth")

            # Early stopping
            if self.epochs_no_improve >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break

        csv_file.close()
        print(f"\nTraining complete. Best F1: {self.best_f1:.4f}")
        print(f"Logs saved to {csv_path}")
        return self.history
