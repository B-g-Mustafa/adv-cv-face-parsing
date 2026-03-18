"""
Full inference pipeline.

preprocess → model → (optional TTA) → (optional ensemble) → post-process → save mask.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from preprocessing.transforms import FacePreprocessor
from postprocessing.pipeline import PostProcessor
from inference.tta import TTAPredictor
from inference.ensemble import CheckpointEnsemble
from utils.helpers import get_device, load_checkpoint, Timer, colorize_mask


class InferencePipeline:
    """End-to-end inference pipeline with configurable TTA, ensemble, and post-processing."""

    def __init__(
        self,
        model_fn,
        cfg: Dict[str, Any],
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg
        self.device = device or get_device()
        self.num_classes = cfg.get("num_classes", 19)
        self.image_size = cfg.get("image_size", 512)

        # Preprocessing
        self.preprocessor = FacePreprocessor(cfg.get("preprocessing", {}))

        # Model
        self.model = model_fn()
        if checkpoint_path is not None:
            load_checkpoint(checkpoint_path, self.model, device=self.device)
        self.model = self.model.to(self.device).eval()

        # TTA
        tta_cfg = cfg.get("tta", {})
        self.use_tta = tta_cfg.get("enabled", False)
        self.tta = TTAPredictor(tta_cfg) if self.use_tta else None

        # Ensemble
        ens_cfg = cfg.get("ensemble", {})
        self.use_ensemble = ens_cfg.get("enabled", False)
        self.ensemble = (
            CheckpointEnsemble(model_fn, cfg, self.device)
            if self.use_ensemble
            else None
        )

        # Post-processing
        pp_cfg = cfg.get("postprocessing", {})
        self.postprocessor = PostProcessor(pp_cfg)

    @torch.no_grad()
    def predict_single(self, image_path: str) -> np.ndarray:
        """
        Run full pipeline on a single image.

        Args:
            image_path: Path to RGB image.

        Returns:
            Integer label mask (H, W).
        """
        # Load original image
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # Resize for model
        img_resized = cv2.resize(
            img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )

        # Preprocess
        img_proc = self.preprocessor(img_resized)  # (H, W, C) float32
        img_tensor = torch.from_numpy(img_proc.transpose(2, 0, 1)).unsqueeze(0).float()

        # Forward pass (choose: ensemble > TTA > standard)
        if self.use_ensemble:
            probs = self.ensemble.predict(img_tensor)
        elif self.use_tta:
            probs = self.tta.predict(self.model, img_tensor, self.device)
        else:
            logits = self.model(img_tensor.to(self.device))
            probs = F.softmax(logits, dim=1)

        # To numpy (C, H, W)
        probs_np = probs.squeeze(0).cpu().numpy()

        # Post-processing (CRF needs original-resolution uint8 image)
        img_for_crf = cv2.resize(
            img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        mask = self.postprocessor(probs_np, img_for_crf, self.num_classes)

        # Resize mask back to original resolution
        if (orig_h, orig_w) != (self.image_size, self.image_size):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

        return mask

    def predict_batch(self, image_dir: str, output_dir: str) -> None:
        """
        Run inference on all images in a directory and save masks.

        Args:
            image_dir: Path to input image directory.
            output_dir: Path to save output masks (as PNG).
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        img_dir = Path(image_dir)

        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
        image_files = sorted(
            f for f in img_dir.iterdir() if f.suffix.lower() in extensions
        )

        print(f"Running inference on {len(image_files)} images...")
        for img_file in image_files:
            with Timer() as t:
                mask = self.predict_single(str(img_file))
            # Save mask using PIL with palette
            save_path = out_path / f"{img_file.stem}.png"
            colorized = colorize_mask(mask)
            colorized.save(str(save_path))
            print(f"  {img_file.name} → {save_path.name}  ({t.elapsed:.3f}s)")

        print(f"Done. Masks saved to {output_dir}")
