import torch

# ==============================================================================
# LITE CONFIGURATION FOR 90% F1 TRAINING
# No YAML parsers, no complexity. Just pure Python variables.
# ==============================================================================
CFG = {
    "data": {
        "train_img_dir": "/home/msai/birul001/advcv-celeb-face-parsing/data/split-data-1000/train/images",
        "train_mask_dir": "/home/msai/birul001/advcv-celeb-face-parsing/data/split-data-1000/train/masks",
        "val_img_dir": "/home/msai/birul001/advcv-celeb-face-parsing/data/split-data-1000/val/images",
        "val_mask_dir": "/home/msai/birul001/advcv-celeb-face-parsing/data/split-data-1000/val/masks",
        "test_img_dir": "/home/msai/birul001/advcv-celeb-face-parsing/data/split-data-1000/test/images",
    },
    
    # ── 1. HYPERPARAMETERS ────────────────────────────────────────────────────
    "epochs": 150,
    "batch_size": 16,
    "learning_rate": 5e-4,     # <--- OPTIMAL AdamW LR for UNet from scratch
    "weight_decay": 1e-4,
    
    # ── 2. ARCHITECTURE & LOSS ────────────────────────────────────────────────
    "image_size": 512,
    "num_classes": 19,
    "loss": {
        "use_class_weights": False,  # CRITICAL: Prevent double-balancing punishment
        "dice_weight": 1.0,
        "ce_weight": 1.0,
        "boundary_weight": 0.5,
        "label_smoothing": 0.05
    },
    
    # ── 3. AUGMENTATION & PREPROCESSING ───────────────────────────────────────
    "preprocessing": {
        "use_clahe": True,
        "clahe_clip_limit": 2.0,
        "clahe_grid_size": 8,
        "use_bilateral": True,
        "use_sobel_edge": True,      # Gives model perfect physical edge hints (4th channel)
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225]
    },
    "augmentation": {
        "enabled": True,
        "horizontal_flip_p": 0.5,    # STRATEGY: 2x shape data (we fix L/R symmetry in inference hack)
        "rotate_limit": 15,
        "scale_limit": 0.1,
        "shift_limit": 0.05
    },
    "copy_paste": {
        "enabled": True,
        "probability": 0.5,          # Boosts rare class accuracy immensely
        "max_paste_per_image": 3,
        "blend_sigma": 3,
        "target_classes": [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
    },
    
    # ── 4. SYSTEM ─────────────────────────────────────────────────────────────
    "device": torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
    "save_dir": "lite_outputs",
    "num_workers": 4
}
