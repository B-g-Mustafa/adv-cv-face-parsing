import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

class CelebAMaskDataset(Dataset):
    """
    Cleaned dataset loader.
    Includes the 'Genius Strategy' Horizontal Flip to double shape data
    for training masks by intentionally causing left/right class fusion.
    """
    def __init__(self, image_dir, mask_dir, is_train=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        # Only load mapping for files that exist
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        
        # Mask might be .png while image is .jpg
        base_name = os.path.splitext(img_filename)[0]
        mask_path = os.path.join(self.mask_dir, f"{base_name}.png")
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # 1. Deterministic Resize (Nearest neighbor for mask is CRITICAL!)
        image = TF.resize(image, (512, 512))
        mask = TF.resize(mask, (512, 512), interpolation=Image.NEAREST)
        
        # 2. Intentional Left/Right Shape Fusion Hack
        if self.is_train and random.random() > 0.5:
            # Flips BOTH image and mask visually without swapping the class IDs internally
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            
        # 3. To Tensor
        image_tensor = TF.to_tensor(image) # Normalizes to [0,1]
        mask_tensor = torch.tensor(np.array(mask), dtype=torch.long)
        
        return image_tensor, mask_tensor
