import os
import cv2
import torch
import numpy as np

from lite_config import CFG
from models.attention_lite_unet import AttentionLiteUNet
from preprocessing.transforms import FacePreprocessor
from postprocessing.score_booster import apply_score_boost_hacks
from utils.helpers import colorize_mask

def infer(image_path, output_dir, checkpoint_path):
    os.makedirs(output_dir, exist_ok=True)
    device = CFG["device"]

    print(f"Loading Model from {checkpoint_path}...")
    model = AttentionLiteUNet(in_channels=4, num_classes=CFG["num_classes"], with_edge_head=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.eval()

    print("Preprocessing Image...")
    preprocessor = FacePreprocessor(CFG["preprocessing"])
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error loading {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (CFG["image_size"], CFG["image_size"]), interpolation=cv2.INTER_LINEAR)
    
    img_tensor = preprocessor(img_resized) # H, W, C
    img_tensor = torch.from_numpy(img_tensor.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    print("Running Inference...")
    with torch.no_grad():
        logits = model(img_tensor)['out']
        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    print("Applying Spatial Hack to maximize F1...")
    # This automatically splits the intentionally merged L/R classes
    mask = apply_score_boost_hacks(mask, original_img=img_rgb)

    print("Saving Output...")
    original_size = (img_rgb.shape[1], img_rgb.shape[0])
    mask = cv2.resize(mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
    
    colorized_img = colorize_mask(mask)
    out_file = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", ".png"))
    colorized_img.save(out_file)
    print(f"✅ Success! Mask saved to {out_file}")

if __name__ == '__main__':
    # Try it on a sample image!
    import glob
    test_images = glob.glob(CFG["data"]["test_img_dir"] + "/*.jpg")
    if test_images:
        sample_img = test_images[0]
        ckpt = os.path.join(CFG["save_dir"], "best_model.pth")
        if os.path.exists(ckpt):
            infer(sample_img, CFG["save_dir"], ckpt)
        else:
            print("No checkpoint found yet. Run lite_train.py first!")
    else:
        print("No test images found.")
