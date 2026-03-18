import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import SimpleUNet
from postprocess import SpatialHeuristicFixer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use standard CelebAMask Color Map instead of matplotlib jet!
CELEBAMASK_COLORS = [
    [0, 0, 0],       # 0: background
    [204, 0, 0],     # 1: skin
    [76, 153, 0],    # 2: l_brow
    [204, 204, 0],   # 3: r_brow
    [51, 51, 255],   # 4: l_eye
    [204, 0, 204],   # 5: r_eye
    [0, 255, 255],   # 6: eye_g
    [255, 204, 204], # 7: l_ear
    [102, 51, 0],    # 8: r_ear
    [255, 0, 0],     # 9: ear_r
    [102, 204, 0],   # 10: nose
    [255, 255, 0],   # 11: mouth
    [0, 0, 153],     # 12: u_lip
    [0, 0, 204],     # 13: l_lip
    [255, 51, 153],  # 14: neck
    [0, 204, 204],   # 15: neck_l
    [0, 51, 0],      # 16: cloth
    [255, 153, 51],  # 17: hair
    [0, 204, 0]      # 18: hat
]

def apply_colormap(mask_array):
    """Converts a 2D class-index mask into a colorized RGB Pillow image."""
    h, w = mask_array.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(19):
        color_mask[mask_array == cls_id] = CELEBAMASK_COLORS[cls_id]
    return Image.fromarray(color_mask)

def infer(image_path, model, output_folder="outpt_masks"):
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.basename(image_path)
    
    # Preprocess
    frame = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(frame).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Apply Spatial Hack! (This cleanly un-merges the classes fused by horizontal_flip)
    fixer = SpatialHeuristicFixer()
    mask = fixer(mask)

    # Colorize and Resize
    mask_colored = apply_colormap(mask)
    save_path = os.path.join(output_folder, filename.replace(".jpg", ".png"))
    mask_colored.save(save_path)
    print(f"✅ Mask saved: {save_path}")

    # Blend with native image for easy visual debugging
    mask_colored_resized = mask_colored.resize(frame.size, Image.BILINEAR)
    overlay = Image.blend(frame, mask_colored_resized, alpha=0.4)
    overlay_path = os.path.join(output_folder, f"overlay_{filename}")
    overlay.save(overlay_path)
    print(f"✅ Overlay saved: {overlay_path}")


if __name__ == "__main__":
    model = SimpleUNet(in_channels=3, out_channels=19).to(DEVICE)
    
    checkpoint_path = "face_parsing_unet_v2.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"Loaded {checkpoint_path}")
    else:
        print(f"Warning: {checkpoint_path} not found. Running with random weights.")
        
    model.eval()
    
    # Try it on a sample Kaggle test image
    test_img = "/kaggle/input/testimagesfor-this/test/images/sample.jpg"
    if os.path.exists(test_img):
        infer(test_img, model, output_folder="/kaggle/working/outpt_masks")
    else:
        print("Please provide a valid test image path!")
