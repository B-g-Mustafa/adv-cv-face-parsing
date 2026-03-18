import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SimpleUNet
from dataset import CelebAMaskDataset

# --- SETTINGS ---
DATA_DIR = "/kaggle/working/split_data"
EPOCHS = 10
BATCH_SIZE = 8
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME_MODEL = "face_parsing_unet_v2.pth"  # Set to None or empty string to train from scratch

def f_measure(preds, targets, num_classes=19):
    """Original Kaggle notebook F1 measure."""
    preds = torch.argmax(preds, dim=1)
    f1_scores = []
    for cls in range(num_classes):
        tp = torch.sum((preds == cls) & (targets == cls)).float()
        fp = torch.sum((preds == cls) & (targets != cls)).float()
        fn = torch.sum((preds != cls) & (targets == cls)).float()
        
        if tp + fp + fn == 0:
            f1_scores.append(1.0)  
        else:
            f1_scores.append((2 * tp) / (2 * tp + fp + fn + 1e-8))
    
    return torch.mean(torch.tensor(f1_scores)).item()

def train(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    total_f1 = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            total_f1 += f_measure(outputs, masks)
    
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, F1 Score: {total_f1/len(val_loader):.4f}")

if __name__ == "__main__":
    train_dir_img = os.path.join(DATA_DIR, "train/images")
    train_dir_mask = os.path.join(DATA_DIR, "train/masks")
    val_dir_img = os.path.join(DATA_DIR, "val/images")
    val_dir_mask = os.path.join(DATA_DIR, "val/masks")
    
    train_dataset = CelebAMaskDataset(train_dir_img, train_dir_mask, is_train=True)
    val_dataset = CelebAMaskDataset(val_dir_img, val_dir_mask, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleUNet(in_channels=3, out_channels=19).to(DEVICE)
    
    # --- RESUME LOGIC ---
    if RESUME_MODEL and os.path.exists(RESUME_MODEL):
        print(f"Loading weights from {RESUME_MODEL} to resume training...")
        model.load_state_dict(torch.load(RESUME_MODEL, map_location=DEVICE))
    else:
        print("Training from scratch with random weights...")

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Starting Training...")
    train(model, train_loader, criterion, optimizer, epochs=EPOCHS)
    
    print("Running Validation...")
    validate(model, val_loader, criterion)

    # Save Model
    torch.save(model.state_dict(), "face_parsing_unet_v2.pth")
    print("Model saved successfully as face_parsing_unet_v2.pth!")
