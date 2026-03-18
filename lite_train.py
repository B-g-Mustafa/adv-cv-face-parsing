import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from lite_config import CFG
from models.attention_lite_unet import AttentionLiteUNet
from datasets.face_dataset import FaceParsingDataset
from training.losses import CombinedLoss
from evaluation.metrics import compute_multiclass_fscore

def train():
    os.makedirs(CFG["save_dir"], exist_ok=True)
    device = CFG["device"]

    print("Loading Datasets...")
    train_ds = FaceParsingDataset(CFG["data"]["train_img_dir"], CFG["data"]["train_mask_dir"], split="train", cfg=CFG)
    val_ds = FaceParsingDataset(CFG["data"]["val_img_dir"], CFG["data"]["val_mask_dir"], split="val", cfg=CFG)
    
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=CFG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=CFG["num_workers"])

    print("Initializing Model & Optimizer...")
    # in_channels=4 because we use Sobel Edge 4th channel
    model = AttentionLiteUNet(in_channels=4, num_classes=CFG["num_classes"], with_edge_head=True).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    criterion = CombinedLoss(cfg=CFG)

    best_f1 = 0.0

    print("Starting Training Loop...")
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG['epochs']} [TRAIN]")
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            edges = batch['edge'].to(device) if 'edge' in batch else None

            optimizer.zero_grad()
            outputs = model(images)
            
            targets = {'mask': masks, 'edge': edges}
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        
        model.eval()
        val_f1s = []
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{CFG['epochs']} [VAL]")
        with torch.no_grad():
            for batch in pbar_val:
                images = batch['image'].to(device)
                masks = batch['mask'].numpy() # Keep ground truth on CPU
                
                outputs = model(images)
                preds = outputs['out'].argmax(dim=1).cpu().numpy()
                
                for gt, pr in zip(masks, preds):
                    val_f1s.append(compute_multiclass_fscore(gt, pr))
                    
        epoch_f1 = sum(val_f1s) / len(val_f1s)
        avg_loss = train_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\n=> Epoch {epoch} Summary: Loss: {avg_loss:.4f} | Val F1: {epoch_f1:.4f} | LR: {current_lr:.2e}")

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            save_path = os.path.join(CFG["save_dir"], "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"★ Saved new best model to {save_path} (F1: {best_f1:.4f})")

if __name__ == '__main__':
    train()
