import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from deeplabv3_model import get_deeplabv3_model
from tqdm import tqdm
from train import DiceLoss, calculate_metrics

def train_deeplabv3():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = r"c:\Users\sm080\Downloads\glacial lake dataset\glacial-lake-dataset"
    batch_size = 2
    epochs = 80
    warmup_epochs = 4
    
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=batch_size, num_workers=0)
    
    # We use pretrained=True as standard for DeepLabV3
    model = get_deeplabv3_model(pretrained=True, num_classes=1).to(device)
    
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    total_iters = epochs * len(train_loader)
    warmup_iters = warmup_epochs * len(train_loader)
    
    def lr_lambda(current_step):
        if current_step < warmup_iters:
            return float(current_step) / float(max(1, warmup_iters))
        return max(0.0, (1.0 - (current_step - warmup_iters) / (total_iters - warmup_iters)) ** 0.9)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_iou = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # DeepLabV3 returns a dict: {'out': ..., 'aux': ...}
            outputs = model(images)['out']
            
            loss_bce = criterion_bce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = loss_bce + loss_dice
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_prec, val_rec, val_f1, val_iou = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)['out']
                
                loss_bce = criterion_bce(outputs, masks)
                loss_dice = criterion_dice(outputs, masks)
                loss = loss_bce + loss_dice
                val_loss += loss.item()
                
                p, r, f, iou = calculate_metrics(outputs, masks)
                val_prec += p
                val_rec += r
                val_f1 += f
                val_iou += iou
                
        val_loss /= len(val_loader)
        val_prec /= len(val_loader)
        val_rec /= len(val_loader)
        val_f1 /= len(val_loader)
        val_iou /= len(val_loader)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Metrics - Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}, mIoU: {val_iou:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(os.path.dirname(data_dir), "dbcnet_glacial_lakes", "best_deeplabv3.pth"))
            print(f"--> Saved best DeepLabV3 model with mIoU: {best_iou:.4f}")
            
    print("Training finished.")

if __name__ == "__main__":
    train_deeplabv3()
