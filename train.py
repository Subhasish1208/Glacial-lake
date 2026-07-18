import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from dataset import get_dataloaders
from dbcnet import DBCNet
from tqdm import tqdm

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        intersection = torch.sum(predict * target)
        union = torch.sum(predict) + torch.sum(target)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.max_iters = max_iters
        self.power = power
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]

def calculate_metrics(predict, target, threshold=0.5):
    predict = (torch.sigmoid(predict) > threshold).float()
    
    tp = torch.sum(predict * target)
    fp = torch.sum(predict * (1 - target))
    fn = torch.sum((1 - predict) * target)
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    iou = tp / (tp + fp + fn + 1e-6)
    
    return precision.item(), recall.item(), f1.item(), iou.item()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = r"c:\Users\sm080\Downloads\glacial lake dataset\glacial-lake-dataset"
    batch_size = 2
    epochs = 80
    warmup_epochs = 4
    
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=batch_size, num_workers=0)
    
    model = DBCNet(in_channels=3, out_channels=1).to(device)
    
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
            
            outputs = model(images)
            
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
                
                outputs = model(images)
                
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
            torch.save(model.state_dict(), os.path.join(os.path.dirname(data_dir), "dbcnet_glacial_lakes", "best_model.pth"))
            print(f"--> Saved best model with mIoU: {best_iou:.4f}")
            
    print("Training finished.")

if __name__ == "__main__":
    train()
