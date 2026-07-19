import os
import torch
from dataset import get_dataloaders
from vmamba_seg import VMambaSeg
from train import calculate_metrics
from tqdm import tqdm

def evaluate_test_set():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for evaluation: {device}")
    
    data_dir = r"c:\Users\sm080\Downloads\glacial lake dataset\glacial-lake-dataset"
    batch_size = 2
    
    _, _, test_loader = get_dataloaders(data_dir, batch_size=batch_size, num_workers=0)
    
    model = VMambaSeg(in_channels=3, out_channels=1).to(device)
    model_path = os.path.join(os.path.dirname(data_dir), "dbcnet_glacial_lakes", "best_vmamba.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded best VMamba model from {model_path}")
    else:
        print(f"Error: Model weights not found at {model_path}")
        return
        
    model.eval()
    
    test_prec, test_rec, test_f1, test_iou = 0.0, 0.0, 0.0, 0.0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing VMambaSeg")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            p, r, f, iou = calculate_metrics(outputs, masks)
            test_prec += p
            test_rec += r
            test_f1 += f
            test_iou += iou
            
    test_prec /= len(test_loader)
    test_rec /= len(test_loader)
    test_f1 /= len(test_loader)
    test_iou /= len(test_loader)
    
    print("\n" + "="*40)
    print("Final VMambaSeg Evaluation on Test Set")
    print("="*40)
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1-score:  {test_f1:.4f}")
    print(f"mIoU:      {test_iou:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_test_set()
