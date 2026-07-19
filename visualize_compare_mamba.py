import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from dbcnet import DBCNet
from deeplabv3_model import get_deeplabv3_model
from vmamba_seg import VMambaSeg

def generate_mamba_comparison():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = r"c:\Users\sm080\Downloads\glacial lake dataset\glacial-lake-dataset"
    batch_size = 4
    
    _, _, test_loader = get_dataloaders(data_dir, batch_size=batch_size, num_workers=0)
    
    # Load DBCNet model
    dbcnet = DBCNet(in_channels=3, out_channels=1).to(device)
    dbcnet_path = os.path.join(os.path.dirname(data_dir), "dbcnet_glacial_lakes", "best_model.pth")
    dbcnet.load_state_dict(torch.load(dbcnet_path, map_location=device))
    dbcnet.eval()
    
    # Load DeepLabV3 model
    deeplabv3 = get_deeplabv3_model(pretrained=True, num_classes=1).to(device)
    deeplabv3_path = os.path.join(os.path.dirname(data_dir), "dbcnet_glacial_lakes", "best_deeplabv3.pth")
    deeplabv3.load_state_dict(torch.load(deeplabv3_path, map_location=device))
    deeplabv3.eval()
    
    # Load VMambaSeg model
    vmamba = VMambaSeg(in_channels=3, out_channels=1).to(device)
    vmamba_path = os.path.join(os.path.dirname(data_dir), "dbcnet_glacial_lakes", "best_vmamba.pth")
    vmamba.load_state_dict(torch.load(vmamba_path, map_location=device))
    vmamba.eval()
    
    output_dir = os.path.join(os.path.dirname(data_dir), "dbcnet_glacial_lakes", "output_visuals")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get one batch of images
    images, masks = next(iter(test_loader))
    images = images.to(device)
    masks = masks.to(device)
    
    with torch.no_grad():
        dbcnet_out = dbcnet(images)
        dbcnet_preds = (torch.sigmoid(dbcnet_out) > 0.5).float()
        
        deeplabv3_out = deeplabv3(images)['out']
        deeplabv3_preds = (torch.sigmoid(deeplabv3_out) > 0.5).float()
        
        vmamba_out = vmamba(images)
        vmamba_preds = (torch.sigmoid(vmamba_out) > 0.5).float()
        
    # Move to CPU for plotting
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    dbcnet_preds = dbcnet_preds.cpu().numpy()
    deeplabv3_preds = deeplabv3_preds.cpu().numpy()
    vmamba_preds = vmamba_preds.cpu().numpy()
    
    fig, axes = plt.subplots(batch_size, 6, figsize=(20, 3.5 * batch_size))
    
    for i in range(batch_size):
        # Image
        img = np.transpose(images[i], (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        
        # Mask
        mask = masks[i][0]
        
        # DBCNet Pred
        db_pred = dbcnet_preds[i][0]
        
        # DeepLabV3 Pred
        dl_pred = deeplabv3_preds[i][0]
        
        # VMambaSeg Pred
        vm_pred = vmamba_preds[i][0]
        
        # Plotting
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image") if i == 0 else None
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth Mask") if i == 0 else None
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(db_pred, cmap='gray')
        axes[i, 2].set_title("DBCNet (93.03% mIoU)") if i == 0 else None
        axes[i, 2].axis("off")
        
        axes[i, 3].imshow(dl_pred, cmap='gray')
        axes[i, 3].set_title("DeepLabV3 (87.60% mIoU)") if i == 0 else None
        axes[i, 3].axis("off")
        
        axes[i, 4].imshow(vm_pred, cmap='gray')
        axes[i, 4].set_title("VMambaSeg (93.65% mIoU)") if i == 0 else None
        axes[i, 4].axis("off")
        
        # VMambaSeg Prediction Overlay
        vm_overlay = img.copy()
        vm_overlay[vm_pred == 1, 0] = 1.0  # Red overlay for VMamba prediction
        axes[i, 5].imshow(vm_overlay)
        axes[i, 5].set_title("VMambaSeg Overlay") if i == 0 else None
        axes[i, 5].axis("off")
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, "mamba_comparison_results.png")
    plt.savefig(save_path, dpi=300)
    print(f"Mamba comparison visuals saved to {save_path}")

if __name__ == "__main__":
    generate_mamba_comparison()
