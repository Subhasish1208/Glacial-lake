import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from dbcnet import DBCNet
from torchvision.transforms import functional as TF

def generate_visuals():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = r"c:\Users\sm080\Downloads\glacial lake dataset\glacial-lake-dataset"
    batch_size = 4
    
    _, _, test_loader = get_dataloaders(data_dir, batch_size=batch_size, num_workers=0)
    
    model = DBCNet(in_channels=3, out_channels=1).to(device)
    model_path = os.path.join(os.path.dirname(data_dir), "dbcnet_glacial_lakes", "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    output_dir = os.path.join(os.path.dirname(data_dir), "dbcnet_glacial_lakes", "output_visuals")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get one batch of images
    images, masks = next(iter(test_loader))
    images = images.to(device)
    masks = masks.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).float()
    
    # Move to CPU for plotting
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()
    
    # Assuming images were normalized with some mean and std (if not, just clip)
    # The dataloader might have ToTensor which normalizes to [0,1], or Normalize.
    # We will just normalize them to [0, 1] for visualization.
    
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
    
    for i in range(batch_size):
        # Image
        img = np.transpose(images[i], (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        
        # Mask
        mask = masks[i][0]
        
        # Prediction
        pred = preds[i][0]
        
        # Overlay
        overlay = img.copy()
        overlay[pred == 1, 0] = 1.0  # Red overlay for lake predictions
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")
        
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Overlay (Pred in Red)")
        axes[i, 3].axis("off")
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, "prediction_results.png")
    plt.savefig(save_path, dpi=300)
    print(f"Visuals saved to {save_path}")

if __name__ == "__main__":
    generate_visuals()
