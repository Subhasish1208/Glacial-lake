import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class GlacialLakeDataset(Dataset):
    def __init__(self, data_dir, split, transform=None, mean=None, std=None):
        """
        Args:
            data_dir: Path to the dataset directory (containing images/ and masks/)
            split: "train", "val", or "test"
            transform: Albumentations transforms
            mean: List of channel means
            std: List of channel stds
        """
        self.data_dir = data_dir
        self.split = split
        
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "masks")
        
        splits_file = os.path.join(os.path.dirname(data_dir), "dbcnet_glacial_lakes", "dataset_splits.json")
        with open(splits_file, 'r') as f:
            splits_data = json.load(f)
            
        self.samples = splits_data[split]
        self.mean = mean if mean else splits_data.get("mean", [0.5, 0.5, 0.5])
        self.std = std if std else splits_data.get("std", [0.5, 0.5, 0.5])
        
        self.transform = transform
        
        # We need mapping from basename to exact filename
        image_files = os.listdir(self.images_dir)
        mask_files = os.listdir(self.masks_dir)
        
        self.image_map = {os.path.splitext(f)[0]: f for f in image_files}
        self.mask_map = {os.path.splitext(f)[0]: f for f in mask_files}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        basename = self.samples[idx]
        img_name = self.image_map[basename]
        mask_name = self.mask_map[basename]
        
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Load image (RGB)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load mask (Grayscale)
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 127).astype(np.float32) # Binary 0/1 mask
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Mask needs to have a channel dimension for BCE loss (C, H, W)
        mask = mask.unsqueeze(0)
        
        return image, mask

def get_dataloaders(data_dir, batch_size=2, num_workers=4):
    splits_file = os.path.join(os.path.dirname(data_dir), "dbcnet_glacial_lakes", "dataset_splits.json")
    with open(splits_file, 'r') as f:
        splits_data = json.load(f)
    
    mean = splits_data["mean"]
    std = splits_data["std"]
    
    train_transform = A.Compose([
        A.Resize(512, 512),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    
    train_dataset = GlacialLakeDataset(data_dir, "train", transform=train_transform, mean=mean, std=std)
    val_dataset = GlacialLakeDataset(data_dir, "val", transform=val_transform, mean=mean, std=std)
    test_dataset = GlacialLakeDataset(data_dir, "test", transform=val_transform, mean=mean, std=std)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    data_dir = r"c:\Users\sm080\Downloads\glacial lake dataset\glacial-lake-dataset"
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=2, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    for images, masks in train_loader:
        print("Image batch shape:", images.shape)
        print("Mask batch shape:", masks.shape)
        print("Image min/max:", images.min().item(), images.max().item())
        print("Mask unique values:", torch.unique(masks))
        break
