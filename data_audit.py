import os
import random
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

def audit_dataset():
    data_dir = r"c:\Users\sm080\Downloads\glacial lake dataset\glacial-lake-dataset"
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    
    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))
    
    print(f"Total images found: {len(image_files)}")
    print(f"Total masks found: {len(mask_files)}")
    
    # Check if image and mask names match
    image_basenames = [os.path.splitext(f)[0] for f in image_files]
    mask_basenames = [os.path.splitext(f)[0] for f in mask_files]
    
    missing_masks = set(image_basenames) - set(mask_basenames)
    if missing_masks:
        print(f"WARNING: Missing masks for images: {missing_masks}")
    
    # Create splits (70/15/15)
    random.seed(42)
    valid_samples = list(set(image_basenames).intersection(set(mask_basenames)))
    random.shuffle(valid_samples)
    
    total_valid = len(valid_samples)
    train_end = int(0.7 * total_valid)
    val_end = int(0.85 * total_valid)
    
    train_samples = valid_samples[:train_end]
    val_samples = valid_samples[train_end:val_end]
    test_samples = valid_samples[val_end:]
    
    print(f"Train split: {len(train_samples)}")
    print(f"Val split: {len(val_samples)}")
    print(f"Test split: {len(test_samples)}")
    
    # Calculate Mean and Std on Train set
    print("Calculating Mean and Std on Train set...")
    pixel_num = 0
    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)
    
    for sample in tqdm(train_samples):
        # We assume jpg, png, or tif. Let's find the exact extension.
        img_name = [f for f in image_files if f.startswith(sample + ".")][0]
        img_path = os.path.join(images_dir, img_name)
        
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img) / 255.0 # Scale to 0-1
        
        pixel_num += (img_np.shape[0] * img_np.shape[1])
        channel_sum += np.sum(img_np, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(img_np), axis=(0, 1))
        
    mean = channel_sum / pixel_num
    std = np.sqrt(channel_sum_squared / pixel_num - np.square(mean))
    
    print(f"Dataset Mean: {mean}")
    print(f"Dataset Std: {std}")
    
    # Save splits and stats
    output = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
        "mean": mean.tolist(),
        "std": std.tolist()
    }
    
    output_path = r"c:\Users\sm080\Downloads\glacial lake dataset\dbcnet_glacial_lakes\dataset_splits.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"Splits and stats saved to {output_path}")
    
    # Check a mask
    if mask_files:
        mask_path = os.path.join(masks_dir, mask_files[0])
        mask = Image.open(mask_path)
        mask_np = np.array(mask)
        print(f"Sample mask shape: {mask_np.shape}, unique values: {np.unique(mask_np)}")

if __name__ == '__main__':
    audit_dataset()
