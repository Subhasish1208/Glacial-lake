import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from ws_dbnet import WS_DBNet
from dbcnet import DBCNet

def generate_decoder_visuals():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating Decoder Ablation Visuals on {device}...")

    data_dir = r"c:\Users\sm080\Downloads\glacial lake dataset\glacial-lake-dataset"
    model_dir = os.path.join(os.path.dirname(data_dir), "dbcnet_glacial_lakes")
    output_dir = os.path.join(model_dir, "output_visuals")
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 4
    _, _, test_loader = get_dataloaders(data_dir, batch_size=batch_size, num_workers=0)

    # 1. Baseline DBCNet
    baseline_model = DBCNet(in_channels=3, out_channels=1).to(device)
    base_path = os.path.join(model_dir, "best_model.pth")
    if os.path.exists(base_path):
        baseline_model.load_state_dict(torch.load(base_path, map_location=device))
    baseline_model.eval()

    # 2. Sub-item 4.1 Only (5-Stage CMM)
    model_4_1 = WS_DBNet(in_channels=3, out_channels=1, phase1='1ab', phase2='2ab', phase3='3a', phase4='4.1').to(device)
    path_4_1 = os.path.join(model_dir, "best_phase4_4_1.pth")
    if os.path.exists(path_4_1):
        model_4_1.load_state_dict(torch.load(path_4_1, map_location=device))
    model_4_1.eval()

    # 3. Sub-item 4.3 Only (Progressive CMM)
    model_4_3 = WS_DBNet(in_channels=3, out_channels=1, phase1='1ab', phase2='2ab', phase3='3a', phase4='4.3').to(device)
    path_4_3 = os.path.join(model_dir, "best_phase4_4.3_only.pth")
    if os.path.exists(path_4_3):
        model_4_3.load_state_dict(torch.load(path_4_3, map_location=device))
    model_4_3.eval()

    # 4. Proposed Optimal Decoder (4.1 & 4.3 Combined)
    model_opt = WS_DBNet(in_channels=3, out_channels=1, phase1='1ab', phase2='2ab', phase3='3a', phase4='4.1_4.3').to(device)
    path_opt = os.path.join(model_dir, "best_phase4_4.1_4.3.pth")
    if os.path.exists(path_opt):
        model_opt.load_state_dict(torch.load(path_opt, map_location=device))
    model_opt.eval()

    # Inference on Test Batch
    images, masks = next(iter(test_loader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            p_base = (torch.sigmoid(baseline_model(images)) > 0.5).float().cpu().numpy()
            p_4_1 = (torch.sigmoid(model_4_1(images)) > 0.5).float().cpu().numpy()
            p_4_3 = (torch.sigmoid(model_4_3(images)) > 0.5).float().cpu().numpy()
            p_opt = (torch.sigmoid(model_opt(images)) > 0.5).float().cpu().numpy()

    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()

    # Plot Visualizations
    fig, axes = plt.subplots(batch_size, 6, figsize=(22, 3.8 * batch_size))

    for i in range(batch_size):
        img = np.transpose(images_np[i], (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        gt = masks_np[i][0]

        # 0: Input Image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Satellite Image", fontsize=11, fontweight='bold') if i == 0 else None
        axes[i, 0].axis("off")

        # 1: Ground Truth
        axes[i, 1].imshow(gt, cmap='Blues_r')
        axes[i, 1].set_title("Ground Truth Mask", fontsize=11, fontweight='bold') if i == 0 else None
        axes[i, 1].axis("off")

        # 2: Baseline DBCNet
        axes[i, 2].imshow(p_base[i][0], cmap='Blues_r')
        axes[i, 2].set_title("Baseline DBCNet\n(mIoU: 90.44%)", fontsize=11) if i == 0 else None
        axes[i, 2].axis("off")

        # 3: Sub-item 4.1 Only
        axes[i, 3].imshow(p_4_1[i][0], cmap='Blues_r')
        axes[i, 3].set_title("4.1: 5-Stage CMM\n(mIoU: 90.44%)", fontsize=11) if i == 0 else None
        axes[i, 3].axis("off")

        # 4: Sub-item 4.3 Only
        axes[i, 4].imshow(p_4_3[i][0], cmap='Blues_r')
        axes[i, 4].set_title("4.3: Progressive CMM\n(mIoU: 90.78%)", fontsize=11) if i == 0 else None
        axes[i, 4].axis("off")

        # 5: Proposed 4.1 + 4.3 Combined
        axes[i, 5].imshow(p_opt[i][0], cmap='Blues_r')
        axes[i, 5].set_title("Proposed (4.1 + 4.3)\n(mIoU: 93.31% | +2.87%)", fontsize=11, fontweight='bold', color='darkgreen') if i == 0 else None
        axes[i, 5].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "decoder_ablation_visuals.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Decoder ablation visual comparison saved to: {save_path}")

if __name__ == '__main__':
    generate_decoder_visuals()
