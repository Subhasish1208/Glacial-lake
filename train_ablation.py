import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
import json
import argparse
import torch
import torch.optim as optim
from dataset import get_dataloaders
from ws_dbnet import WS_DBNet
from losses import CombinedWSLoss
from train import calculate_metrics
from tqdm import tqdm

def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Running Experiment: {args.exp_name} on {device} ===")
    print(f"Switches -> P1: {args.phase1}, P2: {args.phase2}, P3: {args.phase3}, P4: {args.phase4}, BoundaryLoss: {args.use_boundary}, clDiceLoss: {args.use_cldice}")

    data_dir = r"c:\Users\sm080\Downloads\glacial lake dataset\glacial-lake-dataset"
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=args.batch_size, num_workers=0)

    model = WS_DBNet(
        in_channels=3,
        out_channels=1,
        phase1=args.phase1,
        phase2=args.phase2,
        phase3=args.phase3,
        phase4=args.phase4
    ).to(device)

    criterion = CombinedWSLoss(
        use_boundary=args.use_boundary,
        use_cldice=args.use_cldice,
        lambda_cldice=0.5
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    total_iters = args.epochs * len(train_loader)
    warmup_iters = 4 * len(train_loader)

    def lr_lambda(current_step):
        if current_step < warmup_iters:
            return float(current_step) / float(max(1, warmup_iters))
        return max(0.0, (1.0 - (current_step - warmup_iters) / max(1, total_iters - warmup_iters)) ** 0.9)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_iou = 0.0
    best_weights_path = os.path.join(os.path.dirname(data_dir), "dbcnet_glacial_lakes", f"best_{args.exp_name}.pth")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss, val_prec, val_rec, val_f1, val_iou = 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
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

        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f} (F1: {val_f1:.4f})", flush=True)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_weights_path)

    # Evaluate on Test Set using Best Saved Model
    print(f"\n--> Evaluating Best Model for {args.exp_name} on Test Split...")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    test_prec, test_rec, test_f1, test_iou = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
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

    res_dict = {
        "exp_name": args.exp_name,
        "phase1": args.phase1,
        "phase2": args.phase2,
        "phase3": args.phase3,
        "phase4": args.phase4,
        "use_boundary": args.use_boundary,
        "use_cldice": args.use_cldice,
        "test_precision": round(test_prec * 100, 2),
        "test_recall": round(test_rec * 100, 2),
        "test_f1": round(test_f1 * 100, 2),
        "test_miou": round(test_iou * 100, 2)
    }

    print(f"[{args.exp_name} TEST RESULTS] Precision: {res_dict['test_precision']}% | Recall: {res_dict['test_recall']}% | F1: {res_dict['test_f1']}% | mIoU: {res_dict['test_miou']}%\n")

    json_path = os.path.join(os.path.dirname(data_dir), "dbcnet_glacial_lakes", "ablation_results.json")
    results_data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                results_data = json.load(f)
        except Exception:
            results_data = {}

    results_data[args.exp_name] = res_dict
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--phase1', type=str, default='v1')
    parser.add_argument('--phase2', type=str, default='v1')
    parser.add_argument('--phase3', type=str, default='v1')
    parser.add_argument('--phase4', type=str, default='v1')
    parser.add_argument('--use_boundary', action='store_true')
    parser.add_argument('--use_cldice', action='store_true')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    run_experiment(args)
