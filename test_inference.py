import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging

import configs.BEFUnet_Config as configs
from models.BEFUnet import BEFUnet3D
from models.DataLoader import get_train_val_loaders
from utils import calculate_metric_percase

# -------------------------------
# 🔹 Device setup
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 🔹 Inference & Evaluation
# -------------------------------
@torch.no_grad()
def inference_3d(model, testloader, args, test_save_path=None, visualize=False):
    """
    Unified inference & evaluation with optional visualization.
    """
    model.eval()
    metric_sum = None
    metric_counts = None  # track valid counts per class

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), ncols=70):
        image = sampled_batch["image"].to(DEVICE)   # (1, C, D, H, W)
        label = sampled_batch["label"].to(DEVICE)   # (1, D, H, W)
        case_name = sampled_batch["case_name"][0]

        # Forward pass
        seg_logits, _, _ = model(image)

        # Prediction
        pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)  # (1, D, H, W)

        prediction_np = pred.squeeze(0).cpu().numpy()
        label_np = label.squeeze(0).cpu().numpy()

        
        # ✅ Get per-class metrics as dict
        metrics_dict = calculate_metric_percase(prediction_np, label_np, num_classes=args.num_classes)

        # Accumulate results
        for c in range(1, args.num_classes):
            dice, hd95 = metrics_dict[c]
            if dice is not None:   # valid case
                metric_sum[c] += np.array([dice, hd95])
                metric_counts[c] += 1

        # Per-case logging (mean across valid classes)
        valid_scores = [(d, h) for (d, h) in [metrics_dict[c] for c in range(1, args.num_classes)] if d is not None]
        if valid_scores:
            mean_dice_case = np.mean([d for d, _ in valid_scores])
            mean_hd95_case = np.mean([h for _, h in valid_scores])
            logging.info(' idx %d case %s mean_dice %f mean_hd95 %f',
                         i_batch, case_name, mean_dice_case, mean_hd95_case)

        # Optional save
        if test_save_path is not None:
            pass

    # Compute averages per class
    metric_mean = {}
    for c in range(1, args.num_classes):
        if metric_counts[c] > 0:
            metric_mean[c] = metric_sum[c] / metric_counts[c]
        else:
            metric_mean[c] = (0.0, 0.0)

    # Class names
    class_names = {1: "ET", 2: "TC", 3: "WT"} if args.num_classes == 4 else {
        i: f"class{i}" for i in range(1, args.num_classes)
    }

    for c in range(1, args.num_classes):
        dice, hd95 = metric_mean[c]
        logging.info(
            f"Mean {class_names[c]}: Dice = {dice:.4f}, HD95 = {hd95:.4f}"
        )

    # Overall averages (BraTS style = mean across ET, TC, WT)
    dices = [metric_mean[c][0] for c in range(1, args.num_classes) if metric_counts[c] > 0]
    hd95s = [metric_mean[c][1] for c in range(1, args.num_classes) if metric_counts[c] > 0]

    performance = np.mean(dices) if dices else 0
    mean_hd95 = np.mean(hd95s) if hd95s else 0

    logging.info('Testing performance (best-val model): mean_dice: %f  mean_hd95: %f',
                 performance, mean_hd95)

    return performance, mean_hd95
       
# -------------------------------
# 🔹 Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default='/content/drive/MyDrive/outputs/evaluation', help='root dir for output log')
    parser.add_argument('--model_name', type=str,
                        default='BEFUnet3D')
    parser.add_argument('--root_path', type=str,
                        default='/content/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData',
                        help='root dir for training data')
    parser.add_argument('--model_path', type=str,
                        default='/content/drive/MyDrive/outputs/BEFUnet3D/BEFUnet3D_best.pth')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--visualize', action='store_true', help='save sample visualizations')
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configs & data
    CONFIGS = {'BEFUnet3D': configs.get_BEFUnet_configs()}
    _, test_loader = get_train_val_loaders(args.root_path, batch_size=1)

    # Load model
    model = BEFUnet3D(
        config=CONFIGS['BEFUnet3D'],
        n_classes=args.num_classes).to(DEVICE)
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print("✅ Loaded trained model.")

    # Run evaluation
    inference_3d(model, test_loader, args, test_save_path=args.output_dir, visualize=args.visualize)
