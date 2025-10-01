import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import random

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
    Uses valid_mask logic to ignore empty cases (same as newer version).
    """
    model.eval()
    metric_sum = None
    metric_counts = None

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

        # Dice/HD95 per class
        metric_i = []
        valid_mask = []
        for c in range(1, args.num_classes):  # skip background
            dice_hd = calculate_metric_percase(
            (prediction_np == c).astype(np.uint8),
            (label_np == c).astype(np.uint8)
            )

        # Ensure it's a tuple of floats (or None handled)
        if dice_hd is not None:
            try:
                dice_val, hd95_val = map(float, dice_hd)
                if not (np.isnan(dice_val) or np.isnan(hd95_val)):
                    metric_i.append((dice_val, hd95_val))
                    valid_mask.append(True)
                else:
                    metric_i.append((0.0, 0.0))
                    valid_mask.append(False)
            except Exception:
                metric_i.append((0.0, 0.0))
                valid_mask.append(False)
        else:
            metric_i.append((0.0, 0.0))
            valid_mask.append(False)


        metric_i = np.array(metric_i, dtype=np.float32)

        if metric_sum is None:
            metric_sum = np.zeros_like(metric_i, dtype=float)  # (num_classes-1, 2)
            metric_counts = np.zeros((args.num_classes - 1,), dtype=int)

        # accumulate only valid metrics
        for j, valid in enumerate(valid_mask):
            if valid:
                metric_sum[j] += metric_i[j]
                metric_counts[j] += 1

        # Logging per case (average over valid classes only)
        valid_scores = [metric_i[j] for j, v in enumerate(valid_mask) if v]
        if valid_scores:
            mean_dice_case = np.mean([d for d, _ in valid_scores])
            mean_hd95_case = np.mean([h for _, h in valid_scores])
            print(f"[{i_batch}] {case_name}: Dice={mean_dice_case:.4f}, HD95={mean_hd95_case:.4f}")
        else:
            print(f"[{i_batch}] {case_name}: skipped (no valid classes)")

        # Visualization (optional, single slice per case)
        if visualize:
            slice_idx = random.randint(0, prediction_np.shape[0] - 1)  # random slice
            modality_names = ["T1", "T1ce", "T2", "FLAIR"]
            img_slices = [image[0, m, slice_idx].cpu().numpy() for m in range(image.shape[1])]
            pred_slice = prediction_np[slice_idx]
            label_slice = label_np[slice_idx]

            plt.figure(figsize=(18, 6))
            for i, (mod_name, img_slice) in enumerate(zip(modality_names, img_slices)):
                plt.subplot(2, 3, i+1)
                plt.imshow(img_slice, cmap="gray")
                plt.title(mod_name)
                plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(label_slice, cmap="viridis")
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(pred_slice, cmap="viridis")
            plt.title("Prediction")
            plt.axis("off")

            plt.suptitle(f"Case: {case_name}, Slice {slice_idx}")
            plt.tight_layout()

            os.makedirs(test_save_path, exist_ok=True)
            out_file = os.path.join(test_save_path, f"{case_name}_slice{slice_idx}_result.png")
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"🖼 Saved visualization to {out_file}")

    # ---- Final aggregation ----
    metric_mean = []
    for j in range(args.num_classes - 1):
        if metric_counts[j] > 0:
            metric_mean.append(metric_sum[j] / metric_counts[j])
        else:
            metric_mean.append((0.0, 0.0))  # no valid cases for this class
    metric_mean = np.array(metric_mean)

    class_names = {1: "ET", 2: "TC", 3: "WT"} if args.num_classes == 4 else {
        i: f"class{i}" for i in range(1, args.num_classes)
    }

    print("\n📊 Final Evaluation Results")
    for i in range(1, args.num_classes):
        dice_i, hd95_i = metric_mean[i-1]
        print(f"{class_names[i]}: Dice={dice_i:.4f}, HD95={hd95_i:.4f}")

    # mean over valid classes only
    performance = np.mean([m[0] for m in metric_mean if m[0] > 0]) if any(m[0] > 0 for m in metric_mean) else 0
    mean_hd95 = np.mean([m[1] for m in metric_mean if m[1] > 0]) if any(m[1] > 0 for m in metric_mean) else 0
    print(f"Mean Dice: {performance:.4f}, Mean HD95: {mean_hd95:.4f}")

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
