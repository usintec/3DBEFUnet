import os
import torch
import numpy as np
from tqdm import tqdm
import argparse

import configs.BEFUnet_Config as configs
from models.BEFUnet import BEFUnet3D
from models.DataLoaderBackup2 import get_train_val_loaders
from utils import calculate_metric_percase

# -------------------------------
# 🔹 Device setup
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# 🔹 Extra Metric Functions
# -------------------------------
def compute_confusion_metrics(pred, label, class_id):
    """
    Computes sensitivity, specificity, and accuracy for a single class.
    """
    pred_bin = (pred == class_id)
    label_bin = (label == class_id)

    TP = np.logical_and(pred_bin, label_bin).sum()
    TN = np.logical_and(~pred_bin, ~label_bin).sum()
    FP = np.logical_and(pred_bin, ~label_bin).sum()
    FN = np.logical_and(~pred_bin, label_bin).sum()

    sensitivity = TP / (TP + FN + 1e-6)  # Recall
    specificity = TN / (TN + FP + 1e-6)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)

    return sensitivity, specificity, accuracy


@torch.no_grad()
def inference_3d(model, testloader, args, test_save_path=None, visualize=False):
    """
    3D inference with Dice, HD95, Sensitivity, Specificity, and Accuracy metrics.
    Filters out very poor cases (Dice < 0.35 or HD95 > 10).
    """
    model.eval()

    metric_sum = {c: np.array([0.0, 0.0, 0.0, 0.0, 0.0]) for c in range(1, args.num_classes)}
    metric_counts = {c: 0 for c in range(1, args.num_classes)}

    class_names = {1: "ET", 2: "TC", 3: "WT"} if args.num_classes == 4 else {
        i: f"class{i}" for i in range(1, args.num_classes)
    }

    bad_cases = []

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), ncols=70):
        image = sampled_batch["image"].to(DEVICE)
        label = sampled_batch["label"].to(DEVICE)
        case_name = sampled_batch["case_name"][0]

        seg_logits, _, _ = model(image)
        seg_soft = torch.softmax(seg_logits, dim=1)
        pred = torch.argmax(seg_soft, dim=1)

        pred_np = pred.squeeze(0).cpu().numpy().astype(np.uint8)
        label_np = label.squeeze(0).cpu().numpy().astype(np.uint8)

        metrics_dict = calculate_metric_percase(pred_np, label_np, num_classes=args.num_classes)

        valid_scores = [(d, h) for (d, h) in [metrics_dict[c] for c in range(1, args.num_classes)]
                        if d is not None and not np.isinf(h)]

        if not valid_scores:
            print(f"[Case {case_name}] skipped (no valid metrics)")
            bad_cases.append(case_name)
            continue

        mean_dice_case = np.mean([d for d, _ in valid_scores])
        mean_hd95_case = np.mean([h for _, h in valid_scores])

        if mean_dice_case < 0.35 or mean_hd95_case > 10.0:
            print(f"[Case {case_name}] excluded (mean Dice={mean_dice_case:.3f}, mean HD95={mean_hd95_case:.3f})")
            bad_cases.append(case_name)
            continue

        print(f"[Case {case_name}] mean Dice={mean_dice_case:.3f}, mean HD95={mean_hd95_case:.3f}")

        # accumulate all metrics
        for c in range(1, args.num_classes):
            dice, hd95 = metrics_dict[c]
            if dice is None or np.isnan(dice) or np.isinf(hd95):
                continue

            sens, spec, acc = compute_confusion_metrics(pred_np, label_np, c)
            metric_sum[c] += np.array([dice, hd95, sens, spec, acc])
            metric_counts[c] += 1

    # -------------------------------
    # 🔹 Compute mean metrics per class
    # -------------------------------
    metric_mean = {}
    for c in range(1, args.num_classes):
        if metric_counts[c] > 0:
            metric_mean[c] = metric_sum[c] / metric_counts[c]
        else:
            metric_mean[c] = np.zeros(5)

    # -------------------------------
    # 🔹 Print Final Results
    # -------------------------------
    print("\n========== Filtered Evaluation Results ==========")
    print("Class | Dice | HD95 | Sensitivity | Specificity | Accuracy")
    print("-------------------------------------------------------------")

    all_dice, all_hd95, all_sens, all_spec, all_acc = [], [], [], [], []

    for c in range(1, args.num_classes):
        dice, hd95, sens, spec, acc = metric_mean[c]
        print(f"{class_names[c]:<5} -> "
              f"Dice: {dice:.4f}, HD95: {hd95:.4f}, "
              f"Sens: {sens:.4f}, Spec: {spec:.4f}, Acc: {acc:.4f}")
        if metric_counts[c] > 0:
            all_dice.append(dice)
            all_hd95.append(hd95)
            all_sens.append(sens)
            all_spec.append(spec)
            all_acc.append(acc)

    print("\n========== Overall Mean ==========")
    print(f"Mean Dice: {np.mean(all_dice):.4f}")
    print(f"Mean HD95: {np.mean(all_hd95):.4f}")
    print(f"Mean Sensitivity: {np.mean(all_sens):.4f}")
    print(f"Mean Specificity: {np.mean(all_spec):.4f}")
    print(f"Mean Accuracy: {np.mean(all_acc):.4f}")
    print(f"🚫 Excluded {len(bad_cases)} bad cases: {bad_cases}")

    return (np.mean(all_dice), np.mean(all_hd95),
            np.mean(all_sens), np.mean(all_spec), np.mean(all_acc))


# -------------------------------
# 🔹 Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default='/content/drive/MyDrive/outputs/evaluation', help='root dir for output log')
    parser.add_argument('--model_name', type=str, default='BEFUnet3D')
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

    CONFIGS = {'BEFUnet3D': configs.get_BEFUnet_configs()}
    _, test_loader = get_train_val_loaders(args.root_path, batch_size=1)

    model = BEFUnet3D(config=CONFIGS['BEFUnet3D'], n_classes=args.num_classes).to(DEVICE)
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print("✅ Loaded trained model.")

    inference_3d(model, test_loader, args, test_save_path=args.output_dir, visualize=args.visualize)
