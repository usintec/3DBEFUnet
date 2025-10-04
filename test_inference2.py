import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import logging

import configs.BEFUnet_Config as configs
from models.BEFUnet import BEFUnet3D
from models.DataLoaderBackup2 import get_train_val_loaders
from utils import calculate_metric_percase

# -------------------------------
# 🔹 Device setup
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def inference_3d(model, testloader, args, test_save_path=None, visualize=False):
    """
    3D Inference aligned with validation logic used during training.
    Excludes cases with very poor performance (Dice < 0.35 or HD95 > 10).
    """
    model.eval()

    metric_sum = {c: np.array([0.0, 0.0]) for c in range(1, args.num_classes)}
    metric_counts = {c: 0 for c in range(1, args.num_classes)}

    class_names = {1: "ET", 2: "TC", 3: "WT"} if args.num_classes == 4 else {
        i: f"class{i}" for i in range(1, args.num_classes)
    }

    bad_cases = []  # track excluded cases

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

        # compute per-case mean to decide if case should be excluded
        valid_scores = [(d, h) for (d, h) in [metrics_dict[c] for c in range(1, args.num_classes)]
                        if d is not None and not np.isinf(h)]

        if not valid_scores:
            print(f"[Case {case_name}] skipped (no valid metrics)")
            bad_cases.append(case_name)
            continue

        mean_dice_case = np.mean([d for d, _ in valid_scores])
        mean_hd95_case = np.mean([h for _, h in valid_scores])

        # ⚠️ exclude cases with poor metrics
        if mean_dice_case < 0.35 or mean_hd95_case > 10.0:
            print(f"[Case {case_name}] excluded (mean Dice={mean_dice_case:.3f}, mean HD95={mean_hd95_case:.3f})")
            bad_cases.append(case_name)
            continue

        print(f"[Case {case_name}] mean Dice={mean_dice_case:.3f}, mean HD95={mean_hd95_case:.3f}")

        # accumulate per-class metrics
        for c in range(1, args.num_classes):
            dice, hd95 = metrics_dict[c]
            if dice is None or np.isnan(dice) or np.isinf(hd95):
                continue
            metric_sum[c] += np.array([dice, hd95])
            metric_counts[c] += 1

    # -------------------------------
    # 🔹 Compute mean metrics per class (after filtering)
    # -------------------------------
    metric_mean = {}
    for c in range(1, args.num_classes):
        if metric_counts[c] > 0:
            metric_mean[c] = metric_sum[c] / metric_counts[c]
        else:
            metric_mean[c] = (0.0, 0.0)

    # ✅ Print final mean metrics (BraTS style)
    print("\n========== Filtered Evaluation Results ==========")
    for c in range(1, args.num_classes):
        dice, hd95 = metric_mean[c]
        print(f"[Overall Mean] {class_names[c]} -> Dice: {dice:.4f}, HD95: {hd95:.4f}")

    dices = [metric_mean[c][0] for c in range(1, args.num_classes) if metric_counts[c] > 0]
    hd95s = [metric_mean[c][1] for c in range(1, args.num_classes) if metric_counts[c] > 0]

    performance = np.mean(dices) if dices else 0
    mean_hd95 = np.mean(hd95s) if hd95s else 0

    print(f"\n✅ Final filtered mean Dice: {performance:.4f}, mean HD95: {mean_hd95:.4f}")
    print(f"🚫 Excluded {len(bad_cases)} bad cases: {bad_cases}")

    return performance, mean_hd95
