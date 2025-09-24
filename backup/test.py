import os
import sys
import logging
import torch
import numpy as np
from tqdm import tqdm
import csv

# Project-specific imports (update to your repo structure)
from utils import calculate_metric_percase
from models.Clustering import MeanShiftClustering
from models.BEFUnet import BEFUnet3D  # <-- adjust if needed


import configs.BEFUnet_Config as configs

CONFIGS = {
    'BEFUnet3D': configs.get_BEFUnet_configs(),
}

def load_model_for_eval(model_class, checkpoint_path, device, args):
    """
    Load a trained model checkpoint for evaluation.
    """
    model = model_class(
        config=CONFIGS['BEFUnet3D'],
        img_size=(args.img_size, args.img_size, args.img_size),
        in_chans=4,                  # BraTS has 4 modalities
        n_classes=args.num_classes   # use CLI arg
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model



# Map class indices -> BraTS labels
CLASS_LABELS = {1: "TC", 2: "ET", 3: "WT"}

def validate_model(model, val_loader, args, apply_msc=False):
    """
    Run validation on a dataset and return metrics.
    """
    metric_sum = None
    msc = MeanShiftClustering(bandwidth=0.5) if apply_msc else None

    for i_batch, sampled_batch in tqdm(enumerate(val_loader), total=len(val_loader), ncols=70):
        image = sampled_batch["image"].to(next(model.parameters()).device)
        label = sampled_batch["label"].to(image.device)
        case_name = sampled_batch["case_name"][0]

        with torch.no_grad():
            seg_logits, embeddings, _ = model(image)
            if apply_msc:
                seg_logits = msc(embeddings, seg_logits)
            pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)

        prediction_np = pred.squeeze(0).cpu().numpy()
        label_np = label.squeeze(0).cpu().numpy()

        # Per-class metrics
        metric_i = []
        for c in range(1, args.num_classes):
            metric_i.append(calculate_metric_percase(
                (prediction_np == c).astype(np.uint8),
                (label_np == c).astype(np.uint8)
            ))
        metric_i = np.array(metric_i)

        metric_sum = metric_i if metric_sum is None else metric_sum + metric_i

        logging.info(
            " idx %d case %s mean_dice %.4f mean_hd95 %.4f",
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]
        )

    # Average across dataset (per class)
    metric_mean = metric_sum / len(val_loader.dataset)
    per_class_dice = metric_mean[:, 0]
    per_class_hd95 = metric_mean[:, 1]

    # Overall mean
    performance = np.mean(per_class_dice)
    mean_hd95 = np.mean(per_class_hd95)

    logging.info("Final Report: mean_dice %.4f | mean_hd95 %.4f", performance, mean_hd95)
    for i, (d, h) in enumerate(zip(per_class_dice, per_class_hd95), start=1):
        label = CLASS_LABELS.get(i, f"class_{i}")
        logging.info("Mean class %s mean_dice %.6f mean_hd95 %.6f", label, d, h)

    return performance, mean_hd95, per_class_dice, per_class_hd95


def run_evaluations(args):
    from models.DataLoader import get_train_val_loaders
    # val_loader = get_train_val_loaders(args.root_path, batch_size=args.batch_size)
    train_loader, val_loader = get_train_val_loaders(
    args.root_path, batch_size=args.batch_size
)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    results = []

    if os.path.isfile(args.checkpoint):
        ckpts = [args.checkpoint]
    elif os.path.isdir(args.checkpoint):
        ckpts = [os.path.join(args.checkpoint, f)
                 for f in os.listdir(args.checkpoint) if f.endswith(".pth")]
        ckpts.sort()
    else:
        raise FileNotFoundError(f"Invalid checkpoint path: {args.checkpoint}")

    for ckpt in ckpts:
        logging.info("=" * 50)
        logging.info("Evaluating checkpoint: %s", ckpt)

        model = load_model_for_eval(BEFUnet3D, ckpt, device, args)
        mean_dice, mean_hd95, per_class_dice, per_class_hd95 = validate_model(
            model, val_loader, args, apply_msc=args.apply_msc
        )

        results.append((os.path.basename(ckpt), mean_dice, mean_hd95, per_class_dice, per_class_hd95))

    # =========================
    # TABLE 1: MODEL SUMMARY
    # =========================
    logging.info("\n" + "=" * 50)
    logging.info("SUMMARY OF MODELS (Overall + Per-Class)")
    logging.info("=" * 50)

    header = f"{'Checkpoint':40s} | {'Mean Dice':>9s} | {'Mean HD95':>9s}"
    for i in range(1, args.num_classes):
        label = CLASS_LABELS.get(i, f"class_{i}")
        header += f" | Dice_{label} | HD95_{label}"
    print(header)
    print("-" * (len(header) + 5))

    for ckpt_name, dice, hd95, per_class_dice, per_class_hd95 in results:
        row = f"{ckpt_name:40s} | {dice:9.4f} | {hd95:9.4f}"
        for i, (d, h) in enumerate(zip(per_class_dice, per_class_hd95), start=1):
            label = CLASS_LABELS.get(i, f"class_{i}")
            row += f" | {d:9.4f} | {h:9.4f}"
        print(row)

    print("-" * (len(header) + 5))

    # =========================
    # TABLE 2: CLASS METRICS SUMMARY
    # =========================
    logging.info("\n" + "=" * 50)
    logging.info("SUMMARY OF PER-CLASS METRICS")
    logging.info("=" * 50)

    for class_idx in range(1, args.num_classes):
        label = CLASS_LABELS.get(class_idx, f"class_{class_idx}")
        print(f"\nClass {label} metrics:")
        print(f"{'Checkpoint':40s} | {'Dice':>9s} | {'HD95':>9s}")
        print("-" * 65)
        for ckpt_name, _, _, per_class_dice, per_class_hd95 in results:
            print(f"{ckpt_name:40s} | {per_class_dice[class_idx-1]:9.4f} | {per_class_hd95[class_idx-1]:9.4f}")
        print("-" * 65)

    # =========================
    # SAVE TO CSV
    # =========================
    csv_file = "evaluation_summary.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        # header row
        header_row = ["Checkpoint", "MeanDice", "MeanHD95"]
        for i in range(1, args.num_classes):
            label = CLASS_LABELS.get(i, f"class_{i}")
            header_row += [f"Dice_{label}", f"HD95_{label}"]
        writer.writerow(header_row)

        # data rows
        for ckpt_name, dice, hd95, per_class_dice, per_class_hd95 in results:
            row = [ckpt_name, dice, hd95]
            for d, h in zip(per_class_dice, per_class_hd95):
                row += [d, h]
            writer.writerow(row)

    logging.info(f"Saved evaluation summary to {csv_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default='/content/drive/MyDrive/outputs/BEFUnet3D',
                        help="Path to a single checkpoint .pth or directory of checkpoints")
    parser.add_argument("--root_path", type=str,
                        default='/content/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData',
                        help="Dataset root path")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Output channel of network (BraTS: 4 classes)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for validation")
    parser.add_argument("--apply_msc", action="store_true",
                        help="Apply MeanShift clustering on embeddings")
    parser.add_argument("--img_size", type=int, default=96,
                        help="Input patch size for BEFUnet3D (e.g., 96 for 96x96x96)")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    run_evaluations(args)
