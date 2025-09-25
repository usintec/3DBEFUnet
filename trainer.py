import logging
import os
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import DiceLoss, calculate_metric_percase  # uses reduction over all voxels; works for 3D too
from models.DataLoader import get_train_val_loaders
from models.Losses import BalancedLoss, DiceFocalLoss
from torch.optim.lr_scheduler import LambdaLR

@torch.no_grad()
def inference_3d(model, testloader, args, test_save_path=None, apply_msc=False):
    """
    3D inference with per-class metrics (Dice, HD95).
    Returns:
        performance (float): mean Dice across tumor classes (macro-avg)
        mean_hd95 (float): mean HD95 across tumor classes (macro-avg)
        metrics (dict): per-class Dice & HD95, plus macro averages
    """
    model.eval()
    metric_sum = None  # accumulate per-class metrics (dice, hd95), shape: (C-1, 2)

    from models.Clustering import MeanShiftClustering
    msc = MeanShiftClustering(bandwidth=0.5)

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), ncols=70):
        image = sampled_batch["image"].cuda(non_blocking=True)   # (1, C, D, H, W)
        label = sampled_batch["label"].cuda(non_blocking=True)   # (1, D, H, W)
        case_name = sampled_batch['case_name'][0]

        with torch.no_grad():
            seg_logits, embeddings, _ = model(image)
            if apply_msc:
                seg_logits = msc(embeddings, seg_logits)

        pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)  # (1, D, H, W)

        prediction_np = pred.squeeze(0).cpu().numpy()
        label_np = label.squeeze(0).cpu().numpy()

        # Compute per-class metrics
        metric_i = []
        for c in range(1, args.num_classes):  # skip background
            dice_c, hd95_c = calculate_metric_percase(
                (prediction_np == c).astype(np.uint8),
                (label_np == c).astype(np.uint8)
            )
            metric_i.append((dice_c, hd95_c))
        metric_i = np.array(metric_i)  # shape (C-1, 2)

        if metric_sum is None:
            metric_sum = metric_i
        else:
            metric_sum += metric_i

        logging.info(
            f"Case {case_name}: mean Dice = {np.mean(metric_i[:,0]):.4f}, "
            f"mean HD95 = {np.mean(metric_i[:,1]):.4f}"
        )

    # Average over dataset (macro average across cases)
    metric_mean = metric_sum / len(testloader.dataset)

    class_names = {1: "ET", 2: "TC", 3: "WT"} if args.num_classes == 4 else {
        i: f"class{i}" for i in range(1, args.num_classes)
    }

    per_class_metrics = {}
    for i in range(1, args.num_classes):
        dice_i, hd95_i = metric_mean[i-1]
        per_class_metrics[class_names[i]] = {
            "dice": float(dice_i),
            "hd95": float(hd95_i)
        }
        logging.info(
            f"Mean {class_names[i]}: Dice = {dice_i:.4f}, HD95 = {hd95_i:.4f}"
        )

    # Macro averages
    performance = float(np.mean(metric_mean[:, 0]))
    mean_hd95 = float(np.mean(metric_mean[:, 1]))
    per_class_metrics["mean"] = {"dice": performance, "hd95": mean_hd95}

    logging.info(
        f"Testing performance (best-val model): mean_dice = {performance:.4f}, "
        f"mean_hd95 = {mean_hd95:.4f}"
    )

    return performance, mean_hd95, per_class_metrics

def plot_result(train_loss_history, val_dice_history, val_hd95_history, snapshot_path, args):
    """
    Plots:
      1. Training loss vs epochs (separate file)
      2. Per-class Dice (ET, TC, WT, mean) vs epochs (separate file)
      3. Per-class HD95 (ET, TC, WT, mean) vs epochs (separate file)
      4. Combined figure with all three (loss, Dice, HD95)
    """
    epochs = range(1, len(train_loss_history) + 1)

    # --- Plot training loss ---
    plt.figure(figsize=(6, 5))
    plt.plot(epochs, train_loss_history, 'b-o', label='Training Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epochs")
    plt.legend()
    out_file = os.path.join(snapshot_path, f"{args.model_name}_loss_curve.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"✅ Loss curve saved at {out_file}")

    # --- Plot per-class Dice ---
    plt.figure(figsize=(6, 5))
    for k, v in val_dice_history.items():
        plt.plot(epochs[:len(v)], v, '-o', label=f"Dice {k}")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice vs Epochs")
    plt.legend()
    out_file = os.path.join(snapshot_path, f"{args.model_name}_dice_curve.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"✅ Dice curve saved at {out_file}")

    # --- Plot per-class HD95 ---
    plt.figure(figsize=(6, 5))
    for k, v in val_hd95_history.items():
        plt.plot(epochs[:len(v)], v, '-o', label=f"HD95 {k}")
    plt.xlabel("Epochs")
    plt.ylabel("HD95 (mm)")
    plt.title("Validation HD95 vs Epochs")
    plt.legend()
    out_file = os.path.join(snapshot_path, f"{args.model_name}_hd95_curve.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"✅ HD95 curve saved at {out_file}")

    # --- Combined Dashboard ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: Loss
    axs[0].plot(epochs, train_loss_history, 'b-o', label="Training Loss")
    axs[0].set_title("Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Subplot 2: Dice
    for k, v in val_dice_history.items():
        axs[1].plot(epochs[:len(v)], v, '-o', label=f"Dice {k}")
    axs[1].set_title("Validation Dice")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Dice Score")
    axs[1].legend()

    # Subplot 3: HD95
    for k, v in val_hd95_history.items():
        axs[2].plot(epochs[:len(v)], v, '-o', label=f"HD95 {k}")
    axs[2].set_title("Validation HD95")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("HD95 (mm)")
    axs[2].legend()

    plt.tight_layout()
    out_file = os.path.join(snapshot_path, f"{args.model_name}_training_dashboard.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"✅ Combined dashboard saved at {out_file}")

def save_pidinet3d(pidinet, save_path, filename, epoch=None):
    """
    Save PiDiNet3D weights only.
    """
    state_dict = pidinet.state_dict()
    checkpoint = {
        "epoch": epoch,
        "state_dict": state_dict,
    }
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)
    torch.save(checkpoint, filepath)
    print(f"✅ Saved PiDiNet3D checkpoint to {save_path}")

def save_checkpoint(state, snapshot_path, filename):
    os.makedirs(snapshot_path, exist_ok=True)
    filepath = os.path.join(snapshot_path, filename)
    torch.save(state, filepath)

def load_checkpoint(model, optimizer, scaler, snapshot_path, device):
    # List all .pth files in snapshot path
    checkpoints = sorted(
        [f for f in os.listdir(snapshot_path) if f.endswith(".pth")],
        key=lambda x: os.path.getmtime(os.path.join(snapshot_path, x)),
    )

    if checkpoints:
        # Resume from the most recent checkpoint
        latest_ckpt = os.path.join(snapshot_path, checkpoints[-1])
        print(f"🔄 Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)

    else:
        # Fallback: try loading best model
        best_ckpt = os.path.join(snapshot_path, "BEFUnet3D_best.pth")
        if os.path.exists(best_ckpt):
            print(f"✨ No checkpoints found, loading best model: {best_ckpt}")
            checkpoint = torch.load(best_ckpt, map_location=device)
        else:
            print("⚠️ No checkpoints or best model found. Starting from scratch.")
            return model, optimizer, scaler, 0, 0

    # Restore model state
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "scaler_state" in checkpoint and checkpoint["scaler_state"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    iter_num = checkpoint.get("iter_num", 0)

    return model, optimizer, scaler, start_epoch, iter_num

# --- Scheduler helper ---
def lr_lambda(epoch, warmup_epochs=5, max_epochs=300, base_lr=0.001, min_lr=1e-6, power=0.9):
    """Linear warmup + polynomial decay"""
    if epoch < warmup_epochs:
        # Warmup: scale from 1e-4 → base_lr over warmup_epochs
        warmup_start = 1e-4
        return (warmup_start + (base_lr - warmup_start) * (epoch + 1) / warmup_epochs) / base_lr
    else:
        # PolyLR decay after warmup
        decay_epoch = epoch - warmup_epochs
        decay_max = max_epochs - warmup_epochs
        poly_factor = (1 - decay_epoch / decay_max) ** power
        return max(min_lr / base_lr, poly_factor)

def trainer_3d(args, model, snapshot_path):
    date_and_time = datetime.datetime.now()

    os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
    test_save_path = os.path.join(snapshot_path, 'test')

    logging.basicConfig(
        filename=os.path.join(snapshot_path, f"{args.model_name}_{date_and_time:%Y%m%d-%H%M%S}_log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    batch_size = args.batch_size * args.n_gpu
    train_loader, val_loader = get_train_val_loaders(args.root_path, batch_size=batch_size)

    if getattr(args, "max_iterations", None) and args.max_iterations > 0:
        iters_per_epoch = len(train_loader)
        args.max_epochs = math.ceil(args.max_iterations / iters_per_epoch)
        logging.info(
            "Adjusted max_epochs = %d (from max_iterations = %d, iters/epoch = %d)",
            args.max_epochs, args.max_iterations, iters_per_epoch
        )

    model = model.to(device)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # --- Use DiceFocalLoss ---
    loss_fn = DiceFocalLoss(dice_weight=0.5, focal_weight=0.5,class_weights = [1.0, 2.0, 2.0, 3.0], alpha = [0.1, 0.3, 0.3, 0.3], num_classes=4).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    model, optimizer, scaler, start_epoch, iter_num = load_checkpoint(
        model, optimizer, scaler, snapshot_path, device
    )

    max_epoch = args.max_epochs
    logging.info("%d iterations per epoch", len(train_loader))

    best_performance = 0.0
    patience = getattr(args, "patience", 20)
    counter = 0

    train_loss_history = []
    val_dice_history = {"ET": [], "TC": [], "WT": [], "mean": []}
    val_hd95_history = {"ET": [], "TC": [], "WT": [], "mean": []}  # NEW

    for epoch_num in range(start_epoch, max_epoch):
        model.train()
        epoch_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch [{epoch_num+1}/{max_epoch}]", unit="batch") as pbar:
            for i_batch, sampled_batch in enumerate(pbar):
                image_batch = sampled_batch['image'].to(device, non_blocking=True)
                label_batch = sampled_batch['label'].to(device, non_blocking=True)

                try:
                    with torch.amp.autocast("cuda", enabled=True):
                        seg_logits, _, _ = model(image_batch)
                        loss = loss_fn(seg_logits, label_batch)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    iter_num += 1
                    epoch_loss += loss.item()

                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    writer.add_scalar('info/total_loss', loss.item(), iter_num)

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        logging.warning("OOM at iter %d, skipping", iter_num)
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_epoch_loss)
        logging.info(f"Epoch {epoch_num+1}/{max_epoch} finished, Avg Loss = {avg_epoch_loss:.4f}")

        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            mean_dice, mean_hd95, metrics = inference_3d(model, val_loader, args, test_save_path=test_save_path)

            # --- Save Dice & HD95 history ---
            for k in ["ET", "TC", "WT"]:
                val_dice_history[k].append(metrics[k]["dice"])
                val_hd95_history[k].append(metrics[k]["hd95"])
            val_dice_history["mean"].append(mean_dice)
            val_hd95_history["mean"].append(mean_hd95)

            # --- Check improvement ---
            min_dice_threshold = 0.4874
            improved = False
            if mean_dice >= min_dice_threshold and mean_dice >= best_performance:
                best_performance = mean_dice
                counter = 0
                improved = True
                tqdm.write(f"🌟 New best Dice = {mean_dice:.4f} (threshold {min_dice_threshold}) at epoch {epoch_num}")
                save_checkpoint(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "epoch": epoch_num,
                        "iter_num": iter_num,
                    },
                    snapshot_path,
                    f"{args.model_name}_best.pth",
                )
                save_pidinet3d(model.pidinet, snapshot_path, f"{args.model_name}_pidinet_best.pth")

            if not improved:
                counter += 1
                tqdm.write(f"⚠️ No improvement: Dice = {mean_dice:.4f}, best = {best_performance:.4f}, "
                           f"threshold = {min_dice_threshold}, patience counter = {counter}/{patience}")

            if counter >= patience:
                tqdm.write(f"⏹ Early stopping triggered at epoch {epoch_num}")
                break

            scheduler.step(mean_dice)
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], epoch_num)

            model.train()

    # ✅ Updated call
    plot_result(train_loss_history, val_dice_history, val_hd95_history, snapshot_path, args)
    writer.close()
    return "Training Finished!"
