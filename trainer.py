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


def sliding_window_inference(model, volume, patch_size=(96,96,96), stride=(48,48,48), num_classes=4, device="cuda"):
    """
    Perform sliding-window inference on a 3D volume.
    
    Args:
        model: trained segmentation model
        volume: input modalities tensor [C, D, H, W]
        patch_size: size of sliding window
        stride: step size between windows
        num_classes: number of segmentation classes
        device: device for inference
    
    Returns:
        seg_logits: reconstructed logits [num_classes, D, H, W]
    """
    model.eval()
    C, D, H, W = volume.shape
    seg_logits = torch.zeros((num_classes, D, H, W), dtype=torch.float32, device=device)
    norm_map = torch.zeros((1, D, H, W), dtype=torch.float32, device=device)

    # Pad volume if needed
    pad_d = max(0, patch_size[0] - D % stride[0]) if D < patch_size[0] else 0
    pad_h = max(0, patch_size[1] - H % stride[1]) if H < patch_size[1] else 0
    pad_w = max(0, patch_size[2] - W % stride[2]) if W < patch_size[2] else 0
    volume = F.pad(volume.unsqueeze(0), (0,pad_w,0,pad_h,0,pad_d), mode="constant", value=0).squeeze(0)

    _, Dp, Hp, Wp = volume.shape

    # Slide patches
    for z in range(0, Dp - patch_size[0] + 1, stride[0]):
        for y in range(0, Hp - patch_size[1] + 1, stride[1]):
            for x in range(0, Wp - patch_size[2] + 1, stride[2]):
                patch = volume[:, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
                patch = patch.unsqueeze(0).to(device)  # [1,C,D,H,W]

                with torch.no_grad():
                    logits_patch, _, _ = model(patch)  # [1,num_classes,D,H,W]
                    logits_patch = logits_patch.squeeze(0)

                seg_logits[:, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += logits_patch
                norm_map[:, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += 1

    seg_logits = seg_logits / norm_map
    return seg_logits[:, :D, :H, :W]  # remove padding

@torch.no_grad()
def inference_3d_full(model, testloader, args, patch_size=(96,96,96), stride=(48,48,48)):
    """
    Full-volume inference using sliding-window stitching.
    Computes Dice, HD95, Sens, Spec, Prec, Recall, F1 per class and macro avg.
    """
    model.eval()
    metric_sum = None

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), ncols=70):
        image = sampled_batch["image"].squeeze(0)  # (C, D, H, W)
        label = sampled_batch["label"].squeeze(0).numpy()
        case_name = sampled_batch['case_name'][0]

        seg_logits = sliding_window_inference(
            model, image, patch_size=patch_size, stride=stride,
            num_classes=args.num_classes, device="cuda"
        )
        pred = torch.argmax(torch.softmax(seg_logits, dim=0), dim=0).cpu().numpy()

        # Compute metrics per class
        metric_i = []
        for c in range(1, args.num_classes):  # skip background
            
            # Dice + HD95
            dice_c, hd95_c = calculate_metric_percase(
                (pred == c).astype(np.uint8),
                (label == c).astype(np.uint8)
            )

            # Confusion stats
            tp, fp, tn, fn = calculate_confusion_stats(pred, label)

            sens = safe_div(tp, tp + fn)       # Sensitivity = Recall
            spec = safe_div(tn, tn + fp)       # Specificity
            prec = safe_div(tp, tp + fp)       # Precision
            rec  = sens                        # Recall (same as Sensitivity)
            f1   = safe_div(2 * prec * rec, prec + rec)

            metric_i.append((dice_c, hd95_c, sens, spec, prec, rec, f1))
        metric_i = np.array(metric_i)

        if metric_sum is None:
            metric_sum = metric_i
        else:
            metric_sum += metric_i

        hd95_val = np.nanmean(metric_i[:,1]) if not np.all(np.isnan(metric_i[:,1])) else 0.0
        logging.info(
            f"Case {case_name}: mean Dice = {np.mean(metric_i[:,0]):.4f}, "
            f"mean HD95 = {hd95_val:.4f}"
        )

    # Average across cases
    metric_mean = metric_sum / len(testloader.dataset)

    return metric_mean

def calculate_confusion_stats(pred, gt):
    """
    Compute TP, FP, TN, FN for binary masks.
    pred, gt: numpy arrays (0/1)
    """
    tp = np.logical_and(pred == 1, gt == 1).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    return tp, fp, tn, fn

def safe_div(n, d):
    return n / d if d > 0 else 0.0

@torch.no_grad()
def inference_3d(model, testloader, args, test_save_path=None, apply_msc=False):
    """
    3D inference with extended metrics: Dice, HD95, Sensitivity, Specificity,
    Precision, Recall, F1 per class + macro averages.
    """
    model.eval()
    metric_sum = None  # accumulate per-class (dice, hd95, sens, spec, prec, rec, f1), shape: (C-1, 7)

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
            pred_c = (prediction_np == c).astype(np.uint8)
            gt_c = (label_np == c).astype(np.uint8)

            # Dice + HD95
            dice_c, hd95_c = calculate_metric_percase(pred_c, gt_c)

            # Confusion stats
            tp, fp, tn, fn = calculate_confusion_stats(pred_c, gt_c)

            sens = safe_div(tp, tp + fn)       # Sensitivity = Recall
            spec = safe_div(tn, tn + fp)       # Specificity
            prec = safe_div(tp, tp + fp)       # Precision
            rec  = sens                        # Recall (same as Sensitivity)
            f1   = safe_div(2 * prec * rec, prec + rec)

            metric_i.append((dice_c, hd95_c, sens, spec, prec, rec, f1))

        metric_i = np.array(metric_i)  # shape (C-1, 7)

        if metric_sum is None:
            metric_sum = metric_i
        else:
            metric_sum += metric_i

        logging.info(
            f"Case {case_name}: mean Dice = {np.mean(metric_i[:,0]):.4f}, "
            f"mean HD95 = {np.nanmean(metric_i[:,1]):.4f}, "
            f"mean Sens = {np.mean(metric_i[:,2]):.4f}, "
            f"mean Spec = {np.mean(metric_i[:,3]):.4f}"
        )

    # Average over dataset (macro average across cases)
    metric_mean = metric_sum / len(testloader.dataset)

    class_names = {1: "ET", 2: "TC", 3: "WT"} if args.num_classes == 4 else {
        i: f"class{i}" for i in range(1, args.num_classes)
    }

    per_class_metrics = {}
    for i in range(1, args.num_classes):
        dice_i, hd95_i, sens_i, spec_i, prec_i, rec_i, f1_i = metric_mean[i-1]
        per_class_metrics[class_names[i]] = {
            "dice": float(dice_i),
            "hd95": float(hd95_i),
            "sensitivity": float(sens_i),
            "specificity": float(spec_i),
            "precision": float(prec_i),
            "recall": float(rec_i),
            "f1": float(f1_i)
        }
        logging.info(
            f"Mean {class_names[i]}: Dice = {dice_i:.4f}, HD95 = {hd95_i:.4f}, "
            f"Sens = {sens_i:.4f}, Spec = {spec_i:.4f}, Prec = {prec_i:.4f}, F1 = {f1_i:.4f}"
        )

    # Macro averages
    performance = float(np.mean(metric_mean[:, 0]))   # mean Dice
    mean_hd95   = float(np.mean(metric_mean[:, 1]))
    mean_sens   = float(np.mean(metric_mean[:, 2]))
    mean_spec   = float(np.mean(metric_mean[:, 3]))
    mean_prec   = float(np.mean(metric_mean[:, 4]))
    mean_rec    = float(np.mean(metric_mean[:, 5]))
    mean_f1     = float(np.mean(metric_mean[:, 6]))

    per_class_metrics["mean"] = {
        "dice": performance,
        "hd95": mean_hd95,
        "sensitivity": mean_sens,
        "specificity": mean_spec,
        "precision": mean_prec,
        "recall": mean_rec,
        "f1": mean_f1
    }

    logging.info(
        f"Testing performance (best-val model): mean_dice = {performance:.4f}, "
        f"mean_hd95 = {mean_hd95:.4f}, mean_sens = {mean_sens:.4f}, "
        f"mean_spec = {mean_spec:.4f}, mean_prec = {mean_prec:.4f}, mean_f1 = {mean_f1:.4f}"
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
    """
    Load latest checkpoint if available.
    Returns: model, optimizer, scaler, start_epoch, iter_num, mode
    mode ∈ {"scratch", "resume", "finetune"}
    """
    if not os.path.exists(snapshot_path):
        print(f"⚠️ Snapshot path {snapshot_path} does not exist. Starting from scratch.")
        return model, optimizer, scaler, 0, 0, "scratch"

    checkpoints = sorted(
        [f for f in os.listdir(snapshot_path)
         if f.endswith(".pth") and not f.endswith("_pidinet_best.pth")],
        key=lambda x: os.path.getmtime(os.path.join(snapshot_path, x)),
    )

    checkpoint = None
    latest_ckpt = None
    mode = "scratch"

    if checkpoints:
        latest_ckpt = os.path.join(snapshot_path, checkpoints[-1])
        print(f"🔄 Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        mode = "resume"
    else:
        best_ckpt = os.path.join(snapshot_path, "BEFUnet3D_best.pth")
        if os.path.exists(best_ckpt):
            print(f"✨ Fine-tuning from best model only: {best_ckpt}")
            checkpoint = torch.load(best_ckpt, map_location=device)
            mode = "finetune"
        else:
            print("⚠️ No checkpoints found. Starting from scratch.")
            return model, optimizer, scaler, 0, 0, "scratch"

    # Restore model state
    model.load_state_dict(checkpoint["model_state"])

    # Resume: restore optimizer/scaler
    start_epoch, iter_num = 0, 0
    if mode == "resume":
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint and checkpoint["scaler_state"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        iter_num = checkpoint.get("iter_num", 0)

    return model, optimizer, scaler, start_epoch, iter_num, mode

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
    # alpha = [0.1, 0.3, 0.3, 0.3],
    loss_fn = DiceFocalLoss(dice_weight=0.6, focal_weight=0.4, class_weights = [1.0, 2.0, 2.0, 4.0], num_classes=4).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6
    )

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # Load checkpoint (resume / finetune / scratch)
    model, optimizer, scaler, start_epoch, iter_num, load_mode = load_checkpoint(
        model, optimizer, scaler, snapshot_path, device
    )

    if load_mode == "finetune":
        # Reset counters for a fresh training run
        start_epoch, iter_num = 0, 0

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
            # mean_dice, mean_hd95, metrics = inference_3d(model, val_loader, args, test_save_path=test_save_path)
            mean_dice, mean_hd95, metrics = inference_3d_full(model, val_loader, args, test_save_path=test_save_path)

            # --- Save Dice & HD95 history ---
            for k in ["ET", "TC", "WT"]:
                val_dice_history[k].append(metrics[k]["dice"])
                val_hd95_history[k].append(metrics[k]["hd95"])
            val_dice_history["mean"].append(mean_dice)
            val_hd95_history["mean"].append(mean_hd95)

            # --- Check improvement ---
            min_dice_threshold = 0.0000
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

        # scheduler.step(mean_dice)
        scheduler.step()
        writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], epoch_num)

        model.train()

    # ✅ Updated call
    plot_result(train_loss_history, val_dice_history, val_hd95_history, snapshot_path, args)
    writer.close()
    return "Training Finished!"
