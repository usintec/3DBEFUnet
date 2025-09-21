import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss  # uses reduction over all voxels; works for 3D too
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from models.DataLoader import get_train_val_loaders
# 🔑 Import BalancedLoss
from models.Losses import BalancedLoss

@torch.no_grad()
@torch.no_grad()
@torch.no_grad()
def inference_3d(model, testloader, args, test_save_path=None, apply_msc=False):
    """
    3D inference with per-class metrics (Dice, HD95).
    Returns:
        performance (float): mean Dice across classes
        mean_hd95 (float): mean HD95 across classes
        metrics (dict): per-class Dice & HD95, plus mean
    """
    model.eval()
    metric_sum = None  # accumulate per-class metrics (dice, hd95), shape: (C-1, 2)

    from models.Clustering import MeanShiftClustering
    msc = MeanShiftClustering(bandwidth=0.5)

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), ncols=70):
        image = sampled_batch["image"].cuda(non_blocking=True)   # (1, C, D, H, W)
        label = sampled_batch["label"].cuda(non_blocking=True)   # (1, D, H, W)
        case_name = sampled_batch['case_name'][0]

        seg_logits, embeddings, _ = model(image)
        if apply_msc:
            seg_logits = msc(embeddings, seg_logits)

        pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)  # (1, D, H, W)

        from utils import calculate_metric_percase
        prediction_np = pred.squeeze(0).cpu().numpy()
        label_np = label.squeeze(0).cpu().numpy()

        metric_i = []
        for c in range(1, args.num_classes):
            metric_i.append(calculate_metric_percase((prediction_np == c).astype(np.uint8),
                                                     (label_np == c).astype(np.uint8)))
        metric_i = np.array(metric_i)  # shape (C-1, 2)

        if metric_sum is None:
            metric_sum = metric_i
        else:
            metric_sum += metric_i

        logging.info(' idx %d case %s mean_dice %f mean_hd95 %f',
                     i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1])

    # Average over dataset
    metric_mean = metric_sum / len(testloader.dataset)

    class_names = {1: "ET", 2: "TC", 3: "WT"} if args.num_classes == 4 else {i: f"class{i}" for i in range(1, args.num_classes)}
    per_class_metrics = {}
    for i in range(1, args.num_classes):
        dice_i, hd95_i = metric_mean[i-1]
        per_class_metrics[class_names[i]] = {"dice": float(dice_i), "hd95": float(hd95_i)}
        logging.info('Mean %s: Dice = %.4f, HD95 = %.4f', class_names[i], dice_i, hd95_i)

    performance = float(np.mean(metric_mean, axis=0)[0])
    mean_hd95 = float(np.mean(metric_mean, axis=0)[1])
    per_class_metrics["mean"] = {"dice": performance, "hd95": mean_hd95}

    logging.info('Testing performance (best-val model): mean_dice: %.4f  mean_hd95: %.4f',
                 performance, mean_hd95)

    # 🔑 Keep backward compatibility
    return performance, mean_hd95, per_class_metrics

def plot_result(train_loss_history, val_dice_history, snapshot_path, args):
    """
    Plots training loss and validation Dice (ET, TC, WT) vs epochs.
    """
    epochs = range(1, len(train_loss_history) + 1)

    # --- Plot training loss ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, 'b-o', label='Training Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epochs")
    plt.legend()

    # --- Plot per-class Dice ---
    plt.subplot(1, 2, 2)
    for k, v in val_dice_history.items():
        plt.plot(epochs[:len(v)], v, '-o', label=f"Dice {k}")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice vs Epochs")
    plt.legend()

    plt.tight_layout()
    out_file = os.path.join(snapshot_path, f"{args.model_name}_training_curves.png")
    plt.savefig(out_file)
    plt.close()
    print(f"✅ Training curves saved at {out_file}")

def plot_result2(dice, h, snapshot_path, args):
    data = {'mean_dice': dice, 'mean_hd95': h}
    df = pd.DataFrame(data)
    resolution_value = 300

    # Dice curve
    ax = df['mean_dice'].plot(title='Mean Dice')
    fig = ax.get_figure()
    fn = f'{args.model_name}_{datetime.datetime.now():%Y%m%d-%H%M%S}_dice.png'
    fig.savefig(os.path.join(snapshot_path, fn), format="png", dpi=resolution_value)
    plt.close(fig)

    # HD95 curve
    ax = df['mean_hd95'].plot(title='Mean HD95')
    fig = ax.get_figure()
    fn = f'{args.model_name}_{datetime.datetime.now():%Y%m%d-%H%M%S}_hd95.png'
    fig.savefig(os.path.join(snapshot_path, fn), format="png", dpi=resolution_value)
    plt.close(fig)

    # CSV
    fn = f'{args.model_name}_{datetime.datetime.now():%Y%m%d-%H%M%S}_results.csv'
    df.to_csv(os.path.join(snapshot_path, fn), sep='\t', index=False)

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

def trainer_3d(args, model, snapshot_path):
    import math
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

    # --- Use BalancedLoss ---
    # Example weights for [BG, ET, TC, WT] → emphasize ET
    class_weights = [1.0, 2.0, 1.0, 1.0]
    loss_fn = BalancedLoss(
        num_classes=args.num_classes,
        ce_weight=0.8,
        dice_weight=0.2,
        class_weights=class_weights
    ).to(device)

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

            for k in val_dice_history.keys():
                val_dice_history[k].append(metrics[k]["dice"])

            min_dice_threshold = 0.4814
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

    plot_result(train_loss_history, val_dice_history, snapshot_path, args)
    writer.close()
    return "Training Finished!"