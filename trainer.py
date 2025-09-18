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

@torch.no_grad()
@torch.no_grad()
def inference_3d(model, testloader, args, test_save_path=None, apply_msc=False):
    """
    3D inference over whole volumes. Expects batch size == 1 from testloader.
    testloader must yield dicts with keys: 
      'image': (1,C,D,H,W), 
      'label': (1,D,H,W), 
      'case_name': [str]

    Args:
        model: trained BEFUnet model
        testloader: dataloader for test set
        args: arguments with num_classes, etc.
        test_save_path: optional directory for saving predictions
        apply_msc: bool, if True → refine segmentation with Mean Shift Clustering (MSC)
    """
    model.eval()
    metric_sum = None  # accumulate per-class metrics (dice, hd95), shape: (C-1, 2)

    # Import MSC module
    from models.Clustering import MeanShiftClustering
    msc = MeanShiftClustering(bandwidth=0.5)

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), ncols=70):
        image = sampled_batch["image"].cuda(non_blocking=True)   # (1, C, D, H, W)
        label = sampled_batch["label"].cuda(non_blocking=True)   # (1, D, H, W)
        case_name = sampled_batch['case_name'][0]

        # Forward pass → model now returns seg_logits, embeddings, dlf_loss
        seg_logits, embeddings, _ = model(image)

        # Optionally refine with MSC
        if apply_msc:
            seg_logits = msc(embeddings, seg_logits)

        # Prediction
        pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)  # (1, D, H, W)

        # Compute per-case metrics with your utility (expects numpy)
        from utils import calculate_metric_percase  # must be 3D-aware
        prediction_np = pred.squeeze(0).cpu().numpy()
        label_np = label.squeeze(0).cpu().numpy()

        # Accumulate metrics per class (1..num_classes-1)
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

        # Optional save
        if test_save_path is not None:
            # Here you could add NIfTI saving (e.g., using nibabel) if required
            pass

    # Average over cases
    metric_mean = metric_sum / len(testloader.dataset)

    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f',
                     i, metric_mean[i-1][0], metric_mean[i-1][1])

    performance = np.mean(metric_mean, axis=0)[0]
    mean_hd95 = np.mean(metric_mean, axis=0)[1]
    logging.info('Testing performance (best-val model): mean_dice: %f  mean_hd95: %f',
                 performance, mean_hd95)

    return performance, mean_hd95


def plot_result(dice, h, snapshot_path, args):
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

from tqdm import tqdm

def trainer_3d(args, model, snapshot_path):
    import datetime, math
    date_and_time = datetime.datetime.now()

    os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
    test_save_path = os.path.join(snapshot_path, 'test')

    # Logging
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

    # 🔑 Adjust max_epochs from max_iterations if provided
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

    # --- Losses ---
    from models.Losses import GeneralizedDiceLoss
    ce_loss = CrossEntropyLoss()
    gdl_loss = GeneralizedDiceLoss(softmax=True)

    # --- Optimizer & Scheduler ---
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # Resume training if checkpoint exists
    model, optimizer, scaler, start_epoch, iter_num = load_checkpoint(
        model, optimizer, scaler, snapshot_path, device
    )

    max_epoch = args.max_epochs
    logging.info("%d iterations per epoch", len(train_loader))

    best_performance = 0.0
    patience = getattr(args, "patience", 2)
    counter = 0
    dice_hist, hd95_hist = [], []

    for epoch_num in range(start_epoch, max_epoch):
        model.train()
        epoch_loss = 0.0

        # ✅ Wrap train_loader with tqdm
        with tqdm(train_loader, desc=f"Epoch [{epoch_num+1}/{max_epoch}]", unit="batch") as pbar:
            for i_batch, sampled_batch in enumerate(pbar):
                image_batch = sampled_batch['image'].to(device, non_blocking=True)
                label_batch = sampled_batch['label'].to(device, non_blocking=True)

                try:
                    with torch.amp.autocast("cuda", enabled=True):
                        seg_logits, _, _ = model(image_batch)
                        loss_ce = ce_loss(seg_logits, label_batch.long())
                        loss_gdl = gdl_loss(seg_logits, label_batch)

                        # ✅ New loss: 0.5 * CE + 0.5 * GDL
                        loss = 0.5 * loss_ce + 0.5 * loss_gdl

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    iter_num += 1
                    epoch_loss += loss.item()

                    # update progress bar
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "CE": f"{loss_ce.item():.4f}",
                        "GDL": f"{loss_gdl.item():.4f}"
                    })

                    # log to tensorboard
                    writer.add_scalar('info/total_loss', loss.item(), iter_num)
                    writer.add_scalar('info/loss_ce', loss_ce.item(), iter_num)
                    writer.add_scalar('info/loss_gdl', loss_gdl.item(), iter_num)

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        logging.warning("OOM at iter %d, skipping", iter_num)
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise

        avg_epoch_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch_num+1}/{max_epoch} finished, Avg Loss = {avg_epoch_loss:.4f}")

        # Step the scheduler once per epoch
        scheduler.step()
        writer.add_scalar('info/lr', scheduler.get_last_lr()[0], epoch_num)

        # 🔑 Validation every eval_interval
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            mean_dice, mean_hd95 = inference_3d(model, val_loader, args, test_save_path=test_save_path)
            dice_hist.append(mean_dice)
            hd95_hist.append(mean_hd95)

            if mean_dice > best_performance:
                best_performance = mean_dice
                counter = 0
                tqdm.write(f"🌟 New best Dice = {mean_dice:.4f} at epoch {epoch_num}")
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
            else:
                counter += 1
                tqdm.write(f"No improvement. Patience counter = {counter}/{patience}")

            if counter >= patience:
                tqdm.write(f"⏹ Early stopping triggered at epoch {epoch_num}")
                break
            model.train()

    plot_result(dice_hist, hd95_hist, snapshot_path, args)
    writer.close()
    return "Training Finished!"
