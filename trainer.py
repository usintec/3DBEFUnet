# trainer.py (updated)
import os
import sys
import random
import logging
import datetime
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from monai.losses import FocalLoss


# import your utilities and model
from utils import DiceLoss, calculate_metric_percase
from models.BEFUnet import BEFUnet3D
from models.DataLoaderBackup import get_train_val_loaders  # keep your local loader path
from models.Losses import ClassWiseDiscriminativeLoss  # keep using your DLF implementation

# -----------------------
# Reproducibility helper
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# -----------------------
# Boundary Loss (device-safe)
# -----------------------
class BoundaryLossSafe(nn.Module):
    """
    Device-safe Boundary Loss implementation:
    - computes Signed Distance Function (SDF) for each class on CPU using distance_transform_edt,
      then moves SDF to the same device as the prediction probabilities.
    - computes per-class boundary term and averages (skips background by default).
    Reference: Kervadec et al., 2019 (Boundary loss using SDF).
    """
    def __init__(self, num_classes, ignore_background=True):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background

    @staticmethod
    def compute_sdf_np(binary_mask):
        """
        binary_mask: np.array (D,H,W) dtype=uint8 (1 foreground, 0 background)
        returns signed distance map (float32) shape (D,H,W)
        """
        if binary_mask.dtype != np.bool_:
            binary_mask = binary_mask.astype(bool)

        if binary_mask.any():
            pos = distance_transform_edt(binary_mask)
            neg = distance_transform_edt(~binary_mask)
            sdf = neg.astype(np.float32) - pos.astype(np.float32)
        else:
            # if no positive voxels, sdf = negative distance to background (all zeros)
            sdf = -distance_transform_edt(np.ones_like(binary_mask)).astype(np.float32)
        return sdf

    def forward(self, inputs, target):
        """
        inputs: logits or probabilities (B, C, D, H, W)
        target: ints (B, D, H, W)
        """
        device = inputs.device
        # if logits, convert to probabilities
        if inputs.requires_grad:
            probs = torch.softmax(inputs, dim=1)
        else:
            probs = torch.softmax(inputs, dim=1)

        B = target.shape[0]
        total_loss = 0.0
        classes_used = 0

        # compute per-batch per-class SDF on CPU (numpy) to avoid expensive GPU EDTs and device mismatches
        for c in range(self.num_classes):
            if self.ignore_background and c == 0:
                continue

            # build numpy SDF batch
            batch_sdf = []
            for b in range(B):
                gt_b = (target[b].cpu().numpy() == c).astype(np.uint8)  # (D,H,W) uint8
                sdf = self.compute_sdf_np(gt_b)  # (D,H,W)
                batch_sdf.append(sdf)
            batch_sdf = np.stack(batch_sdf, axis=0).astype(np.float32)  # (B, D, H, W)

            # convert to tensor and move to device
            sdf_t = torch.from_numpy(batch_sdf).to(device)  # (B, D, H, W)

            # pick probability for class c
            pc = probs[:, c, ...]  # (B, D, H, W)

            # boundary loss = mean( pc * sdf ), as in Kervadec; note sign conventions vary in literature
            # we use absolute weighting to encourage alignment; the exact variant can be tuned
            # use elementwise multiplication and mean
            loss_c = torch.mean(pc * sdf_t)
            total_loss = total_loss + loss_c
            classes_used += 1

        if classes_used == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total_loss / classes_used

# -----------------------
# Inference used for validation/testing
# -----------------------
@torch.no_grad()
def inference_3d(model, testloader, args, device, test_save_path=None, visualize=False):
    """
    Evaluate model on entire volumes (BraTS style ET, TC, WT)
    - ignores HD95 == inf during averaging
    """
    model.eval()
    metric_sum = {c: np.array([0.0, 0.0], dtype=np.float64) for c in range(1, args.num_classes)}
    metric_counts = {c: 0 for c in range(1, args.num_classes)}

    # class names
    class_names = {1: "ET", 2: "TC", 3: "WT"} if args.num_classes == 4 else {i: f"class{i}" for i in range(1, args.num_classes)}

    pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=70)
    for i_batch, sampled_batch in pbar:
        image = sampled_batch["image"].to(device, non_blocking=True)   # (1, C, D, H, W)
        label = sampled_batch["label"].to(device, non_blocking=True)   # (1, D, H, W)
        case_name = sampled_batch["case_name"][0]

        seg_logits, embeddings, _ = model(image)
        pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)  # (1, D, H, W)

        prediction_np = pred.squeeze(0).cpu().numpy()
        label_np = label.squeeze(0).cpu().numpy()

        metrics_dict = calculate_metric_percase(prediction_np, label_np, num_classes=args.num_classes)

        # print per-case results and accumulate, ignoring invalid hd95 (inf)
        for c in range(1, args.num_classes):
            dice, hd95 = metrics_dict[c]
            if dice is not None and not np.isinf(hd95):
                metric_sum[c] += np.array([dice, hd95])
                metric_counts[c] += 1
                print(f"[Case {case_name}] {class_names[c]} -> Dice: {dice:.4f}, HD95: {hd95:.4f}")
            else:
                print(f"[Case {case_name}] {class_names[c]} -> skipped (Dice={dice}, HD95={hd95})")

        # optional per-case summary
        valid_scores = [(d, h) for (d, h) in [metrics_dict[c] for c in range(1, args.num_classes)] if d is not None and not np.isinf(h)]
        if valid_scores:
            mean_dice_case = np.mean([d for d, _ in valid_scores])
            mean_hd95_case = np.mean([h for _, h in valid_scores])
            logging.info(' idx %d case %s mean_dice %f mean_hd95 %f', i_batch, case_name, mean_dice_case, mean_hd95_case)

    # averages per class
    metric_mean = {}
    for c in range(1, args.num_classes):
        if metric_counts[c] > 0:
            metric_mean[c] = metric_sum[c] / metric_counts[c]
        else:
            metric_mean[c] = (0.0, 0.0)

    for c in range(1, args.num_classes):
        dice, hd95 = metric_mean[c]
        logging.info(f"Mean {class_names[c]}: Dice = {dice:.4f}, HD95 = {hd95:.4f}")

    dices = [metric_mean[c][0] for c in range(1, args.num_classes) if metric_counts[c] > 0]
    hd95s = [metric_mean[c][1] for c in range(1, args.num_classes) if metric_counts[c] > 0]
    performance = np.mean(dices) if dices else 0
    mean_hd95 = np.mean(hd95s) if hd95s else 0

    logging.info('Testing performance: mean_dice: %f  mean_hd95: %f', performance, mean_hd95)
    return performance, mean_hd95

# -----------------------
# Trainer
# -----------------------
def trainer_3d(args, model, snapshot_path):
    # logging setup
    date_and_time = datetime.datetime.now()
    os.makedirs(snapshot_path, exist_ok=True)
    log_fn = os.path.join(snapshot_path, f"{args.model_name}_{date_and_time:%Y%m%d-%H%M%S}_log.txt")
    logging.basicConfig(filename=log_fn, level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # data
    train_loader, val_loader = get_train_val_loaders(args.root_path, batch_size=args.batch_size)
    logging.info("%d train iterations per epoch", len(train_loader))

    # model to device
    model = model.to(device)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # losses
    class_weights = torch.tensor([1.0, 2.0, 2.0, 4.0]).to(device)
    ce_loss = CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(args.num_classes)
    dlf_loss_fn = ClassWiseDiscriminativeLoss(ignore_index=0)
    boundary_loss_fn = BoundaryLossSafe(args.num_classes, ignore_background=True)

    # optimizer + scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    max_iterations = args.max_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations, eta_min=1e-6)

    # amp scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # resume if exists
    def load_checkpoint(snapshot_path):
        ckpts = sorted([f for f in os.listdir(snapshot_path) if f.endswith(".pth") and "_best" not in f])
        if not ckpts:
            return None
        ckpt = os.path.join(snapshot_path, ckpts[-1])
        logging.info("Loading checkpoint %s", ckpt)
        return torch.load(ckpt, map_location=device)

    # optional resume
    start_epoch = 0
    iter_num = 0
    ckpt_state = load_checkpoint(snapshot_path)
    if ckpt_state:
        model.load_state_dict(ckpt_state["model_state"])
        optimizer.load_state_dict(ckpt_state["optimizer_state"])
        if "scaler_state" in ckpt_state and ckpt_state["scaler_state"] is not None:
            scaler.load_state_dict(ckpt_state["scaler_state"])
        start_epoch = ckpt_state.get("epoch", 0) + 1
        iter_num = ckpt_state.get("iter_num", 0)
        logging.info("Resumed from epoch %d iter %d", start_epoch, iter_num)

    writer = SummaryWriter(os.path.join(snapshot_path, "log"))

    best_performance = 0.479687
    patience = getattr(args, "patience", 20)
    counter = 0

    dice_hist, hd95_hist = [], []

    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        for i_batch, sampled_batch in pbar:
            image_batch = sampled_batch["image"].to(device, non_blocking=True)
            label_batch = sampled_batch["label"]
            if label_batch is not None:
                label_batch = label_batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=True):
                seg_logits, embeddings, _ = model(image_batch)

                if label_batch is not None:
                    # loss_ce = ce_loss(seg_logits, label_batch.long())
                    loss_dice = dice_loss(seg_logits, label_batch, softmax=True)
                    # loss_dlf = dlf_loss_fn(embeddings, label_batch)
                    loss_bound = boundary_loss_fn(seg_logits, label_batch)
                    focal_loss = FocalLoss(gamma=2.0)  # ✅ initialize focal_loss
                    # Weighted combination
                    # loss = (0.2 * loss_ce) + (0.5 * loss_dice) + (0.2 * loss_dlf) + (0.1 * loss_bound)
                    # Final hybrid loss
                    loss = 0.4 * loss_dice + 0.4 * focal_loss(seg_logits, label_batch) + 0.2 * loss_bound
                else:
                    loss = None

            if loss is not None:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # step per iteration

                iter_num += 1
                # logging/tensorboard
                writer.add_scalar("train/total_loss", loss.item(), iter_num)
                writer.add_scalar("train/loss_ce", loss_ce.item(), iter_num)
                writer.add_scalar("train/loss_dice", loss_dice.item(), iter_num)
                writer.add_scalar("train/loss_dlf", loss_dlf.item(), iter_num)
                writer.add_scalar("train/loss_bound", loss_bound.item(), iter_num)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], iter_num)

                pbar.set_description(f"Epoch {epoch} Iter {iter_num} Loss {loss.item():.4f}")

                # checkpoint every N iters (lightweight)
                if iter_num % getattr(args, "save_iter", 500) == 0:
                    ckpt = {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "epoch": epoch,
                        "iter_num": iter_num,
                    }
                    fname = f"{args.model_name}_iter{iter_num}.pth"
                    torch.save(ckpt, os.path.join(snapshot_path, fname))
                    logging.info("Saved checkpoint %s", fname)

        # validation
        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            mean_dice, mean_hd95 = inference_3d(model, val_loader, args, device, test_save_path=snapshot_path)
            dice_hist.append(mean_dice)
            hd95_hist.append(mean_hd95)

            writer.add_scalar("val/mean_dice", mean_dice, epoch)
            writer.add_scalar("val/mean_hd95", mean_hd95, epoch)

            if mean_dice > best_performance:
                best_performance = mean_dice
                counter = 0
                logging.info("New best Dice = %.4f at epoch %d", mean_dice, epoch)

                best_ckpt = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "epoch": epoch,
                    "iter_num": iter_num,
                }
                torch.save(best_ckpt, os.path.join(snapshot_path, f"{args.model_name}_best.pth"))
                # if your model has pidinet submodule and you want to save it separately:
                try:
                    if hasattr(model, "All2Cross") and hasattr(model.All2Cross, "pyramid") and hasattr(model.All2Cross.pyramid, "pidinet"):
                        pidinet_state = model.All2Cross.pyramid.pidinet.state_dict()
                        torch.save({"state_dict": pidinet_state}, os.path.join(snapshot_path, f"{args.model_name}_pidinet_best.pth"))
                except Exception as ex:
                    logging.warning("Failed to save pidinet: %s", ex)
            else:
                counter += 1
                logging.info("No improvement. Patience %d/%d", counter, patience)

            if counter >= patience:
                logging.info("Early stopping triggered.")
                break

    # finalize
    writer.close()
    # optional: save training history
    np.savez(os.path.join(snapshot_path, f"{args.model_name}_history.npz"), dice_hist=dice_hist, hd95_hist=hd95_hist)
    logging.info("Training finished. Best performance: %.4f", best_performance)
    return best_performance

# -----------------------
# If this file is executed directly (example)
# -----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="BEFUnet3D")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--base_lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save_iter", type=int, default=500)
    args = parser.parse_args()

    # attach attributes used in functions
    args.model_name = args.model_name
    args.num_classes = args.num_classes
    args.n_gpu = args.n_gpu
    args.base_lr = args.base_lr
    args.max_epochs = args.max_epochs
    args.eval_interval = args.eval_interval
    args.patience = args.patience
    args.save_iter = args.save_iter
    args.root_path = args.root_path

    model = BEFUnet3D(config=None, n_classes=args.num_classes)  # set config appropriately
    trainer_3d(args, model, snapshot_path=f"./outputs/{args.model_name}")
