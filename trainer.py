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
from models.DataLoaderBackup import BraTSDataset, get_train_val_loaders
from utils import calculate_metric_percase

@torch.no_grad()
def inference_3d(model, testloader, args, test_save_path=None, apply_msc=False):
    """
    3D inference over whole volumes with per-class BraTS-style metrics (ET, TC, WT).
    """
    model.eval()
    metric_sum = {c: np.array([0.0, 0.0]) for c in range(1, args.num_classes)}
    metric_counts = {c: 0 for c in range(1, args.num_classes)}

    # Import MSC module
    # from models.Clustering import MeanShiftClustering
    # msc = MeanShiftClustering(bandwidth=0.5)

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), ncols=70):
        image = sampled_batch["image"].cuda(non_blocking=True)   # (1, C, D, H, W)
        label = sampled_batch["label"].cuda(non_blocking=True)   # (1, D, H, W)
        case_name = sampled_batch['case_name'][0]

        # Forward pass
        seg_logits, embeddings, _ = model(image)

        # Optionally refine with MSC
        # if apply_msc:
        #     seg_logits = msc(embeddings, seg_logits)

        # Prediction
        pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)  # (1, D, H, W)

        # Convert to numpy
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

def load_checkpoint(model, optimizer, scaler, snapshot_path, device):
    checkpoints = sorted(
        [
            f for f in os.listdir(snapshot_path)
            if f.endswith(".pth")
            and "pidinet" not in f   # 🚨 skip ALL pidinet checkpoints (dynamic names)
        ],
        key=lambda x: os.path.getmtime(os.path.join(snapshot_path, x)),
    )
    if not checkpoints:
        return model, optimizer, scaler, 0, 0  # no resume

    latest_ckpt = os.path.join(snapshot_path, checkpoints[-1])
    print(f"🔄 Resuming from checkpoint: {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "scaler_state" in checkpoint and checkpoint["scaler_state"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])

    start_epoch = checkpoint.get("epoch", 0) + 1

    iter_num = checkpoint.get("iter_num", 0)

    return model, optimizer, scaler, start_epoch, iter_num

def trainer_3d(args, model, snapshot_path):
    import datetime
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

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    train_loader, val_loader = get_train_val_loaders(args.root_path, batch_size=batch_size)

    model = model.to(device)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    class_weights = torch.tensor([1.0, 2.0, 2.0, 4.0]).to(device)
    ce_loss = CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(num_classes)
    from models.Losses import ClassWiseDiscriminativeLoss, BoundaryLoss
    dlf_loss_fn = ClassWiseDiscriminativeLoss(ignore_index=0)
    boundary_loss_fn = BoundaryLoss(num_classes)

    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)
    logging.info("%d iterations per epoch. %d max iterations", len(train_loader), max_iterations)


    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-7)
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    model, optimizer, scaler, start_epoch, iter_num = load_checkpoint(
        model, optimizer, scaler, snapshot_path, device
    )

    best_performance = 0.479687
    patience = getattr(args, "patience", 20)  # 🔑 stop if no improvement for N evals
    counter = 0

    dice_hist, hd95_hist = [], []
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70, initial=iter_num, total=max_iterations)

    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch = sampled_batch['image'].to(device, non_blocking=True)
            label_batch = sampled_batch['label']
            if label_batch is not None:
                label_batch = label_batch.to(device, non_blocking=True)

            try:
                with torch.cuda.amp.autocast(enabled=True):
                    seg_logits, embeddings, _ = model(image_batch)

                    if label_batch is not None:
                        loss_ce = ce_loss(seg_logits, label_batch.long())
                        loss_dice = dice_loss(seg_logits, label_batch, softmax=True)
                        loss_dlf = dlf_loss_fn(embeddings, label_batch)
                        loss_bound = boundary_loss_fn(seg_logits, label_batch)
                        # 🔹 New weighted combination
                        loss = (0.2 * loss_ce) + (0.5 * loss_dice) + (0.2 * loss_dlf) + (0.1 * loss_bound)
                    else:
                        loss = None

                if loss is not None:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
    
                    # 🔥 Scheduler update (replaces manual LR decay)
                    scheduler.step()

                    # Logging
                    current_lr = optimizer.param_groups[0]['lr']
                    iter_num += 1
                    writer.add_scalar('info/lr', current_lr, iter_num)
                    writer.add_scalar('info/total_loss', loss.item(), iter_num)
                    writer.add_scalar('info/loss_ce', loss_ce.item(), iter_num)
                    writer.add_scalar('info/loss_dice', loss_dice.item(), iter_num)
                    writer.add_scalar('info/loss_dlf', loss_dlf.item(), iter_num)

                    logging.info(
                        'iter %d : total %.5f | ce %.5f | dice %.5f | dlf %.5f',
                        iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_dlf.item()
                    )

                    # 🔑 Save checkpoints often (keep last 3 only)
                    if iter_num % 50 == 0:
                        ckpt_filename = f"{args.model_name}_iter{iter_num}.pth"
                        pidinet_filename = f"{args.model_name}_pidinet{iter_num}.pth"
                        save_checkpoint(
                            {
                                "model_state": model.state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                                "scaler_state": scaler.state_dict(),
                                "epoch": epoch_num,
                                "iter_num": iter_num,
                            },
                            snapshot_path,
                            ckpt_filename,
                        )
                        logging.info(f"Checkpoint saved: {ckpt_filename}")
                        save_pidinet3d(model.All2Cross.pyramid.pidinet, snapshot_path, f"{pidinet_filename}")

                        # ✅ Delete older checkpoints, keep last 3
                        ckpts = sorted(
                            [f for f in os.listdir(snapshot_path) if f.startswith(args.model_name) and "iter" in f and "pidinet" not in f],
                            key=lambda x: int(x.split("iter")[1].split(".pth")[0])
                        )
                        pidinet_ckpts = sorted(
                            [
                                f for f in os.listdir(snapshot_path)
                                if f.startswith(args.model_name) and "pidinet" in f and "_best" not in f
                            ],
                            key=lambda x: int(x.split("pidinet")[1].split(".pth")[0])
                        )
                        if len(ckpts) > 3:
                            old_ckpts = ckpts[:-3]
                            for old in old_ckpts:
                                os.remove(os.path.join(snapshot_path, old))
                                logging.info(f"Deleted old checkpoint: {old}")

                        if len(pidinet_ckpts) > 3:
                            old_pidinet = pidinet_ckpts[:-3]
                            for old in old_pidinet:
                                os.remove(os.path.join(snapshot_path, old))
                                logging.info(f"Deleted old pidinet checkpoint: {old}")

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logging.warning("OOM at iter %d, skipping", iter_num)
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

        # 🔑 Validation every eval_interval
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            mean_dice, mean_hd95 = inference_3d(model, val_loader, args, test_save_path=test_save_path)
            dice_hist.append(mean_dice)
            hd95_hist.append(mean_hd95)

            if mean_dice > best_performance:
                best_performance = mean_dice
                counter = 0  # reset patience
                logging.info("New best Dice = %.4f at epoch %d", mean_dice, epoch_num)

                # 🔑 Save best model
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
                save_pidinet3d(model.All2Cross.pyramid.pidinet, snapshot_path, f"{args.model_name}_pidinet_best.pth")
            else:
                counter += 1
                logging.info("No improvement. Patience counter = %d/%d", counter, patience)

            if counter >= patience:
                logging.info("⏹ Early stopping triggered at epoch %d", epoch_num)
                break
            model.train()

    plot_result(dice_hist, hd95_hist, snapshot_path, args)
    writer.close()
    return "Training Finished!"