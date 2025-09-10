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
from models.DataLoader import BraTSDataset, get_train_val_loaders

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


# def trainer_3d(args, model, snapshot_path):
#     """
#     3D trainer with CE + Dice + Class-wise Discriminative Loss (DLF).
#     Assumes dataset yields ('image': B,C,D,H,W) and 'label': B,D,H,W.
#     Model returns: seg_logits, embeddings, dlf_loss (placeholder in model).
#     """
#     date_and_time = datetime.datetime.now()
#     os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
#     test_save_path = os.path.join(snapshot_path, 'test')

#     # Logging
#     logging.basicConfig(
#         filename=os.path.join(snapshot_path, f"{args.model_name}_{date_and_time:%Y%m%d-%H%M%S}_log.txt"),
#         level=logging.INFO,
#         format='[%(asctime)s.%(msecs)03d] %(message)s',
#         datefmt='%H:%M:%S'
#     )
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))

#     base_lr = args.base_lr
#     num_classes = args.num_classes
#     batch_size = args.batch_size * args.n_gpu

#     # Dataset loaders
#     data_dir = data_dir=args.root_path
#     train_loader, val_loader = get_train_val_loaders(data_dir, batch_size=2)

#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)

#     if args.n_gpu > 1:
#         model = nn.DataParallel(model)
#     model.train()

#     # Losses
#     ce_loss = CrossEntropyLoss()
#     dice_loss = DiceLoss(num_classes)
#     from models.Losses import ClassWiseDiscriminativeLoss
#     dlf_loss_fn = ClassWiseDiscriminativeLoss(ignore_index=0)

#     optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
#     writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

#     iter_num = 0
#     max_epoch = args.max_epochs
#     max_iterations = args.max_epochs * len(train_loader)
#     logging.info("%d iterations per epoch. %d max iterations", len(train_loader), max_iterations)

#     best_performance = 0.0
#     dice_hist, hd95_hist = [], []
#     iterator = tqdm(range(max_epoch), ncols=70)

#     for epoch_num in iterator:
#         for i_batch, sampled_batch in enumerate(train_loader):
#             image_batch = sampled_batch['image'].cuda(non_blocking=True)   # (B, C, D, H, W)
#             label_batch = sampled_batch['label']

#             # ✅ Only move labels to GPU if not None
#             if label_batch is not None:
#                 label_batch = label_batch.cuda(non_blocking=True)

#             # Forward pass
#             seg_logits, embeddings, _ = model(image_batch)

#             if label_batch is not None:
#                 # ✅ Compute losses only if labels exist
#                 loss_ce = ce_loss(seg_logits, label_batch.long())
#                 loss_dice = dice_loss(seg_logits, label_batch, softmax=True)
#                 loss_dlf = dlf_loss_fn(embeddings, label_batch)

#                 # Weighted sum
#                 loss = 0.4 * loss_ce + 0.6 * loss_dice + 0.1 * loss_dlf

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 # Poly LR decay
#                 lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#                 for param_group in optimizer.param_groups:
#                     param_group['lr'] = lr_

#                 # Logging
#                 iter_num += 1
#                 writer.add_scalar('info/lr', lr_, iter_num)
#                 writer.add_scalar('info/total_loss', loss.item(), iter_num)
#                 writer.add_scalar('info/loss_ce', loss_ce.item(), iter_num)
#                 writer.add_scalar('info/loss_dice', loss_dice.item(), iter_num)
#                 writer.add_scalar('info/loss_dlf', loss_dlf.item(), iter_num)

#                 logging.info(
#                     'iter %d : total %.5f | ce %.5f | dice %.5f | dlf %.5f',
#                     iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_dlf.item()
#                 )

#                 # ✅ Only log GT mid-slice if labels exist
#                 if iter_num % 10 == 0:
#                     try:
#                         with torch.no_grad():
#                             B, C, D, H, W = image_batch.shape
#                             mid = D // 2
#                             img_slice = image_batch[0, 0, mid]
#                             img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-6)
#                             writer.add_image('train/Image_mid', img_slice.unsqueeze(0), iter_num)

#                             pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)
#                             writer.add_image('train/Prediction_mid', (pred[0, mid].unsqueeze(0) * 50).float(), iter_num)

#                             if label_batch is not None:
#                                 writer.add_image('train/GroundTruth_mid', (label_batch[0, mid].unsqueeze(0) * 50).float(), iter_num)
#                     except Exception as e:
#                         logging.warning("TB image logging failed: %s", str(e))


#     plot_result(dice_hist, hd95_hist, snapshot_path, args)
#     writer.close()
#     return "Training Finished!"

def trainer_3d(args, model, snapshot_path):
    """
    3D trainer with CE + Dice + Class-wise Discriminative Loss (DLF).
    Uses AMP (mixed precision) to reduce GPU memory.
    """
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

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # Dataset loaders (use the computed batch_size)
    data_dir = args.root_path
    train_loader, val_loader = get_train_val_loaders(data_dir, batch_size=batch_size)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # Model -> device / DataParallel if requested
    model = model.to(device)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    # Losses
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    from models.Losses import ClassWiseDiscriminativeLoss
    dlf_loss_fn = ClassWiseDiscriminativeLoss(ignore_index=0)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)
    logging.info("%d iterations per epoch. %d max iterations", len(train_loader), max_iterations)

    best_performance = 0.0
    dice_hist, hd95_hist = [], []
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(train_loader):
            # Move inputs to device
            image_batch = sampled_batch['image'].to(device, non_blocking=True)   # (B, C, D, H, W)
            label_batch = sampled_batch['label']

            if label_batch is not None:
                label_batch = label_batch.to(device, non_blocking=True)  # (B, D, H, W)

            try:
                # Forward + loss under autocast
                with torch.cuda.amp.autocast(enabled=True):
                    seg_logits, embeddings, _ = model(image_batch)

                    if label_batch is not None:
                        loss_ce = ce_loss(seg_logits, label_batch.long())
                        loss_dice = dice_loss(seg_logits, label_batch, softmax=True)
                        loss_dlf = dlf_loss_fn(embeddings, label_batch)

                        loss = 0.4 * loss_ce + 0.6 * loss_dice + 0.1 * loss_dlf
                    else:
                        loss = None

                # Backprop only if we have labels (i.e., training batch)
                if loss is not None:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # Poly LR decay
                    lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_

                    # Logging
                    iter_num += 1
                    writer.add_scalar('info/lr', lr_, iter_num)
                    writer.add_scalar('info/total_loss', loss.item(), iter_num)
                    writer.add_scalar('info/loss_ce', loss_ce.item(), iter_num)
                    writer.add_scalar('info/loss_dice', loss_dice.item(), iter_num)
                    writer.add_scalar('info/loss_dlf', loss_dlf.item(), iter_num)

                    logging.info(
                        'iter %d : total %.5f | ce %.5f | dice %.5f | dlf %.5f',
                        iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_dlf.item()
                    )

                    # TensorBoard middle slice visualization
                    if iter_num % 10 == 0:
                        try:
                            with torch.no_grad():
                                B, C, D, H, W = image_batch.shape
                                mid = D // 2
                                img_slice = image_batch[0, 0, mid]
                                img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-6)
                                writer.add_image('train/Image_mid', img_slice.unsqueeze(0), iter_num)

                                pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)
                                writer.add_image('train/Prediction_mid', (pred[0, mid].unsqueeze(0) * 50).float(), iter_num)

                                if label_batch is not None:
                                    writer.add_image('train/GroundTruth_mid', (label_batch[0, mid].unsqueeze(0) * 50).float(), iter_num)
                        except Exception as e:
                            logging.warning("TB image logging failed: %s", str(e))

                # If this was an inference-only batch (no labels), you might still want to log predictions:
                else:
                    # small optional logging for prediction-only batch
                    with torch.no_grad():
                        pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)
                        # (optionally) save or visualize predictions here

                # clear some cache occasionally
                if iter_num % 50 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                # Handle OOM gracefully: skip batch and continue (log it)
                if 'out of memory' in str(e):
                    logging.warning("CUDA OOM at iter %d: skipping batch. Error: %s", iter_num, str(e))
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

        # Validation (run every eval_interval epochs)
        if (epoch_num + 1) % args.eval_interval == 0:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("Saved model to %s", save_mode_path)

            logging.info("*" * 20)
            logging.info("Running Inference after epoch %d", epoch_num)
            model.eval()
            mean_dice, mean_hd95 = inference_3d(model, val_loader, args, test_save_path=test_save_path)
            dice_hist.append(mean_dice)
            hd95_hist.append(mean_hd95)
            best_performance = max(best_performance, mean_dice)
            model.train()

        # Final epoch checkpoint + possible last inference
        if epoch_num >= max_epoch - 1:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("Saved model to %s", save_mode_path)

            if not ((epoch_num + 1) % args.eval_interval == 0):
                logging.info("*" * 20)
                logging.info("Running Inference after epoch %d (Last Epoch)", epoch_num)
                model.eval()
                mean_dice, mean_hd95 = inference_3d(model, val_loader, args, test_save_path=test_save_path)
                dice_hist.append(mean_dice)
                hd95_hist.append(mean_hd95)
                best_performance = max(best_performance, mean_dice)
                model.train()

            iterator.close()
            break

    plot_result(dice_hist, hd95_hist, snapshot_path, args)
    writer.close()
    return "Training Finished!"

