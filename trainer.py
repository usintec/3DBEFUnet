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
from models.DataLoader import BraTSDataset
# ⚠️ Replace these with your 3D dataset + transforms
# from datasets.dataset_3d import BrainTumor3DDataset, RandomGenerator3D
# If you already have a Synapse-like 3D dataset, import that instead:
# from datasets.dataset_synapse_3d import SynapseDataset3D as BrainTumor3DDataset
# from datasets.dataset_synapse_3d import RandomGenerator3D


@torch.no_grad()
def inference_3d(model, testloader, args, test_save_path=None):
    """
    3D inference over whole volumes. Expects batch size == 1 from testloader.
    testloader must yield dicts with keys: 'image': (1,C,D,H,W), 'label': (1,D,H,W), 'case_name': [str]
    """
    model.eval()
    metric_sum = None  # accumulate per-class metrics (dice, hd95), shape: (C-1, 2)

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), ncols=70):
        image = sampled_batch["image"].cuda(non_blocking=True)   # (1, C, D, H, W)
        label = sampled_batch["label"].cuda(non_blocking=True)   # (1, D, H, W)
        case_name = sampled_batch['case_name'][0]

        # Forward
        logits = model(image)  # (1, num_classes, D, H, W)
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)  # (1, D, H, W)

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
            # If you have a 3D NIfTI writer, call it here. Otherwise skip to keep this generic.
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


def trainer_3d(args, model, snapshot_path):
    """
    3D trainer:
      - assumes dataset yields ('image': B,C,D,H,W) and 'label': B,D,H,W
      - model returns logits: B,num_classes,D,H,W
    """
    date_and_time = datetime.datetime.now()
    os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
    test_save_path = os.path.join(snapshot_path, 'test')

    # Logs
    logging.basicConfig(
        filename=os.path.join(snapshot_path, f"{args.model_name}_{date_and_time:%Y%m%d-%H%M%S}_log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # Datasets (replace with your 3D dataset)
    # Trainer snippet
    db_train = BraTSDataset(
        data_dir=args.root_path,
        transform=None  # or your RandomGenerator3D if you have it
    )

    db_test = BraTSDataset(
        data_dir=args.test_path,
        transform=None
    )
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    testloader = DataLoader(
        db_test,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss = CrossEntropyLoss()              # expects B,C,D,H,W vs B,D,H,W
    dice_loss = DiceLoss(num_classes)         # voxelwise dice, works for 3D volumes

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("%d iterations per epoch. %d max iterations", len(trainloader), max_iterations)

    best_performance = 0.0
    dice_hist, hd95_hist = [], []
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch['image'].cuda(non_blocking=True)   # (B, C, D, H, W)
            label_batch = sampled_batch['label'].cuda(non_blocking=True)   # (B, D, H, W)

            # ⭐ No channel expansion—MRI may already be 1 or multi-channel (e.g., 4 modalities)
            logits = model(image_batch)  # (B, num_classes, D, H, W)

            loss_ce = ce_loss(logits, label_batch.long())
            loss_dice = dice_loss(logits, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Poly LR decay
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/loss_ce', loss_ce.item(), iter_num)
            writer.add_scalar('info/loss_dice', loss_dice.item(), iter_num)

            logging.info('iter %d : loss %.5f | ce %.5f | dice %.5f',
                         iter_num, loss.item(), loss_ce.item(), loss_dice.item())

            # TensorBoard visualization: show middle axial slice
            try:
                if iter_num % 10 == 0:
                    with torch.no_grad():
                        B, C, D, H, W = image_batch.shape
                        mid = D // 2
                        # Normalize per-slice for visualization
                        img_slice = image_batch[0, 0, mid]  # (H, W)
                        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-6)
                        writer.add_image('train/Image_mid', img_slice.unsqueeze(0), iter_num)  # (1,H,W)

                        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)  # (B,D,H,W)
                        writer.add_image('train/Prediction_mid', (pred[0, mid].unsqueeze(0) * 50).float(), iter_num)
                        writer.add_image('train/GroundTruth_mid', (label_batch[0, mid].unsqueeze(0) * 50).float(), iter_num)
            except Exception as e:
                logging.warning("TB image logging failed: %s", str(e))

        # Validation
        if (epoch_num + 1) % args.eval_interval == 0:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("Saved model to %s", save_mode_path)

            logging.info("*" * 20)
            logging.info("Running Inference after epoch %d", epoch_num)
            mean_dice, mean_hd95 = inference_3d(model, testloader, args, test_save_path=test_save_path)
            dice_hist.append(mean_dice)
            hd95_hist.append(mean_hd95)
            best_performance = max(best_performance, mean_dice)
            model.train()

        # Final epoch handling
        if epoch_num >= max_epoch - 1:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("Saved model to %s", save_mode_path)

            if not ((epoch_num + 1) % args.eval_interval == 0):
                logging.info("*" * 20)
                logging.info("Running Inference after epoch %d (Last Epoch)", epoch_num)
                mean_dice, mean_hd95 = inference_3d(model, testloader, args, test_save_path=test_save_path)
                dice_hist.append(mean_dice)
                hd95_hist.append(mean_hd95)
                best_performance = max(best_performance, mean_dice)
                model.train()

            iterator.close()
            break

    plot_result(dice_hist, hd95_hist, snapshot_path, args)
    writer.close()
    return "Training Finished!"
