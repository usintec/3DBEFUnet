import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Import your 3D model + configs + trainer
from models.BEFUnet import BEFUnet3D
import configs.BEFUnet_Config as configs
from trainer import trainer_3d


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./content/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData', help='root dir for training data')
parser.add_argument('--test_path', type=str,
                    default='./content/brats2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData', help='root dir for test data')
parser.add_argument('--dataset', type=str,
                    default='BraTS2020', help='experiment name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/brats', help='list dir if using pre-split lists')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network (BraTS: 4 classes)')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iteration number to train')
parser.add_argument('--max_epochs', type=int,
                    default=401, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch size per gpu (keep small for 3D)')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--num_workers', type=int, default=2,
                    help='number of workers')
parser.add_argument('--img_size', type=int, default=128,
                    help='input patch size (H, W)')
parser.add_argument('--img_size_d', type=int, default=128,
                    help='input patch size (D)')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--output_dir', type=str,
                    default='./results', help='root dir for output log')
parser.add_argument('--model_name', type=str,
                    default='BEFUnet3D')
parser.add_argument('--eval_interval', type=int,
                    default=20, help='evaluation epoch')

args = parser.parse_args()

args.output_dir = os.path.join(args.output_dir, args.model_name)
os.makedirs(args.output_dir, exist_ok=True)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    CONFIGS = {
        'BEFUnet3D': configs.get_BEFUnet_configs(),  # keep configs call if unchanged
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    model = BEFUnet3D(
        config=CONFIGS['BEFUnet3D'],
        img_size=args.img_size,
        in_chans=4,  # BraTS has 4 modalities
        n_classes=args.num_classes
    ).cuda()

    trainer_3d(args, model, args.output_dir)
