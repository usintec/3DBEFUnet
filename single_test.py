import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import configs.BEFUnet_Config as configs
import argparse
import random

from models.BEFUnet import BEFUnet3D  # ⚠️ adjust if your model file has a different name
from models.DataLoader import get_train_val_loaders
from utils import calculate_metric_percase

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "/content/drive/MyDrive/outputs/BEFUnet3D/BEFUnet3D_best.pth"

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str,
                    default='/content/drive/MyDrive/outputs/BEFUnet3D', help='root dir for output log')
parser.add_argument('--model_name', type=str,
                    default='BEFUnet3D')
parser.add_argument('--root_path', type=str,
                    # default='C:/Users/Olatayo/Documents/Machine-Learning/3DBEFUnet Model/content/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData', help='root dir for training data')
                    default='/content/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData', help='root dir for training data')
args = parser.parse_args()

args.output_dir = os.path.join(args.output_dir, args.model_name)
os.makedirs(args.output_dir, exist_ok=True)
# -------------------------------
# 🔹 Utility for single-case accuracy
# -------------------------------
def pixel_accuracy(pred, label):
    correct = (pred == label).sum()
    total = np.prod(label.shape)
    return correct / total


# -------------------------------
# 🔹 Visualize single test case
# -------------------------------
@torch.no_grad()
def test_single_case(model, testloader, output_dir):
    model.eval()

    # Pick a random index
    idx = random.randint(0, len(testloader.dataset) - 1)
    batch = testloader.dataset[idx]  # dataset returns dict
    image = batch["image"].unsqueeze(0).to(DEVICE)   # (1, 4, D, H, W)
    label = batch["label"].unsqueeze(0).to(DEVICE)   # (1, D, H, W)
    case_name = batch["case_name"]

    # Run inference
    seg_logits, _, _ = model(image)
    pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)  # (1, D, H, W)

    # Convert to numpy
    prediction_np = pred.squeeze(0).cpu().numpy()
    label_np = label.squeeze(0).cpu().numpy()

    # Random slice instead of fixed middle
    slice_idx = random.randint(0, prediction_np.shape[0] - 1)
    
    # Extract all 4 modalities
    modality_names = ["T1", "T1ce", "T2", "FLAIR"]
    img_slices = [image[0, m, slice_idx].cpu().numpy() for m in range(image.shape[1])]
    
    pred_slice = prediction_np[slice_idx]
    label_slice = label_np[slice_idx]

    # Plot 4 modalities + GT + Prediction
    plt.figure(figsize=(18, 6))
    
    for i, (mod_name, img_slice) in enumerate(zip(modality_names, img_slices)):
        plt.subplot(2, 3, i+1)
        plt.imshow(img_slice, cmap="gray")
        plt.title(mod_name)
        plt.axis("off")

    # Ground Truth
    plt.subplot(2, 3, 5)
    plt.imshow(label_slice, cmap="viridis")
    plt.title("Ground Truth")
    plt.axis("off")

    # Prediction
    plt.subplot(2, 3, 6)
    plt.imshow(pred_slice, cmap="viridis")
    plt.title("Prediction")
    plt.axis("off")

    plt.suptitle(f"Case: {case_name}, Slice {slice_idx}")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{case_name}_slice{slice_idx}_result.png")
    plt.savefig(out_file, dpi=150)
    print(f"🖼 Saved visualization to {out_file}")

    # Accuracy
    acc = pixel_accuracy(prediction_np, label_np)
    print(f"✅ Single-case Accuracy for {case_name}: {acc:.4f}")
    return acc

def test_single_case1(model, testloader, output_dir):
    model.eval()

    # Pick a random index
    idx = random.randint(0, len(testloader.dataset) - 1)
    batch = testloader.dataset[idx]  # dataset returns dict
    image = batch["image"].unsqueeze(0).to(DEVICE)   # add batch dim → (1, C, D, H, W)
    label = batch["label"].unsqueeze(0).to(DEVICE)   # (1, D, H, W)
    case_name = batch["case_name"]

    # Run inference
    seg_logits, _, _ = model(image)
    pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)  # (1, D, H, W)

    # Convert to numpy
    prediction_np = pred.squeeze(0).cpu().numpy()
    label_np = label.squeeze(0).cpu().numpy()

    # Random slice instead of fixed middle
    slice_idx = random.randint(0, prediction_np.shape[0] - 1)
    img_slice = image[0, 0, slice_idx].cpu().numpy()
    pred_slice = prediction_np[slice_idx]
    label_slice = label_np[slice_idx]

    # Plot input, GT, prediction
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_slice, cmap="gray")
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(label_slice, cmap="viridis")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_slice, cmap="viridis")
    plt.title("Prediction")
    plt.axis("off")

    plt.suptitle(f"Case: {case_name}, Slice {slice_idx}")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{case_name}_slice{slice_idx}_result.png")
    plt.savefig(out_file, dpi=150)
    print(f"🖼 Saved visualization to {out_file}")

    # Accuracy
    acc = pixel_accuracy(prediction_np, label_np)
    print(f"✅ Single-case Accuracy for {case_name}: {acc:.4f}")
    return acc

# -------------------------------
# 🔹 Main
# -------------------------------
if __name__ == "__main__":
    CONFIGS = {
        'BEFUnet3D': configs.get_BEFUnet_configs(),  # keep configs call if unchanged
    }
    # Load data (reuse val_loader as test set)
    _, test_loader = get_train_val_loaders("/content/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData", batch_size=1)

    # Load model
    model = BEFUnet3D(
        config=CONFIGS['BEFUnet3D'],
        n_classes=4).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print("✅ Loaded trained model.")

    # Run single-case test
    test_single_case1(model, test_loader, args.output_dir)

