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
@torch.no_grad()
def test_single_case(model, testloader, output_dir):
    model.eval()

    # Pick a random case from the test dataset
    idx = random.randint(0, len(testloader.dataset) - 1)
    batch = testloader.dataset[idx]
    image = batch["image"].unsqueeze(0).to(DEVICE)   # (1, 4, D, H, W)
    label = batch["label"].unsqueeze(0).to(DEVICE)   # (1, D, H, W)
    case_name = batch["case_name"]

    # Run inference
    seg_logits, _, _ = model(image)
    pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)  # (1, D, H, W)

    # Convert to numpy
    prediction_np = pred.squeeze(0).cpu().numpy()
    label_np = label.squeeze(0).cpu().numpy()

    # Choose a slice that contains tumor (fallback to center if none)
    tumor_slices = np.where(label_np.sum(axis=(1, 2)) > 0)[0]
    if len(tumor_slices) > 0:
        slice_idx = random.choice(tumor_slices.tolist())
    else:
        slice_idx = prediction_np.shape[0] // 2

    # Extract all 4 modalities and normalize to [0, 1]
    modality_names = ["T1", "T1ce", "T2", "FLAIR"]
    img_slices = []
    for m in range(image.shape[1]):
        img_slice = image[0, m, slice_idx].cpu().numpy()
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
        img_slices.append(img_slice)

    pred_slice = prediction_np[slice_idx]
    label_slice = label_np[slice_idx]

    # Show class distribution
    unique, counts = np.unique(prediction_np, return_counts=True)
    print(f"Predicted classes for {case_name}: {dict(zip(unique, counts))}")

    # Plot modalities, ground truth, and prediction
    plt.figure(figsize=(18, 6))
    
    for i, (mod_name, img_slice) in enumerate(zip(modality_names, img_slices)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(img_slice, cmap="gray")
        plt.title(mod_name)
        plt.axis("off")

    # Ground Truth (segmentation mask)
    plt.subplot(2, 4, 5)
    plt.imshow(label_slice, cmap="tab10", vmin=0, vmax=3)
    plt.title("Ground Truth")
    plt.axis("off")

    # Prediction (segmentation mask)
    plt.subplot(2, 4, 6)
    plt.imshow(pred_slice, cmap="tab10", vmin=0, vmax=3)
    plt.title("Prediction")
    plt.axis("off")

    # Optional overlay visualization
    plt.subplot(2, 4, 7)
    plt.imshow(img_slices[3], cmap="gray")
    plt.imshow(pred_slice, cmap="tab10", alpha=0.5, vmin=0, vmax=3)
    plt.title("Overlay (FLAIR + Prediction)")
    plt.axis("off")

    plt.suptitle(f"Case: {case_name}, Slice: {slice_idx}")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{case_name}_slice{slice_idx}_result.png")
    plt.savefig(out_file, dpi=150)
    print(f"🖼 Saved visualization to {out_file}")

    # Accuracy (simple pixel-wise)
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
    test_single_case(model, test_loader, args.output_dir)

