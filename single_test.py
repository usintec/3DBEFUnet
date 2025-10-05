import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import configs.BEFUnet_Config as configs
import argparse

from models.BEFUnet import BEFUnet3D
from models.DataLoader import get_train_val_loaders
from utils import calculate_metric_percase

# -------------------------------
# 🔹 Configuration
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/content/drive/MyDrive/outputs/BEFUnet3D/BEFUnet3D_best.pth"

# ✅ Easily change these two values to test a specific case/slice
TARGET_CASE_NAME = "BraTS20_Training_197"
TARGET_SLICE = 38  # change this number to pick another slice

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str,
                    default='/content/drive/MyDrive/outputs/BEFUnet3D', help='root dir for output log')
parser.add_argument('--model_name', type=str,
                    default='BEFUnet3D')
parser.add_argument('--root_path', type=str,
                    default='/content/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData', help='root dir for training data')
args = parser.parse_args()

args.output_dir = os.path.join(args.output_dir, args.model_name)
os.makedirs(args.output_dir, exist_ok=True)

# -------------------------------
# 🔹 Pixel Accuracy
# -------------------------------
def pixel_accuracy(pred, label):
    correct = (pred == label).sum()
    total = np.prod(label.shape)
    return correct / total

# -------------------------------
# 🔹 Test a specific case and slice
# -------------------------------
@torch.no_grad()
def test_specific_case(model, testloader, output_dir, target_case, target_slice):
    model.eval()

    # 🔍 Find the target case
    case_index = None
    for i, data in enumerate(testloader.dataset):
        if data["case_name"] == target_case:
            case_index = i
            break

    if case_index is None:
        print(f"❌ Case {target_case} not found in dataset!")
        return

    batch = testloader.dataset[case_index]
    image = batch["image"].unsqueeze(0).to(DEVICE)
    label = batch["label"].unsqueeze(0).to(DEVICE)
    case_name = batch["case_name"]

    # 🔹 Run inference
    seg_logits, _, _ = model(image)
    pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)

    # 🔹 Convert to numpy
    prediction_np = pred.squeeze(0).cpu().numpy()
    label_np = label.squeeze(0).cpu().numpy()

    # 🔹 Ensure slice number is valid
    target_slice = min(max(0, target_slice), prediction_np.shape[0] - 1)
    slice_idx = target_slice

    # 🔹 Extract modalities and normalize
    modality_names = ["T1", "T1ce", "T2", "FLAIR"]
    img_slices = []
    for m in range(image.shape[1]):
        img_slice = image[0, m, slice_idx].cpu().numpy()
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
        img_slices.append(img_slice)

    pred_slice = prediction_np[slice_idx]
    label_slice = label_np[slice_idx]

    # 🔹 Print prediction class distribution
    unique, counts = np.unique(prediction_np, return_counts=True)
    print(f"Predicted classes for {case_name}: {dict(zip(unique, counts))}")

    # 🔹 Plot
    plt.figure(figsize=(18, 6))
    for i, (mod_name, img_slice) in enumerate(zip(modality_names, img_slices)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(img_slice, cmap="gray")
        plt.title(mod_name)
        plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(label_slice, cmap="tab10", vmin=0, vmax=3)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(pred_slice, cmap="tab10", vmin=0, vmax=3)
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(img_slices[3], cmap="gray")
    plt.imshow(pred_slice, cmap="tab10", alpha=0.5, vmin=0, vmax=3)
    plt.title("Overlay (FLAIR + Prediction)")
    plt.axis("off")

    plt.suptitle(f"Case: {case_name}, Slice: {slice_idx}")
    plt.tight_layout()

    out_file = os.path.join(output_dir, f"{case_name}_slice{slice_idx}_result.png")
    plt.savefig(out_file, dpi=150)
    print(f"🖼 Saved visualization to {out_file}")

    # 🔹 Compute accuracy
    acc = pixel_accuracy(prediction_np, label_np)
    print(f"✅ Single-case Accuracy for {case_name}: {acc:.4f}")

    return acc

# -------------------------------
# 🔹 Main
# -------------------------------
if __name__ == "__main__":
    CONFIGS = {
        'BEFUnet3D': configs.get_BEFUnet_configs(),
    }

    # Load dataset (reuse val_loader as test set)
    _, test_loader = get_train_val_loaders(args.root_path, batch_size=1)

    # Load model
    model = BEFUnet3D(config=CONFIGS['BEFUnet3D'], n_classes=4).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print("✅ Loaded trained model.")

    # Run test on the chosen case and slice
    test_specific_case(model, test_loader, args.output_dir,
                       TARGET_CASE_NAME, TARGET_SLICE)
