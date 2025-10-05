import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import random
import time

import configs.BEFUnet_Config as configs
from models.BEFUnet import BEFUnet3D
from models.DataLoader import get_train_val_loaders
from utils import calculate_metric_percase

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "/content/drive/MyDrive/outputs/BEFUnet3D/BEFUnet3D_best.pth"

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str,
                    default='/content/drive/MyDrive/outputs/BEFUnet3D', help='root dir for output log')
parser.add_argument('--model_name', type=str, default='BEFUnet3D')
parser.add_argument('--root_path', type=str,
                    default='/content/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData',
                    help='root dir for training data')
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
# 🔹 Visualize single test case (only tumor cases)
# -------------------------------
@torch.no_grad()
def test_single_case(model, testloader, output_dir):
    model.eval()

    # ✅ Ensure a different random seed each run
    random.seed(time.time())
    np.random.seed(int(time.time()) % 2**32)
    torch.manual_seed(int(time.time()) % 2**32)

    valid_case_found = False
    attempt = 0

    while not valid_case_found and attempt < len(testloader.dataset):
        idx = random.randint(0, len(testloader.dataset) - 1)
        batch = testloader.dataset[idx]
        label = batch["label"]

        # ✅ Check if tumor exists
        if torch.any(label > 0):
            valid_case_found = True
            print(f"✅ Selected random tumor case: {batch['case_name']} (index: {idx})")
        else:
            attempt += 1
            continue

    if not valid_case_found:
        print("⚠️ No tumor case found in the dataset.")
        return None

    # -------------------------------
    # 🔹 Extract and infer
    # -------------------------------
    image = batch["image"].unsqueeze(0).to(DEVICE)
    label = batch["label"].unsqueeze(0).to(DEVICE)
    case_name = batch["case_name"]

    seg_logits, _, _ = model(image)
    pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)

    prediction_np = pred.squeeze(0).cpu().numpy()
    label_np = label.squeeze(0).cpu().numpy()

    tumor_slices = np.where(label_np.sum(axis=(1, 2)) > 0)[0]
    if len(tumor_slices) == 0:
        print(f"⚠️ No visible tumor slice in {case_name}. Skipping visualization.")
        return None

    slice_idx = random.choice(tumor_slices)

    # -------------------------------
    # 🔹 Visualization
    # -------------------------------
    modality_names = ["T1", "T1ce", "T2", "FLAIR"]
    img_slices = [image[0, m, slice_idx].cpu().numpy() for m in range(image.shape[1])]
    pred_slice = prediction_np[slice_idx]
    label_slice = label_np[slice_idx]

    plt.figure(figsize=(18, 6))
    for i, (mod_name, img_slice) in enumerate(zip(modality_names, img_slices)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img_slice, cmap="gray")
        plt.title(mod_name)
        plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(label_slice, cmap="viridis")
    plt.title("Ground Truth")
    plt.axis("off")

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

    acc = pixel_accuracy(prediction_np, label_np)
    print(f"✅ Single-case Accuracy for {case_name}: {acc:.4f}")
    return acc


# -------------------------------
# 🔹 Main
# -------------------------------
if __name__ == "__main__":
    CONFIGS = {'BEFUnet3D': configs.get_BEFUnet_configs()}
    _, test_loader = get_train_val_loaders(args.root_path, batch_size=1)

    model = BEFUnet3D(config=CONFIGS['BEFUnet3D'], n_classes=4).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print("✅ Loaded trained model.")

    test_single_case(model, test_loader, args.output_dir)
