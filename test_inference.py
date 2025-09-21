import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.BEFUnet3D import BEFUnet3D   # ⚠️ adjust if your model file has a different name
from models.DataLoader import get_train_val_loaders
from utils import calculate_metric_percase

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "/content/drive/MyDrive/outputs/BEFUnet3D/BEFUnet3D_best.pth"


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
def test_single_case(model, testloader):
    model.eval()
    batch = next(iter(testloader))  # take first case
    image = batch["image"].to(DEVICE)       # (1, C, D, H, W)
    label = batch["label"].to(DEVICE)       # (1, D, H, W)
    case_name = batch["case_name"][0]

    seg_logits, _, _ = model(image)
    pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)  # (1, D, H, W)

    # Convert to numpy for metrics
    prediction_np = pred.squeeze(0).cpu().numpy()
    label_np = label.squeeze(0).cpu().numpy()

    # Pick central slice for visualization
    mid_slice = prediction_np.shape[0] // 2
    img_slice = image[0, 0, mid_slice].cpu().numpy()
    pred_slice = prediction_np[mid_slice]
    label_slice = label_np[mid_slice]

    # Plot input, GT, and prediction
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

    plt.suptitle(f"Case: {case_name}")
    plt.show()

    acc = pixel_accuracy(prediction_np, label_np)
    print(f"✅ Single-case Accuracy for {case_name}: {acc:.4f}")
    return acc


# -------------------------------
# 🔹 Evaluate all test cases
# -------------------------------
@torch.no_grad()
def evaluate_all(model, testloader, num_classes=4):
    model.eval()
    metric_sum = None
    acc_sum = 0.0

    for i, batch in tqdm(enumerate(testloader), total=len(testloader)):
        image = batch["image"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        seg_logits, _, _ = model(image)
        pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)

        prediction_np = pred.squeeze(0).cpu().numpy()
        label_np = label.squeeze(0).cpu().numpy()

        # Accuracy
        acc_sum += pixel_accuracy(prediction_np, label_np)

        # Dice/HD95 per class
        metric_i = []
        for c in range(1, num_classes):  # skip background
            metric_i.append(
                calculate_metric_percase(
                    (prediction_np == c).astype(np.uint8),
                    (label_np == c).astype(np.uint8)
                )
            )
        metric_i = np.array(metric_i)

        if metric_sum is None:
            metric_sum = metric_i
        else:
            metric_sum += metric_i

    metric_mean = metric_sum / len(testloader.dataset)
    mean_acc = acc_sum / len(testloader)

    print("\n📊 Evaluation Results (All Test Cases)")
    for i in range(1, num_classes):
        dice_i, hd95_i = metric_mean[i-1]
        print(f"Class {i}: Dice={dice_i:.4f}, HD95={hd95_i:.4f}")

    print(f"Mean Dice: {np.mean(metric_mean, axis=0)[0]:.4f}")
    print(f"Mean HD95: {np.mean(metric_mean, axis=0)[1]:.4f}")
    print(f"Mean Accuracy: {mean_acc:.4f}")


# -------------------------------
# 🔹 Main
# -------------------------------
if __name__ == "__main__":
    # Load data (reuse val_loader as test set)
    _, test_loader = get_train_val_loaders(root_path="/content/data", batch_size=1)

    # Load model
    model = BEFUnet3D(num_classes=4).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print("✅ Loaded trained model.")

    # Run single-case test
    test_single_case(model, test_loader)

    # Run full evaluation
    evaluate_all(model, test_loader, num_classes=4)
