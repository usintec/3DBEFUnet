import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from models.DataLoader import BraTSDataset

# ===================================
# Analyze Processed Dataset
# ===================================
def analyze_processed_dataset(dataset, name="Processed Dataset", output_dir="./outputs"):
    class_counts = Counter()
    voxel_counts = Counter()
    tumor_fractions = []

    for sample in dataset:
        seg = sample["label"]
        case_name = sample["case_name"]

        if seg is None:
            continue

        seg = seg.numpy()

        # Count voxels per class
        for c in np.unique(seg):
            class_counts[c] += np.sum(seg == c)

        voxel_counts['total'] += seg.size
        voxel_counts['tumor'] += np.sum(seg > 0)
        voxel_counts['background'] += np.sum(seg == 0)

        tumor_fractions.append(np.mean(seg > 0))

    # ================================
    # Print Summary
    # ================================
    print(f"\n===== 📊 {name} Summary =====")
    print(f"📂 Cases: {len(dataset)}")

    for c, count in sorted(class_counts.items()):
        label = {
            0: "Background",
            1: "Necrotic/Non-enhancing",
            2: "Edema",
            3: "Enhancing Tumor"
        }.get(c, f"Class {c}")
        print(f"  {label:25s}: {count:,} voxels ({count/voxel_counts['total']*100:.3f}%)")

    print(f"\n   Total voxels: {voxel_counts['total']:,}")
    print(f"   Tumor voxels: {voxel_counts['tumor']:,} ({voxel_counts['tumor']/voxel_counts['total']*100:.3f}%)")
    print(f"   Background voxels: {voxel_counts['background']:,} ({voxel_counts['background']/voxel_counts['total']*100:.3f}%)")

    tumor_fractions = np.array(tumor_fractions)
    print(f"\n⚖️ Tumor fraction (mean): {tumor_fractions.mean():.4f}")
    print(f"   Min: {tumor_fractions.min():.4f}, Max: {tumor_fractions.max():.4f}")

    # ================================
    # Plot Histogram and Save
    # ================================
    plt.figure(figsize=(6,4))
    plt.hist(tumor_fractions, bins=30, color="steelblue", edgecolor="black")
    plt.title(f"Tumor voxel fraction per case ({name})")
    plt.xlabel("Tumor fraction")
    plt.ylabel("Number of cases")
    plt.tight_layout()

    out_file = os.path.join(output_dir, f"{name.replace(' ','_')}_processed_tumor_fraction.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"📁 Saved chart to: {out_file}")


# ===================================
# Main
# ===================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/content/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData',
                        help='path to BraTS data root')
    parser.add_argument('--output_dir', type=str,
                        default='/content/drive/MyDrive/outputs/EdaProcessedDataset',
                        help='directory for saving analytics charts')
    parser.add_argument('--target_shape', type=int, nargs=3, default=(128,128,128),
                        help='resize shape for images/labels')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Gather all cases
    all_cases = sorted([os.path.join(args.data_dir, d)
                        for d in os.listdir(args.data_dir)
                        if d.startswith("BraTS20_Training_")])

    train_cases, val_cases = train_test_split(all_cases, test_size=0.2, random_state=42)

    # Create processed datasets (match DataLoader.py)
    train_dataset = BraTSDataset(train_cases, transform=True, target_shape=tuple(args.target_shape))
    val_dataset = BraTSDataset(val_cases, transform=False, target_shape=tuple(args.target_shape))

    # Run analytics on processed datasets
    analyze_processed_dataset(train_dataset, name="Training Cases (Processed)", output_dir=args.output_dir)
    analyze_processed_dataset(val_dataset, name="Validation Cases (Processed)", output_dir=args.output_dir)
