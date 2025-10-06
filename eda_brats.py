import os, glob, argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

# ================================
# Collect all case directories
# ================================
def get_all_cases(data_dir):
    """Return all BraTS case directories"""
    cases = sorted(glob.glob(os.path.join(data_dir, "BraTS20_Training_*")))
    if not cases:
        raise FileNotFoundError(f"No BraTS cases found in {data_dir}")
    print(f"✅ Found {len(cases)} cases in {data_dir}")
    return cases


# ================================
# Dataset Analytics Function
# ================================
def analyze_dataset(cases, name="Dataset", output_dir="./outputs"):
    class_counts = Counter()
    voxel_counts = Counter()
    tumor_fractions = []

    for case in cases:
        seg_file = glob.glob(os.path.join(case, "*seg.nii*"))
        if not seg_file:
            print(f"⚠️ Skipping {case} (no segmentation file found)")
            continue

        seg = nib.load(seg_file[0]).get_fdata().astype(np.int32)

        # Remap BraTS labels (4 -> 3 = Enhancing Tumor)
        seg[seg == 4] = 3

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
    print(f"📂 Cases: {len(cases)}")

    for c, count in sorted(class_counts.items()):
        label = {0:"Background",1:"Necrotic/Non-enhancing",2:"Edema",3:"Enhancing Tumor"}.get(c, f"Class {c}")
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
    plt.figure(figsize=(6, 4))
    plt.hist(tumor_fractions, bins=30, color="steelblue", edgecolor="black")
    plt.title(f"Tumor voxel fraction per case ({name})")
    plt.xlabel("Tumor fraction")
    plt.ylabel("Number of cases")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{name.replace(' ', '_')}_tumor_fraction.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"📁 Saved chart to: {out_file}")


# ================================
# Main
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/content/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData',
        help='Path to BraTS data root'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/content/drive/MyDrive/outputs/RawDataset',
        help='Directory for saving analytics charts'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all cases using the new logic
    all_cases = get_all_cases(args.data_dir)

    # Optionally split into train/validation
    # train_cases, val_cases = train_test_split(all_cases, test_size=0.2, random_state=42)

    # Run analytics
    analyze_dataset(all_cases, name="All Cases", output_dir=args.output_dir)
    # analyze_dataset(train_cases, name="Training Cases", output_dir=args.output_dir)
    # analyze_dataset(val_cases, name="Validation Cases", output_dir=args.output_dir)
