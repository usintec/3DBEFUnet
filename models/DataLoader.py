import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import scipy.ndimage as ndimage
import os, glob, nibabel as nib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# =========================
# 4. Visualize MRI slices
# =========================
def plot_slices(img_data, title="MRI Slices"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_data[:, :, img_data.shape[2]//2], cmap="gray")
    axes[0].set_title("Axial Slice")
    axes[1].imshow(img_data[:, img_data.shape[1]//2, :], cmap="gray")
    axes[1].set_title("Coronal Slice")
    axes[2].imshow(img_data[img_data.shape[0]//2, :, :], cmap="gray")
    axes[2].set_title("Sagittal Slice")
    plt.suptitle(title)
    plt.show()

def plot_mri_modalities(modalities, slice_idx=None, modality_names=None):
    """
    Plot MRI slices for different modalities.

    Args:
        modalities (torch.Tensor or np.ndarray): Shape (4, H, W, D)
        slice_idx (int): Index of the slice along the axial plane (D). If None, uses the middle slice.
        modality_names (list): List of modality labels, e.g. ["T1", "T1ce", "T2", "FLAIR"]
    """
    if isinstance(modalities, torch.Tensor):
        modalities = modalities.numpy()

    if modalities.shape[0] != 4:
        raise ValueError(f"Expected 4 modalities, got {modalities.shape[0]}")

    if slice_idx is None:
        slice_idx = modalities.shape[-1] // 2  # middle slice

    if modality_names is None:
        modality_names = ["T1", "T1ce", "T2", "FLAIR"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        axes[i].imshow(modalities[i, :, :, slice_idx], cmap="gray")
        axes[i].set_title(modality_names[i])
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

# =========================
# Preprocessing Utilities
# =========================
def normalize(volume):
    """Normalize volume to [0,1]."""
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val - min_val == 0:
        return np.zeros(volume.shape)
    volume = (volume - min_val) / (max_val - min_val)
    return volume.astype(np.float32)

def resize_volume(img, target_shape=(128, 128, 128)):
    """Resize 3D volume with scipy ndimage zoom."""
    factors = (
        target_shape[0] / img.shape[0],
        target_shape[1] / img.shape[1],
        target_shape[2] / img.shape[2],
    )
    img = ndimage.zoom(img, factors, order=1)  # linear interpolation
    return img

# =========================
# Data Augmentation
# =========================
def augment(modalities, seg):
    """Random flip, rotation, and noise."""
    # Random flip
    if random.random() > 0.5:
        modalities = np.flip(modalities, axis=2).copy()  # flip W axis
        seg = np.flip(seg, axis=1).copy()

    # Random rotation (±15° around z-axis)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        for i in range(modalities.shape[0]):
            modalities[i] = ndimage.rotate(modalities[i], angle, axes=(0, 1), reshape=False, order=1)
        seg = ndimage.rotate(seg, angle, axes=(0, 1), reshape=False, order=0)

    # Random Gaussian noise
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.01, modalities.shape)
        modalities = modalities + noise

    return modalities, seg

# =========================
# BraTS Dataset Class
# =========================
class BraTSDataset(Dataset):
    def __init__(self, case_dirs, transform=True, target_shape=(128,128,128)):
        """
        Args:
            case_dirs (list): List of paths to case folders
            transform (bool): Whether to apply augmentation
            target_shape (tuple): Shape to resize volumes
        """
        self.case_dirs = case_dirs
        self.transform = transform
        self.target_shape = target_shape

        print(f"Loaded {len(self.case_dirs)} cases | Transform: {self.transform}")

    def __len__(self):
        return len(self.case_dirs)
    
    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]

        # Explicitly filter only the 4 modalities
        modality_files = {
            "t1": glob.glob(os.path.join(case_dir, "*t1.nii*")),
            "t1ce": glob.glob(os.path.join(case_dir, "*t1ce.nii*")),
            "t2": glob.glob(os.path.join(case_dir, "*t2.nii*")),
            "flair": glob.glob(os.path.join(case_dir, "*flair.nii*")),
        }

        # Make sure we found exactly one file for each modality
        for k, v in modality_files.items():
            assert len(v) == 1, f"Missing or duplicate {k} modality in {case_dir}"

        # Load 4 MRI modalities
        modalities = []
        for key in ["t1", "t1ce", "t2", "flair"]:
            img = nib.load(modality_files[key][0]).get_fdata()
            img = normalize(img)
            img = resize_volume(img, self.target_shape)
            modalities.append(img)
        modalities = np.stack(modalities, axis=0)  # (4, H, W, D)

        # Load segmentation if available
        seg_files = glob.glob(os.path.join(case_dir, "*seg.nii*"))
        if len(seg_files) > 0:
            seg = nib.load(seg_files[0]).get_fdata()
            seg = resize_volume(seg, self.target_shape)

            # 🔑 Remap BraTS labels: {0,1,2,4} → {0,1,2,3}
            seg = seg.astype(np.int32)
            seg[seg == 4] = 3

            if self.transform:
                modalities, seg = augment(modalities, seg)

            seg = torch.tensor(seg, dtype=torch.long)  # (H,W,D)
        else:
            seg = None  # No ground truth in validation

        modalities = torch.tensor(modalities, dtype=torch.float32)  # (4,H,W,D)

        return {
            "image": modalities,
            "label": seg,
            "case_name": os.path.basename(case_dir)
        }



# =========================
# Dataset Split
# =========================
def get_train_val_loaders(data_dir, batch_size=2, target_shape=(128,128,128)):
    all_cases = sorted(glob.glob(os.path.join(data_dir, "BraTS20_Training_*")))
    print(f"Found total {len(all_cases)} cases.")

    # 80/20 split
    train_cases, val_cases = train_test_split(all_cases, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = BraTSDataset(train_cases, transform=True, target_shape=target_shape)   # with augmentation
    val_dataset   = BraTSDataset(val_cases, transform=False, target_shape=target_shape)   # no augmentation

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
