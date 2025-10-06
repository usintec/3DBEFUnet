import os, glob, random
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =========================
# Global Seed
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =========================
# Visualization
# =========================
def plot_mri_modalities(modalities, slice_idx=None, modality_names=None):
    if isinstance(modalities, torch.Tensor):
        modalities = modalities.numpy()

    if modalities.shape[0] != 4:
        raise ValueError(f"Expected 4 modalities, got {modalities.shape[0]}")

    if slice_idx is None:
        slice_idx = modalities.shape[-1] // 2
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
# Normalization
# =========================
def normalize(volume):
    """Z-score normalize; fallback to min-max if std≈0"""
    mean, std = np.mean(volume), np.std(volume)
    if std < 1e-6:
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
    else:
        volume = (volume - mean) / (std + 1e-8)
        volume = np.clip(volume, -5, 5)
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
    return volume.astype(np.float32)

# =========================
# Resize Utility
# =========================
def resize_volume(img, target_shape=(128, 128, 128)):
    factors = tuple(t / s for t, s in zip(target_shape, img.shape))
    return ndimage.zoom(img, factors, order=1)

# =========================
# Data Augmentation
# =========================
def augment(modalities, seg):
    if random.random() > 0.5:
        modalities = np.flip(modalities, axis=2).copy()
        seg = np.flip(seg, axis=2).copy()

    # Random rotation (±15°)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        for i in range(modalities.shape[0]):
            modalities[i] = ndimage.rotate(modalities[i], angle, axes=(1, 2), reshape=False, order=1)
        seg = ndimage.rotate(seg, angle, axes=(1, 2), reshape=False, order=0)

    # Random scaling
    if random.random() > 0.5:
        scale = random.uniform(0.9, 1.1)
        modalities = ndimage.zoom(modalities, (1, scale, scale, scale), order=1)
        seg = ndimage.zoom(seg, (scale, scale, scale), order=0)

    # Intensity shift
    if random.random() > 0.5:
        shift = np.random.uniform(-0.1, 0.1)
        modalities = modalities + shift

    # Gamma adjustment (safe for negative or zero values)
    if random.random() > 0.5:
        gamma = np.random.uniform(0.8, 1.2)
        modalities = np.clip(modalities, 0, None) ** gamma


    # Gaussian noise
    if random.random() > 0.5:
        modalities += np.random.normal(0, 0.01, modalities.shape)

    return modalities, seg

# =========================
# Crop Utility
# =========================
def crop_foreground(modalities, seg, crop_size=(96, 96, 96), tumor_ratio=0.8, et_ratio=0.4):
    """Random tumor-centered crop; fallback to random context crop."""
    H, W, D = seg.shape
    cz, cy, cx = crop_size

    if random.random() < tumor_ratio and np.sum(seg) > 0:
        if random.random() < et_ratio and np.any(seg == 1):
            coords = np.argwhere(seg == 1)
        else:
            coords = np.argwhere(seg > 0)
        center = coords[random.randint(0, len(coords) - 1)]
    else:
        center = (random.randint(0, H - 1),
                  random.randint(0, W - 1),
                  random.randint(0, D - 1))

    z, y, x = center
    z1, y1, x1 = max(0, z - cz // 2), max(0, y - cy // 2), max(0, x - cx // 2)
    z2, y2, x2 = min(H, z1 + cz), min(W, y1 + cy), min(D, x1 + cx)
    crop_mods = modalities[:, z1:z2, y1:y2, x1:x2]
    crop_seg = seg[z1:z2, y1:y2, x1:x2]

    # Pad if smaller
    pad_width = [(0, 0)]
    for dim, target in zip(crop_seg.shape, crop_size):
        before = max((target - dim) // 2, 0)
        after = max(target - dim - before, 0)
        pad_width.append((before, after))
    crop_mods = np.pad(crop_mods, pad_width, mode="constant")
    crop_seg = np.pad(crop_seg, pad_width[1:], mode="constant")

    return crop_mods, crop_seg

# =========================
# Dataset
# =========================
class BraTSDataset(Dataset):
    def __init__(self, case_dirs, transform=False, target_shape=(128, 128, 128)):
        self.case_dirs = case_dirs
        self.transform = transform
        self.target_shape = target_shape
        print(f"Loaded {len(self.case_dirs)} cases | Transform: {self.transform}")

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]
        modalities = []
        for m in ["t1", "t1ce", "t2", "flair"]:
            file = glob.glob(os.path.join(case_dir, f"*{m}.nii*"))
            if not file:
                raise FileNotFoundError(f"Missing {m} in {case_dir}")
            img = nib.load(file[0]).get_fdata()
            img = normalize(resize_volume(img, self.target_shape))
            modalities.append(img)
        modalities = np.stack(modalities, axis=0)

        seg = None
        seg_file = glob.glob(os.path.join(case_dir, "*seg.nii*"))
        if seg_file:
            seg = nib.load(seg_file[0]).get_fdata()
            seg = resize_volume(seg, self.target_shape)
            seg[seg == 4] = 3
            seg = seg.astype(np.int32)
            if self.transform:
                modalities, seg = crop_foreground(modalities, seg, crop_size=self.target_shape)
                modalities, seg = augment(modalities, seg)
            seg = torch.tensor(seg, dtype=torch.long)

        return {
            "image": torch.tensor(modalities, dtype=torch.float32),
            "label": seg,
            "case_name": os.path.basename(case_dir)
        }

# =========================
# Loader Function
# =========================
def get_train_val_loaders(data_dir, batch_size=2, target_shape=(96, 96, 96)):
    all_cases = sorted(glob.glob(os.path.join(data_dir, "BraTS20_Training_*")))
    train_cases, val_cases = train_test_split(all_cases, test_size=0.2, random_state=42)

    train_dataset = BraTSDataset(train_cases, transform=True, target_shape=target_shape)
    val_dataset = BraTSDataset(val_cases, transform=False, target_shape=target_shape)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    return train_loader, val_loader
