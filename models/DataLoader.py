import os, glob, random
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =========================
# Global Seed for Reproducibility
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
# Preprocessing Utilities
# =========================
def normalize(volume):
    """Z-score normalize; fallback to min-max if std=0."""
    mean, std = np.mean(volume), np.std(volume)
    if std < 1e-6:
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
    else:
        volume = (volume - mean) / (std + 1e-8)
        volume = np.clip((volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8), 0, 1)
    return volume.astype(np.float32)

def resize_volume(img, target_shape=(128, 128, 128)):
    factors = tuple(t / s for t, s in zip(target_shape, img.shape))
    return ndimage.zoom(img, factors, order=1)

# =========================
# Data Augmentation (Train only)
# =========================
def augment(modalities, seg):
    # Random flip
    if random.random() > 0.5:
        modalities = np.flip(modalities, axis=2).copy()
        seg = np.flip(seg, axis=2).copy()

    # Random rotation (±15°)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        for i in range(modalities.shape[0]):
            modalities[i] = ndimage.rotate(modalities[i], angle, axes=(0, 1), reshape=False, order=1)
        seg = ndimage.rotate(seg, angle, axes=(0, 1), reshape=False, order=0)

    # Add Gaussian noise
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.02, modalities.shape)
        modalities += noise

    return modalities, seg

# =========================
# Collate Function
# =========================
def custom_collate(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = [item["label"] for item in batch]
    case_names = [item["case_name"] for item in batch]
    labels = torch.stack(labels, dim=0) if all(lbl is not None for lbl in labels) else None
    return {"image": images, "label": labels, "case_name": case_names}

# =========================
# Crop Utility
# =========================
def crop_foreground(modalities, seg, crop_size=(96, 96, 96), tumor_ratio=0.8, et_ratio=0.2):
    H, W, D = seg.shape
    cz, cy, cx = crop_size

    def get_random_crop(center):
        z, y, x = center
        z1, y1, x1 = max(0, z - cz // 2), max(0, y - cy // 2), max(0, x - cx // 2)
        z2, y2, x2 = min(H, z1 + cz), min(W, y1 + cy), min(D, x1 + cx)
        return modalities[:, z1:z2, y1:y2, x1:x2], seg[z1:z2, y1:y2, x1:x2]

    # tumor-centered crop
    if random.random() < tumor_ratio and np.sum(seg) > 0:
        if random.random() < et_ratio and np.any(seg == 1):
            coords = np.argwhere(seg == 1)
        else:
            coords = np.argwhere(seg > 0)
        center = coords[random.randint(0, len(coords) - 1)]
    else:
        center = (
            random.randint(0, H - 1),
            random.randint(0, W - 1),
            random.randint(0, D - 1),
        )

    crop_mods, crop_seg = get_random_crop(center)

    # Pad if smaller
    pad_width = [(0, 0)]
    for dim, target in zip(crop_seg.shape, crop_size):
        pad_before = max((target - dim) // 2, 0)
        pad_after = max(target - dim - pad_before, 0)
        pad_width.append((pad_before, pad_after))
    crop_mods = np.pad(crop_mods, pad_width, mode="constant")
    crop_seg = np.pad(crop_seg, pad_width[1:], mode="constant")

    return crop_mods, crop_seg

# =========================
# Dataset Class
# =========================
class BraTSDataset(Dataset):
    def __init__(self, case_dirs, transform=False, target_shape=(128, 128, 128), center_crop=False):
        self.case_dirs = case_dirs
        self.transform = transform
        self.target_shape = target_shape
        self.center_crop = center_crop
        print(f"Loaded {len(self.case_dirs)} cases | Transform: {self.transform}")

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]
        modalities = []
        modality_names = ["t1", "t1ce", "t2", "flair"]

        # Load 4 modalities
        for m in modality_names:
            file = glob.glob(os.path.join(case_dir, f"*{m}.nii*"))
            if not file:
                raise FileNotFoundError(f"Missing {m} in {case_dir}")
            img = nib.load(file[0]).get_fdata()
            img = normalize(resize_volume(img, self.target_shape))
            modalities.append(img)
        modalities = np.stack(modalities, axis=0)

        # Load segmentation
        seg_files = glob.glob(os.path.join(case_dir, "*seg.nii*"))
        seg = None
        if seg_files:
            seg = nib.load(seg_files[0]).get_fdata()
            seg = resize_volume(seg, self.target_shape)
            seg[seg == 4] = 3
            seg = seg.astype(np.int32)

            # Training = crop + augment
            if self.transform:
                modalities, seg = crop_foreground(modalities, seg, crop_size=self.target_shape)
                modalities, seg = augment(modalities, seg)
            elif self.center_crop:  # Validation = centered crop
                modalities, seg = crop_foreground(modalities, seg, crop_size=self.target_shape, tumor_ratio=0.0)

            seg = torch.tensor(seg, dtype=torch.long)

        return {"image": torch.tensor(modalities, dtype=torch.float32),
                "label": seg,
                "case_name": os.path.basename(case_dir)}

# =========================
# DataLoader Function
# =========================
def get_train_val_loaders(data_dir, batch_size=2, target_shape=(96, 96, 96)):
    all_cases = sorted(glob.glob(os.path.join(data_dir, "BraTS20_Training_*")))
    print(f"Found total {len(all_cases)} cases.")
    train_cases, val_cases = train_test_split(all_cases, test_size=0.2, random_state=42)

    train_dataset = BraTSDataset(train_cases, transform=True, target_shape=target_shape)
    val_dataset = BraTSDataset(val_cases, transform=False, target_shape=target_shape, center_crop=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=2, collate_fn=custom_collate)
    return train_loader, val_loader
