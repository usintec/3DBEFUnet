import numpy as np
import torch
import random
import scipy.ndimage as ndimage
import os, glob, nibabel as nib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==============================================================
# 🔹 Advanced 3D Data Augmentation
# ==============================================================
def augment_3d(modalities, seg):
    """Apply rich 3D augmentations."""
    # Random flips (x, y, z axes)
    for axis in range(1, 4):
        if random.random() < 0.5:
            modalities = np.flip(modalities, axis=axis).copy()
            seg = np.flip(seg, axis=axis-1).copy()

    # Random rotation around (x,y,z)
    if random.random() < 0.5:
        angle_x, angle_y, angle_z = np.random.uniform(-15, 15, 3)
        for i in range(modalities.shape[0]):
            modalities[i] = ndimage.rotate(modalities[i], angle_x, axes=(1, 2), reshape=False, order=1)
            modalities[i] = ndimage.rotate(modalities[i], angle_y, axes=(0, 2), reshape=False, order=1)
            modalities[i] = ndimage.rotate(modalities[i], angle_z, axes=(0, 1), reshape=False, order=1)
        seg = ndimage.rotate(seg, angle_z, axes=(0, 1), reshape=False, order=0)

    # Random scaling
    if random.random() < 0.3:
        scale = np.random.uniform(0.9, 1.1)
        modalities = ndimage.zoom(modalities, (1, scale, scale, scale), order=1)
        seg = ndimage.zoom(seg, (scale, scale, scale), order=0)

    # Elastic deformation
    if random.random() < 0.3:
        alpha, sigma = 15, 3
        shape = seg.shape
        dx = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dz = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(x + dx, (-1,)), np.reshape(y + dy, (-1,)), np.reshape(z + dz, (-1,))
        for i in range(modalities.shape[0]):
            modalities[i] = ndimage.map_coordinates(modalities[i], indices, order=1, mode='reflect').reshape(shape)
        seg = ndimage.map_coordinates(seg, indices, order=0, mode='reflect').reshape(shape)

    # Intensity shift and gamma correction
    if random.random() < 0.5:
        shift = np.random.uniform(-0.1, 0.1)
        gamma = np.random.uniform(0.8, 1.2)
        modalities = np.clip(np.power(modalities + shift, gamma), 0, 1)

    # Bias-field augmentation (simulate MRI intensity drift)
    if random.random() < 0.3:
        bias = ndimage.gaussian_filter(np.random.randn(*modalities.shape[1:]), sigma=50)
        bias = (bias - bias.min()) / (bias.max() - bias.min())
        modalities = modalities * (0.9 + 0.2 * bias)

    # Gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.01, modalities.shape)
        modalities = np.clip(modalities + noise, 0, 1)

    return modalities, seg


# ==============================================================
# 🔹 Class-Balanced Patch Sampling
# ==============================================================
def sample_patch_balanced(modalities, seg, crop_size=(96,96,96), tumor_ratio=0.8):
    """
    Class-balanced patch sampling:
    - With probability tumor_ratio, sample a patch containing tumor voxels
    - Otherwise, random background/context patch
    """
    H, W, D = seg.shape
    cz, cy, cx = crop_size

    has_tumor = seg.sum() > 0
    if has_tumor and random.random() < tumor_ratio:
        coords = np.argwhere(seg > 0)
        center = coords[random.randint(0, len(coords) - 1)]
    else:
        center = [random.randint(0, H - 1), random.randint(0, W - 1), random.randint(0, D - 1)]

    z, y, x = center
    z1, y1, x1 = max(0, z - cz//2), max(0, y - cy//2), max(0, x - cx//2)
    z2, y2, x2 = min(H, z1 + cz), min(W, y1 + cy), min(D, x1 + cx)
    crop_mod = modalities[:, z1:z2, y1:y2, x1:x2]
    crop_seg = seg[z1:z2, y1:y2, x1:x2]

    # pad if smaller
    pad_mod = [(0, 0)]
    for s, t in zip(crop_seg.shape, crop_size):
        pad_before = (t - s) // 2 if s < t else 0
        pad_after = t - s - pad_before if s < t else 0
        pad_mod.append((pad_before, pad_after))
    crop_mod = np.pad(crop_mod, pad_mod, mode="constant")
    crop_seg = np.pad(crop_seg, pad_mod[1:], mode="constant")

    return crop_mod, crop_seg


# ==============================================================
# 🔹 Integration in BraTSDataset
# ==============================================================
class BraTSDataset(Dataset):
    def __init__(self, case_dirs, transform=True, target_shape=(128,128,128)):
        self.case_dirs = case_dirs
        self.transform = transform
        self.target_shape = target_shape
        print(f"Loaded {len(self.case_dirs)} cases | Transform: {self.transform}")

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]

        modality_files = {
            "t1": glob.glob(os.path.join(case_dir, "*t1.nii*")),
            "t1ce": glob.glob(os.path.join(case_dir, "*t1ce.nii*")),
            "t2": glob.glob(os.path.join(case_dir, "*t2.nii*")),
            "flair": glob.glob(os.path.join(case_dir, "*flair.nii*")),
        }

        for k, v in modality_files.items():
            if len(v) != 1:
                print(f"⚠️ Skipping {case_dir}, issue with {k}: {v}")
                return self.__getitem__((idx+1) % len(self.case_dirs))

        # Load modalities
        modalities = [nib.load(modality_files[m][0]).get_fdata() for m in ["t1", "t1ce", "t2", "flair"]]
        modalities = np.stack([normalize(m) for m in modalities], axis=0)

        # Load segmentation
        seg_files = glob.glob(os.path.join(case_dir, "*seg.nii*"))
        seg = nib.load(seg_files[0]).get_fdata().astype(np.int32) if seg_files else None
        if seg is not None:
            seg[seg == 4] = 3

            # ✅ Class-balanced sampling before augmentation
            modalities, seg = sample_patch_balanced(modalities, seg, crop_size=self.target_shape)

            # ✅ Apply strong augmentations
            if self.transform:
                modalities, seg = augment_3d(modalities, seg)

            seg = torch.tensor(seg, dtype=torch.long)

        modalities = torch.tensor(modalities, dtype=torch.float32)

        return {"image": modalities, "label": seg, "case_name": os.path.basename(case_dir)}


# ==============================================================
# 🔹 Helper: Normalize
# ==============================================================
def normalize(volume):
    min_val, max_val = np.min(volume), np.max(volume)
    if max_val - min_val == 0:
        return np.zeros(volume.shape)
    return ((volume - min_val) / (max_val - min_val)).astype(np.float32)
