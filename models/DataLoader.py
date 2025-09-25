import os, glob, random
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# =========================
# Normalization
# =========================
def zscore_normalize(img):
    mean, std = np.mean(img), np.std(img)
    if std < 1e-6:
        return np.zeros_like(img)
    return (img - mean) / std

def resize_volume(img, target_shape=(128,128,128)):
    factors = (target_shape[0]/img.shape[0],
               target_shape[1]/img.shape[1],
               target_shape[2]/img.shape[2])
    return ndimage.zoom(img, factors, order=1).astype(np.float32)

# =========================
# Patch-Aligned Crop/Pad
# =========================
def crop_or_pad_patch_aligned(volume, target_shape=(96,96,96), patch_size=4):
    """Crop or pad a 3D/4D volume and ensure all dims are multiples of patch_size."""
    target_shape = [((t + patch_size - 1)//patch_size)*patch_size for t in target_shape]

    if volume.ndim == 3:
        result = np.zeros(target_shape, dtype=volume.dtype)
        slices = [slice(0, min(s,t)) for s,t in zip(volume.shape, target_shape)]
        result[tuple(slices)] = volume[tuple(slices)]
        return result
    elif volume.ndim == 4:
        C = volume.shape[0]
        result = np.zeros((C, *target_shape), dtype=volume.dtype)
        slices = [slice(0,C)] + [slice(0, min(s,t)) for s,t in zip(volume.shape[1:], target_shape)]
        result[tuple(slices)] = volume[tuple(slices)]
        return result
    else:
        raise ValueError(f"Unsupported volume ndim={volume.ndim}")

# =========================
# Tumor-aware Cropping
# =========================
def crop_foreground_backup(modalities, seg, crop_size=(96,96,96)):
    """Crop subvolume around tumor or random if no tumor exists."""
    non_zero = np.argwhere(seg > 0)
    if len(non_zero) > 0:
        center = non_zero[np.random.choice(len(non_zero))]
    else:
        # fallback: pick random location
        center = [np.random.randint(crop_size[i]//2, seg.shape[i]-crop_size[i]//2) for i in range(3)]

    slices = []
    for i in range(3):
        start = max(0, center[i]-crop_size[i]//2)
        end = min(seg.shape[i], start+crop_size[i])
        slices.append(slice(start,end))

    seg_crop = seg[slices[0], slices[1], slices[2]]
    mod_crop = modalities[:, slices[0], slices[1], slices[2]]
    return mod_crop, seg_crop

import numpy as np

def crop_foreground(modalities, seg, crop_size=(96,96,96), tumor_ratio=0.7): 
    """
    Crop subvolume around tumor (with probability tumor_ratio)
    or random context patch otherwise.
    Ensures valid crop regions even if image size < crop size.
    """
    shape = seg.shape

    # Ensure crop_size does not exceed image size
    crop_size = tuple(min(crop_size[i], shape[i]) for i in range(3))

    if np.random.rand() < tumor_ratio and np.any(seg > 0):
        # Tumor-centered crop
        non_zero = np.argwhere(seg > 0)
        center = non_zero[np.random.choice(len(non_zero))]
    else:
        # Random context crop, safe bounds
        center = []
        for i in range(3):
            low = crop_size[i] // 2
            high = max(low + 1, shape[i] - crop_size[i] // 2)
            center.append(np.random.randint(low, high))
    
    # Build slices safely
    slices = []
    for i in range(3):
        start = max(0, center[i] - crop_size[i] // 2)
        end = min(shape[i], start + crop_size[i])

        # Adjust start if crop is smaller than expected
        if end - start < crop_size[i]:
            start = max(0, end - crop_size[i])
        slices.append(slice(start, end))

    seg_crop = seg[slices[0], slices[1], slices[2]]
    mod_crop = modalities[:, slices[0], slices[1], slices[2]]
    return mod_crop, seg_crop


# =========================
# Augmentation
# =========================
def augment_patch_aligned(modalities, seg, target_shape=(96,96,96), patch_size=4):
    """Augment modalities + seg and enforce patch-aligned final shape."""
    # Random flip
    if random.random() > 0.5:
        modalities = np.flip(modalities, axis=2).copy()
        seg = np.flip(seg, axis=2).copy()

    # Random rotation (axes 0-1 only)
    if random.random() > 0.5:
        angle = random.uniform(-15,15)
        for i in range(modalities.shape[0]):
            modalities[i] = ndimage.rotate(modalities[i], angle, axes=(0,1), reshape=False, order=1)
        seg = ndimage.rotate(seg, angle, axes=(0,1), reshape=False, order=0)

    # Random intensity / gamma
    if random.random() > 0.5:
        modalities = modalities * random.uniform(0.9, 1.1)
    if random.random() > 0.5:
        gamma = random.uniform(0.8, 1.2)
        modalities = np.clip(modalities, 0, None)
        modalities = np.power(modalities, gamma)

    # Gaussian noise
    if random.random() > 0.5:
        modalities += np.random.normal(0, 0.01, modalities.shape)

    # Ensure patch-aligned final shape
    modalities = crop_or_pad_patch_aligned(modalities, target_shape, patch_size)
    seg = crop_or_pad_patch_aligned(seg, target_shape, patch_size)

    return modalities, seg

# =========================
# BraTS Dataset
# =========================
class BraTSDataset(Dataset):
    def __init__(self, case_dirs, transform=True, target_shape=(128,128,128), crop_size=(96,96,96), patch_size=4, skip_empty=True):
        self.case_dirs = case_dirs
        self.transform = transform
        self.target_shape = target_shape
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.skip_empty = skip_empty
        print(f"Loaded {len(self.case_dirs)} cases | Transform: {self.transform}")

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]

        # Load modalities
        modality_files = {}
        for m in ["t1","t1ce","t2","flair"]:
            files = glob.glob(os.path.join(case_dir,f"*{m}.nii*"))
            if len(files) != 1:
                print(f"⚠️ Skipping {case_dir}, problem with {m}: {files}")
                return self.__getitem__((idx+1) % len(self.case_dirs))
            modality_files[m] = files[0]

        modalities = []
        for m in ["t1","t1ce","t2","flair"]:
            img = nib.load(modality_files[m]).get_fdata()
            img = zscore_normalize(img)
            img = resize_volume(img, self.target_shape)
            modalities.append(img)
        modalities = np.stack(modalities, axis=0)

        # Load segmentation
        seg_files = glob.glob(os.path.join(case_dir,"*seg.nii*"))
        if len(seg_files) > 0:
            seg = nib.load(seg_files[0]).get_fdata()
            seg = resize_volume(seg, self.target_shape)
            seg = seg.astype(np.int32)
            seg[seg == 4] = 3

            # Skip empty masks if enabled
            if self.skip_empty and np.sum(seg) == 0:
                return self.__getitem__((idx+1) % len(self.case_dirs))

            # Tumor-aware crop
            modalities, seg = crop_foreground(modalities, seg, self.crop_size)

            # Apply patch-aligned augmentation if transform
            if self.transform:
                modalities, seg = augment_patch_aligned(modalities, seg, self.crop_size, self.patch_size)
            else:
                modalities = crop_or_pad_patch_aligned(modalities, self.crop_size, self.patch_size)
                seg = crop_or_pad_patch_aligned(seg, self.crop_size, self.patch_size)

            seg = torch.tensor(seg, dtype=torch.long)
        else:
            seg = None

        modalities = torch.tensor(modalities, dtype=torch.float32)
        return {"image": modalities, "label": seg, "case_name": os.path.basename(case_dir)}

# =========================
# Weighted Sampler
# =========================
def make_weighted_sampler(dataset):
    weights = []
    for case in dataset.case_dirs:
        seg_file = glob.glob(os.path.join(case,"*seg.nii*"))[0]
        seg = nib.load(seg_file).get_fdata()
        tumor_fraction = np.mean(seg>0)
        weights.append(1.0/(tumor_fraction + 1e-3))
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# =========================
# Dataset Split & Loader
# =========================
def get_all_cases(data_dir):
    all_cases = sorted(glob.glob(os.path.join(data_dir,"BraTS20_Training_*")))
    valid_cases = []
    for case in all_cases:
        seg_files = glob.glob(os.path.join(case,"*seg.nii*"))
        if len(seg_files) == 0:
            continue
        has_modalities = all(len(glob.glob(os.path.join(case,f"*{m}.nii*"))) == 1 for m in ["t1","t1ce","t2","flair"])
        if not has_modalities:
            continue
        # Skip cases with empty segmentation
        seg = nib.load(seg_files[0]).get_fdata()
        if np.sum(seg) == 0:
            continue
        valid_cases.append(case)
    print(f"✅ Using {len(valid_cases)}/{len(all_cases)} cases with labels")
    return valid_cases

def get_train_val_loaders(data_dir, batch_size=2, target_shape=(96,96,96), use_weighted_sampler=False, patch_size=4):
    all_cases = get_all_cases(data_dir)
    train_cases, val_cases = train_test_split(all_cases, test_size=0.2, random_state=42)

    train_dataset = BraTSDataset(train_cases, transform=True, target_shape=target_shape, patch_size=patch_size)
    val_dataset = BraTSDataset(val_cases, transform=False, target_shape=target_shape, patch_size=patch_size)

    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader
