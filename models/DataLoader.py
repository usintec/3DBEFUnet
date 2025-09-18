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
        return np.zeros_like(img, dtype=np.float32)
    return ((img - mean) / std).astype(np.float32)

def resize_volume(img, target_shape=(128,128,128)):
    factors = (target_shape[0]/img.shape[0],
               target_shape[1]/img.shape[1],
               target_shape[2]/img.shape[2])
    return ndimage.zoom(img, factors, order=1).astype(np.float32)

# =========================
# Crop or Pad to Target Shape & Patch Alignment
# =========================
def crop_or_pad(volume, target_shape=(96,96,96), patch_size=4):
    """Crop or pad a volume to target_shape, aligned to patch_size."""
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
def crop_foreground(modalities, seg, crop_size=(96,96,96)):
    """Crop subvolume around tumor or random if no tumor exists."""
    non_zero = np.argwhere(seg > 0)
    if len(non_zero) > 0:
        center = non_zero[np.random.choice(len(non_zero))]
    else:
        center = [np.random.randint(crop_size[i]//2, seg.shape[i]-crop_size[i]//2) for i in range(3)]

    slices = []
    for i in range(3):
        start = max(0, center[i]-crop_size[i]//2)
        end = min(seg.shape[i], start+crop_size[i])
        slices.append(slice(start,end))

    seg_crop = seg[slices[0], slices[1], slices[2]]
    mod_crop = modalities[:, slices[0], slices[1], slices[2]]
    return mod_crop, seg_crop

# =========================
# Data Augmentation
# =========================
def augment(modalities, seg, target_shape=(96,96,96), patch_size=4):
    # Random flip
    if random.random() > 0.5:
        modalities = np.flip(modalities, axis=2).copy()
        seg = np.flip(seg, axis=2).copy()

    # Random rotation
    if random.random() > 0.5:
        angle = random.uniform(-15,15)
        for i in range(modalities.shape[0]):
            modalities[i] = ndimage.rotate(modalities[i], angle, axes=(0,1), reshape=False, order=1)
        seg = ndimage.rotate(seg, angle, axes=(0,1), reshape=False, order=0)

    # Random intensity scaling
    if random.random() > 0.5:
        factor = random.uniform(0.9,1.1)
        modalities = modalities * factor

    # Random gamma correction (safe)
    if random.random() > 0.5:
        gamma = random.uniform(0.8,1.2)
        modalities = np.clip(modalities, 0, None)
        modalities = np.power(modalities, gamma)

    # Gaussian noise
    if random.random() > 0.5:
        noise = np.random.normal(0,0.01, modalities.shape)
        modalities = modalities + noise

    # Ensure patch-aligned shape
    modalities = crop_or_pad(modalities, target_shape, patch_size)
    seg = crop_or_pad(seg, target_shape, patch_size)

    return modalities, seg

# =========================
# BraTS Dataset
# =========================
class BraTSDataset(Dataset):
    def __init__(self, case_dirs, transform=True, target_shape=(128,128,128), crop_size=(96,96,96), patch_size=4):
        self.case_dirs = case_dirs
        self.transform = transform
        self.target_shape = target_shape
        self.crop_size = crop_size
        self.patch_size = patch_size
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

            # Tumor-aware crop
            modalities, seg = crop_foreground(modalities, seg, self.crop_size)

            if self.transform:
                modalities, seg = augment(modalities, seg, self.crop_size, self.patch_size)

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
        valid_cases.append(case)
    print(f"✅ Using {len(valid_cases)}/{len(all_cases)} cases with labels")
    return valid_cases

def get_train_val_loaders(data_dir, batch_size=2, target_shape=(96,96,96), crop_size=(96,96,96), patch_size=4, use_weighted_sampler=False):
    all_cases = get_all_cases(data_dir)
    train_cases, val_cases = train_test_split(all_cases, test_size=0.2, random_state=42)

    train_dataset = BraTSDataset(train_cases, transform=True, target_shape=target_shape,
                                 crop_size=crop_size, patch_size=patch_size)
    val_dataset = BraTSDataset(val_cases, transform=False, target_shape=target_shape,
                               crop_size=crop_size, patch_size=patch_size)

    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader
