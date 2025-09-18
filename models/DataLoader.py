import os, glob, random
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import elasticdeform

# =========================
# Visualization Utilities
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
def zscore_normalize(img):
    mean, std = np.mean(img), np.std(img)
    if std < 1e-6:
        return np.zeros_like(img)
    return (img - mean) / std

def resize_volume(img, target_shape=(128, 128, 128)):
    factors = (
        target_shape[0] / img.shape[0],
        target_shape[1] / img.shape[1],
        target_shape[2] / img.shape[2],
    )
    return ndimage.zoom(img, factors, order=1)

def crop_foreground(modalities, seg, crop_size=(96, 96, 96)):
    non_zero = np.argwhere(seg > 0)
    if len(non_zero) > 0:
        center = non_zero[np.random.choice(len(non_zero))]
    else:  # random fallback
        center = [np.random.randint(crop_size[i]//2, seg.shape[i]-crop_size[i]//2) for i in range(3)]

    slices = []
    for i in range(3):
        start = max(0, center[i] - crop_size[i]//2)
        end = min(seg.shape[i], start + crop_size[i])
        slices.append(slice(start, end))

    seg_crop = seg[slices[0], slices[1], slices[2]]
    mod_crop = modalities[:, slices[0], slices[1], slices[2]]
    return mod_crop, seg_crop

# =========================
# Data Augmentation
# =========================
def augment(modalities, seg):
    if random.random() > 0.5:
        modalities = np.flip(modalities, axis=2).copy()
        seg = np.flip(seg, axis=2).copy()

    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        for i in range(modalities.shape[0]):
            modalities[i] = ndimage.rotate(modalities[i], angle, axes=(0, 1), reshape=False, order=1)
        seg = ndimage.rotate(seg, angle, axes=(0, 1), reshape=False, order=0)

    if random.random() > 0.5:
        factor = random.uniform(0.9, 1.1)
        modalities = modalities * factor

    if random.random() > 0.5:
        gamma = random.uniform(0.8, 1.2)
        modalities = np.power(modalities, gamma)

    if random.random() > 0.5:
        [modalities, seg] = elasticdeform.deform_random_grid(
            [modalities, seg], sigma=5, points=3, order=[1, 0]
        )

    if random.random() > 0.5:
        noise = np.random.normal(0, 0.01, modalities.shape)
        modalities = modalities + noise

    return modalities, seg

# =========================
# Collate Function
# =========================
def custom_collate(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = [item["label"] for item in batch]
    case_names = [item["case_name"] for item in batch]

    if all(lbl is not None for lbl in labels):
        labels = torch.stack(labels, dim=0)
    else:
        labels = None

    return {"image": images, "label": labels, "case_name": case_names}

# =========================
# BraTS Dataset
# =========================
class BraTSDataset(Dataset):
    def __init__(self, case_dirs, transform=True, target_shape=(128,128,128), crop_size=(96,96,96)):
        self.case_dirs = case_dirs
        self.transform = transform
        self.target_shape = target_shape
        self.crop_size = crop_size
        print(f"Loaded {len(self.case_dirs)} cases | Transform: {self.transform}")

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]
        modality_files = {
            "t1": glob.glob(os.path.join(case_dir, "*t1.nii*"))[0],
            "t1ce": glob.glob(os.path.join(case_dir, "*t1ce.nii*"))[0],
            "t2": glob.glob(os.path.join(case_dir, "*t2.nii*"))[0],
            "flair": glob.glob(os.path.join(case_dir, "*flair.nii*"))[0],
        }

        modalities = []
        for key in ["t1", "t1ce", "t2", "flair"]:
            img = nib.load(modality_files[key]).get_fdata()
            img = zscore_normalize(img)
            img = resize_volume(img, self.target_shape)
            modalities.append(img)
        modalities = np.stack(modalities, axis=0)

        seg_files = glob.glob(os.path.join(case_dir, "*seg.nii*"))
        if len(seg_files) > 0:
            seg = nib.load(seg_files[0]).get_fdata()
            seg = resize_volume(seg, self.target_shape).astype(np.int32)
            seg[seg == 4] = 3

            modalities, seg = crop_foreground(modalities, seg, self.crop_size)
            if self.transform:
                modalities, seg = augment(modalities, seg)

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
        seg_file = glob.glob(os.path.join(case, "*seg.nii*"))[0]
        seg = nib.load(seg_file).get_fdata()
        tumor_fraction = np.mean(seg > 0)
        weights.append(1.0 / (tumor_fraction + 1e-3))
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# =========================
# Dataset Split
# =========================
def get_all_cases(data_dir):
    all_cases = sorted(glob.glob(os.path.join(data_dir, "BraTS20_Training_*")))
    valid_cases = []

    for case in all_cases:
        seg_files = glob.glob(os.path.join(case, "*seg.nii*"))
        if len(seg_files) == 0:
            continue
        has_modalities = all(
            len(glob.glob(os.path.join(case, f"*{m}.nii*"))) == 1
            for m in ["t1", "t1ce", "t2", "flair"]
        )
        if not has_modalities:
            continue
        valid_cases.append(case)

    print(f"✅ Using {len(valid_cases)}/{len(all_cases)} cases")
    return valid_cases

def get_train_val_loaders(data_dir, batch_size=1, target_shape=(96,96,96), use_sampler=False):
    all_cases = get_all_cases(data_dir)
    train_cases, val_cases = train_test_split(all_cases, test_size=0.2, random_state=42)

    train_dataset = BraTSDataset(train_cases, transform=True, target_shape=target_shape)
    val_dataset   = BraTSDataset(val_cases, transform=False, target_shape=target_shape)

    if use_sampler:
        sampler = make_weighted_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=4, pin_memory=True, collate_fn=custom_collate)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, collate_fn=custom_collate)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=custom_collate)
    return train_loader, val_loader
