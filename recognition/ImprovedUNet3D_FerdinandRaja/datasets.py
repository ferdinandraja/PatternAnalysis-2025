import os
from glob import glob
import numpy as np
import nibabel as nib
import random
import torch
from torch.utils.data import Dataset
import scipy.ndimage as ndi

def list_nifti_files(root):
    paths = glob(os.path.join(root, '**', '*.nii*'), recursive=True)
    paths.sort()
    return paths

def resample_volume(vol, target_shape, order=1):
    factors = (target_shape[0] / vol.shape[0],
               target_shape[1] / vol.shape[1],
               target_shape[2] / vol.shape[2])
    if all(abs(f - 1.0) < 1e-6 for f in factors):
        return vol
    return ndi.zoom(vol, factors, order=order)

def random_flip_3d(img, mask=None, p=0.5):
    if random.random() < p:
        axis = random.choice([0, 1, 2])
        img = np.flip(img, axis=axis).copy()
        if mask is not None:
            mask = np.flip(mask, axis=axis).copy()
    return img, mask


class NiftiVolumeDataset(Dataset):
    """
    3D dataset for HipMRI: pairs *_LFOV.nii.gz (image) with *_SEMANTIC.nii.gz (mask)
    """
    def __init__(self, image_dir, mask_dir=None,
                 target_shape=(64,128,128), norm=True, augment=False):
        self.image_files = list_nifti_files(image_dir)
        self.mask_files = list_nifti_files(mask_dir) if mask_dir else []
        self.mask_map = {}
        if mask_dir:
            for p in self.mask_files:
                key = os.path.basename(p).replace('_SEMANTIC', '')
                self.mask_map[key] = p

        self.index = []
        for img_path in self.image_files:
            base = os.path.basename(img_path).replace('_LFOV', '')
            mask_path = self.mask_map.get(base) if mask_dir else None
            if mask_dir and mask_path is None:
                continue
            self.index.append((img_path, mask_path))
        if len(self.index) == 0:
            raise RuntimeError(f"No matched volumes between {image_dir} and {mask_dir}")

        self.target_shape = target_shape
        self.norm = norm
        self.augment = augment

    def __len__(self):
        return len(self.index)

    def _load_vol(self, path):
        arr = nib.load(path).get_fdata()
        if arr.ndim == 4:
            arr = arr[...,0]
        return arr.astype(np.float32)

    def __getitem__(self, idx):
        img_path, mask_path = self.index[idx]
        img = self._load_vol(img_path)
        mask = self._load_vol(mask_path) if mask_path else None

        if self.augment:
            img, mask = random_flip_3d(img, mask, p=0.5)

        img = resample_volume(img, self.target_shape, order=1)
        if mask is not None:
            mask = resample_volume(mask, self.target_shape, order=0)

        if self.norm:
            img = (img - img.mean()) / (img.std() + 1e-8)

        img = np.expand_dims(img, 0).astype(np.float32)
        if mask is not None:
            mask = (mask > 0).astype(np.uint8)
            mask = np.expand_dims(mask, 0).astype(np.float32)
            return torch.tensor(img), torch.tensor(mask)
        else:
            return torch.tensor(img)
