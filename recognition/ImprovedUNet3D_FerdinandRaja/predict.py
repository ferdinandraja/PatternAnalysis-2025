# predict_3d.py
import os
from glob import glob
import torch
import nibabel as nib
import numpy as np
from modules import ImprovedUNet3D
from datasets import resample_volume
import argparse

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model = ImprovedUNet3D(n_channels=1, n_classes=1, base_c=args.base_c, dropout=args.dropout).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    image_files = sorted(glob(os.path.join(args.images, '**', '*.nii*'), recursive=True))
    for img_path in image_files:
        vol = nib.load(img_path).get_fdata()
        if vol.ndim == 4:
            vol = vol[...,0]
        if vol.ndim == 2:
            vol = np.expand_dims(vol, 0)
        # resample to model target shape
        vol_rs = resample_volume(vol, tuple(args.target_shape), order=1)
        vol_rs = (vol_rs - vol_rs.mean()) / (vol_rs.std()+1e-8)
        x = torch.tensor(vol_rs[None, None, ...], dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0,0]
            pred = (probs > args.threshold).astype(np.uint8)
        # resample back to original shape
        pred_back = resample_volume(pred, vol.shape, order=0)
        # save nifti (reuse affine if present)
        try:
            affine = nib.load(img_path).affine
            nib.Nifti1Image((pred_back).astype(np.uint8), affine).to_filename(os.path.join(args.out_dir, os.path.basename(img_path).replace('.nii.gz','') + '_pred.nii.gz'))
        except Exception:
            np.save(os.path.join(args.out_dir, os.path.basename(img_path) + '_pred.npy'), pred_back)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out_dir', default='preds_3d')
    parser.add_argument('--target_shape', nargs=3, type=int, default=[64,128,128])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--base_c', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    args.target_shape = tuple(args.target_shape)
    run(args)
