# train_3d.py
import os
import argparse
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import csv

from modules import ImprovedUNet3D, DiceLoss3D, dice_coeff_3d
from datasets import NiftiVolumeDataset
from sklearn.model_selection import train_test_split

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    # Build full volume list
    image_files = sorted(glob(os.path.join(args.images, '**', '*.nii*'), recursive=True))
    mask_files = sorted(glob(os.path.join(args.masks, '**', '*.nii*'), recursive=True))
    if len(image_files) == 0:
        raise RuntimeError("No image volumes found")
    if len(mask_files) == 0:
        raise RuntimeError("No mask volumes found")
    # 70% train / 20% val / 10% test split
    train_imgs, temp_imgs = train_test_split(image_files, test_size=0.3, random_state=args.seed)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=1/3, random_state=args.seed)
    print(f"Dataset split → Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

    def build_from_list(img_list):
        ds = NiftiVolumeDataset(args.images, args.masks, target_shape=tuple(args.target_shape), norm=True, augment=args.augment)
        allowed = set(img_list)
        ds.index = [it for it in ds.index if it[0] in allowed]
        if len(ds.index) == 0:
            raise RuntimeError("No volumes after filtering")
        return ds
    train_ds = build_from_list(train_imgs)
    val_ds = build_from_list(val_imgs)
    test_ds = build_from_list(test_imgs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,num_workers=max(0, args.num_workers // 2))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,num_workers=max(0, args.num_workers // 2))

    model = ImprovedUNet3D(n_channels=1, n_classes=1, base_c=args.base_c, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss3D()

    train_losses, val_losses, val_dices = [], [], []
    best_val_dice = 0.0

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'train_log.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_dice'])
#train
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f'Epoch {epoch} - train'):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = bce(logits.squeeze(1), masks.squeeze(1)) + dice_loss(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        v_loss, v_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f'Epoch {epoch} - val'):
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = bce(logits.squeeze(1), masks.squeeze(1)) + dice_loss(logits, masks)
                v_loss += loss.item() * imgs.size(0)
                v_dice += dice_coeff_3d(logits, masks) * imgs.size(0)
        v_loss /= len(val_loader.dataset)
        v_dice /= len(val_loader.dataset)
        val_losses.append(v_loss)
        val_dices.append(v_dice)

        print(f"Epoch {epoch}: train_loss={epoch_loss:.4f} | val_loss={v_loss:.4f} | val_dice={v_dice:.4f}")

        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, epoch_loss, v_loss, v_dice])

        if v_dice > best_val_dice:
            best_val_dice = v_dice
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'dice': v_dice
            }, os.path.join(args.out_dir, 'best.pth'))

#plotting
    plt.figure(); plt.plot(train_losses, label='train_loss'); plt.plot(val_losses, label='val_loss')
    plt.legend(); plt.savefig(os.path.join(args.out_dir, 'loss_curve.png'))
    plt.figure(); plt.plot(val_dices, label='val_dice'); plt.legend()
    plt.savefig(os.path.join(args.out_dir, 'val_dice.png'))
    torch.save({'model_state': model.state_dict()}, os.path.join(args.out_dir, 'final.pth'))
#testing
    model.eval()
    test_dice = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc='Testing'):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            test_dice += dice_coeff_3d(logits, masks) * imgs.size(0)
    test_dice /= len(test_loader.dataset)
    print(f"Final Test Dice coefficient = {test_dice:.4f}")
    with open(os.path.join(args.out_dir, 'final_test_dice.txt'), 'w') as f:
        f.write(f"Final Test Dice coefficient = {test_dice:.4f}\n")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--masks', required=True)
    parser.add_argument('--out_dir', default='runs/unet3d')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base_c', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--target_shape', nargs=3, type=int, default=[64, 128, 128], help='Z Y X')
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()
    args.target_shape = tuple(args.target_shape)
    train(args)
