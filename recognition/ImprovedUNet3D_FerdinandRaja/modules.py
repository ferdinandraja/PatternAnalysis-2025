# modules_3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_ch, out_ch)
        )
    def forward(self, x):
        return self.pool_conv(x)

class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # padding 
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class ImprovedUNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, base_c=16, trilinear=True, dropout=0.1):
        super().__init__()
        self.inc = DoubleConv3D(n_channels, base_c)
        self.down1 = Down3D(base_c, base_c*2)
        self.down2 = Down3D(base_c*2, base_c*4)
        self.down3 = Down3D(base_c*4, base_c*8)
        factor = 2 if trilinear else 1
        self.down4 = Down3D(base_c*8, base_c*16 // factor)

        self.up1 = Up3D(base_c*16, base_c*8 // factor, trilinear)
        self.up2 = Up3D(base_c*8, base_c*4 // factor, trilinear)
        self.up3 = Up3D(base_c*4, base_c*2 // factor, trilinear)
        self.up4 = Up3D(base_c*2, base_c, trilinear)

        self.outc = OutConv3D(base_c, n_classes)
        if dropout and dropout > 0:
            self.drop = nn.Dropout3d(dropout)
        else:
            self.drop = None

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        if self.drop is not None: x2 = self.drop(x2)
        x3 = self.down2(x2)
        if self.drop is not None: x3 = self.drop(x3)
        x4 = self.down3(x3)
        if self.drop is not None: x4 = self.drop(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Dice metric / loss for 3D
def dice_coeff_3d(pred, target, eps=1e-6):
    if pred.dim() == 5 and pred.size(1) > 1:
        pred = torch.softmax(pred, dim=1)[:,1,:,:,:]
    elif pred.dim() == 5:
        pred = torch.sigmoid(pred[:,0,:,:,:])
    else:
        pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1).float()
    intersection = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean().item()

class DiceLoss3D(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()
        intersection = (inputs * targets).sum(1)
        union = inputs.sum(1) + targets.sum(1)
        loss = 1 - ((2. * intersection + self.eps) / (union + self.eps))
        return loss.mean()
