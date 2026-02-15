import torch
import torch.nn as nn
import torch.nn.functional as F

def center_crop_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    # Crop x spatially to match ref (H,W)
    _, _, H, W = x.shape
    _, _, Hr, Wr = ref.shape
    y0 = (H - Hr) // 2
    x0 = (W - Wr) // 2
    return x[:, :, y0:y0+Hr, x0:x0+Wr]

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        e3c = center_crop_to(e3, d3)
        d3 = self.dec3(torch.cat([d3, e3c], dim=1))

        d2 = self.up2(d3)
        e2c = center_crop_to(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2c], dim=1))

        d1 = self.up1(d2)
        e1c = center_crop_to(e1, d1)
        d1 = self.dec1(torch.cat([d1, e1c], dim=1))

        return self.out(d1)
