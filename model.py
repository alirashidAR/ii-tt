import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class MiniUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=4):
        super(MiniUNet, self).__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.up1 = Up(128 + 64, 64)
        self.up2 = Up(64 + 32, 32)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits

def get_sota_model(n_channels=3, n_classes=4):
    """
    Returns a DeepLabV3+ model with a ResNet50 ImageNet-pretrained backbone.
    This is a SOTA model for semantic segmentation on high-resolution images.
    """
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",        # ImageNet-pretrained backbone
        encoder_weights="imagenet",      # Pre-trained weights
        in_channels=n_channels,         # RGB = 3
        classes=n_classes,              # BG, Road, Built, Water
    )
    return model


if __name__ == "__main__":
    model = get_sota_model(n_channels=3, n_classes=4)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
