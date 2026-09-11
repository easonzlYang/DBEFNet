"""Backbones used by DBEFNet.

The remote-sensing encoder combines a ResNet-34 C5 feature with an FPN-style
aggregation of Swin stages 2--4.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34

from .swin_transformer import SwinBackbone


class ResNetEncoder(nn.Module):
    """ResNet-34 feature encoder with output stride 32 and 512 channels."""

    out_channels = 512

    def __init__(self, in_channels=3):
        super().__init__()
        backbone = resnet34(weights=None)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.features = nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x):
        return self.features(x)


class SwinPyramidEncoder(nn.Module):
    """Return four Swin feature stages at strides 4, 8, 16, and 32.

    The backbone uses patch size 4, dimensions (96, 192, 384, 768), depths
    (2, 2, 6, 2), head counts (3, 6, 12, 24), and window size 7.
    """

    channels = (96, 192, 384, 768)

    def __init__(self):
        super().__init__()
        self.backbone = SwinBackbone(
            img_size=512, patch_size=4, in_chans=3, embed_dim=96,
            depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
            window_size=7, out_indices=(0, 1, 2, 3),
        )

    def forward(self, image):
        return self.backbone(image)


class SwinPyramidAggregation(nn.Module):
    """Aggregate Swin stages 2--4 through a top-down feature pyramid.

    S4 is projected to 256 channels, upsampled and added to projected S3.
    The smoothed P3 is then added to projected S2. The final P2 map is
    resized to the CNN C5 grid.
    """

    out_channels = 256

    def __init__(self, out_channels=256):
        super().__init__()
        self.reduce_s2 = nn.Conv2d(192, out_channels, 1, bias=False)
        self.reduce_s3 = nn.Conv2d(384, out_channels, 1, bias=False)
        self.reduce_s4 = nn.Conv2d(768, out_channels, 1, bias=False)
        self.smooth3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, features, output_size):
        _, s2, s3, s4 = features
        p4 = self.reduce_s4(s4)
        p3 = self.reduce_s3(s3) + F.interpolate(
            p4, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.smooth3(p3)
        p2 = self.reduce_s2(s2) + F.interpolate(
            p3, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.smooth2(p2)
        return F.interpolate(p2, size=output_size, mode="bilinear", align_corners=False)


class SwinFPNBranch(nn.Module):
    """Swin multi-stage encoder followed by top-down aggregation."""

    out_channels = 256

    def __init__(self):
        super().__init__()
        self.encoder = SwinPyramidEncoder()
        self.aggregation = SwinPyramidAggregation(out_channels=256)

    def forward(self, image, output_size):
        return self.aggregation(self.encoder(image), output_size)


class RSIEncoder(nn.Module):
    """Remote-sensing branch: ResNet C5 + Swin FPN feature."""

    out_channels = 512

    def __init__(self):
        super().__init__()
        self.cnn = ResNetEncoder(in_channels=3)
        self.transformer = SwinFPNBranch()
        self.projection = nn.Sequential(
            nn.Conv2d(512 + 256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

    def forward(self, image):
        local = self.cnn(image)
        context = self.transformer(image, local.shape[-2:])
        return self.projection(torch.cat([local, context], dim=1))
