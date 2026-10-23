"""DBEFNet: imagery/building encoding, MFEM, AFWM, HAFM and segmentation.

Input:
    image:     (B, 3, H, W), normalized RGB imagery.
    buildings: (B, 5, H, W), [density, area, NND, height, footprint].
Output:
    logits:    (B, 2, H, W), unnormalized class scores.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import ResNetEncoder, RSIEncoder


class MFEM(nn.Module):
    """Multi-scale Feature Extraction Module.

    Three dilation branches aggregate spatial context; a learned spatial/channel
    gate modulates the fused features before residual addition.
    """

    def __init__(self, channels=512):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=rate, dilation=rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 1),
            ) for rate in (1, 2, 3)
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.Sigmoid(),
        )

    def forward(self, x):
        multi_scale = self.fusion(torch.cat([branch(x) for branch in self.branches], dim=1))
        return x + self.attention(multi_scale) * multi_scale


class AFWM(nn.Module):
    """Adaptive Feature Weighted Module for encoded building features.

    Average/max channel descriptors pass through separate Conv-BN-ReLU paths.
    Their concatenation is projected to the input width to generate a gate.
    """

    def __init__(self, channels=512):
        super().__init__()
        self.avg_path = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.max_path = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())

    def forward(self, x):
        average = self.avg_path(F.adaptive_avg_pool2d(x, 1))
        maximum = self.max_path(F.adaptive_max_pool2d(x, 1))
        weights = self.fusion(torch.cat([average, maximum], dim=1))
        return x * weights


class HAFM(nn.Module):
    """Hybrid Attention Fusion Module.

    Three independent attention paths:
        1. RSI queries building keys/values.
        2. Buildings query RSI keys/values.
        3. Buildings attend to themselves.
    Concatenated outputs are projected, channel-weighted and residual-refined.
    """

    def __init__(self, channels=512, num_heads=8):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.rs_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.bf_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.rs_queries_bf = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.bf_queries_rs = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.bf_self_attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, rsi, buildings):
        batch, channels, height, width = rsi.shape
        rsi_tokens = self.rs_projection(rsi).flatten(2).transpose(1, 2)
        bf_tokens = self.bf_projection(buildings).flatten(2).transpose(1, 2)

        r_to_b, _ = self.rs_queries_bf(rsi_tokens, bf_tokens, bf_tokens)
        b_to_r, _ = self.bf_queries_rs(bf_tokens, rsi_tokens, rsi_tokens)
        b_self, _ = self.bf_self_attention(bf_tokens, bf_tokens, bf_tokens)

        maps = [tokens.transpose(1, 2).reshape(batch, channels, height, width)
                for tokens in (r_to_b, b_to_r, b_self)]
        fused = self.fusion(torch.cat(maps, dim=1))
        return self.norm(fused + fused * self.channel_attention(fused))


class FeatureReinjectionBlock(nn.Module):
    """Fuse an upsampled decoder state with the original HAFM feature.

    Re-injecting the original fusion feature at every scale provides a direct
    path from HAFM to the progressively restored spatial grids.
    """

    def __init__(self, previous_channels, fusion_channels, output_channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(previous_channels + fusion_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        )

    def forward(self, previous, fusion, scale):
        output_size = (fusion.shape[-2] * scale, fusion.shape[-1] * scale)
        previous = F.interpolate(previous, size=output_size, mode="bilinear", align_corners=False)
        fusion = F.interpolate(fusion, size=output_size, mode="bilinear", align_corners=False)
        return self.refine(torch.cat([previous, fusion], dim=1))


class SegmentationHead(nn.Module):
    """Segmentation decoder with multi-scale fusion-feature reinjection.

    For a 16x16 HAFM feature, Layer 4 remains at 16x16. Layers 3, 2, and 1
    operate at 32x32, 64x64, and 128x128. At each restored scale, the original
    512-channel fusion feature is resized and concatenated with the previous
    decoder state. The classifier logits are finally resized by four.
    """

    def __init__(self, channels=512, num_classes=2):
        super().__init__()
        self.layer4 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.BatchNorm2d(channels // 2), nn.ReLU(inplace=True),
        )
        self.layer3 = FeatureReinjectionBlock(channels // 2, channels, channels // 4)
        self.layer2 = FeatureReinjectionBlock(channels // 4, channels, channels // 8)
        self.layer1 = FeatureReinjectionBlock(channels // 8, channels, channels // 16)
        self.classifier = nn.Conv2d(channels // 16, num_classes, 1)

    def forward(self, x):
        layer4 = self.layer4(x)
        layer3 = self.layer3(layer4, x, scale=2)
        layer2 = self.layer2(layer3, x, scale=4)
        layer1 = self.layer1(layer2, x, scale=8)
        logits = self.classifier(layer1)
        return F.interpolate(logits, scale_factor=4, mode="bilinear", align_corners=False)


class DBEFNet(nn.Module):
    """Dual-branch encoding and fusion for binary urban village segmentation."""

    def __init__(self):
        super().__init__()
        self.rsi_encoder = RSIEncoder()
        self.mfem = MFEM(channels=512)
        self.building_encoder = ResNetEncoder(in_channels=5)
        self.afwm = AFWM(channels=512)
        self.hafm = HAFM(channels=512, num_heads=8)
        self.segmentation_head = SegmentationHead(channels=512, num_classes=2)

    def forward(self, image, buildings):
        if image.ndim != 4 or buildings.ndim != 4:
            raise ValueError("Inputs must have shape (batch, channels, height, width)")
        if image.shape[1] != 3 or buildings.shape[1] != 5:
            raise ValueError("Expected 3 RSI bands and 5 building morphology bands")
        if image.shape[0] != buildings.shape[0] or image.shape[-2:] != buildings.shape[-2:]:
            raise ValueError("The two inputs must share batch and spatial dimensions")
        if any(size < 32 or size % 32 for size in image.shape[-2:]):
            raise ValueError("Input height and width must be positive multiples of 32")
        if self.training and image.shape[0] < 2:
            raise ValueError("Training requires batch size >= 2 for pooled BatchNorm features")

        rsi_features = self.mfem(self.rsi_encoder(image))
        building_features = self.afwm(self.building_encoder(buildings))
        fused_features = self.hafm(rsi_features, building_features)
        return self.segmentation_head(fused_features)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect DBEFNet input/output dimensions")
    parser.add_argument("--input_shape", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = DBEFNet().to(args.device).eval()
    image = torch.randn(1, 3, args.input_shape, args.input_shape, device=args.device)
    buildings = torch.rand(1, 5, args.input_shape, args.input_shape, device=args.device)
    buildings[:, 4] = (buildings[:, 4] > 0.5).float()
    with torch.inference_mode():
        logits = model(image, buildings)
    print(f"RSI: {tuple(image.shape)}")
    print(f"Buildings: {tuple(buildings.shape)}")
    print(f"Logits: {tuple(logits.shape)}")
