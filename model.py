"""A small segmentation network intended for CPU inference."""
import torch
from torch import nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class LiteRoadNet(nn.Module):
    """19,233 parameters; logits output. State-dict layout remains unchanged."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU6(inplace=True))
        self.enc1 = DepthwiseSeparableConv(16, 24, 2)   # 1/4
        self.enc2 = DepthwiseSeparableConv(24, 40, 2)   # 1/8
        self.enc3 = DepthwiseSeparableConv(40, 64, 2)   # 1/16
        self.mid = DepthwiseSeparableConv(64, 64)
        self.dec2 = DepthwiseSeparableConv(64 + 40, 40)
        self.dec1 = DepthwiseSeparableConv(40 + 24, 24)
        self.head = nn.Sequential(DepthwiseSeparableConv(24 + 16, 16), nn.Conv2d(16, 1, 1))

    @staticmethod
    def _up(x, reference):
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        stem = self.stem(x)
        e1 = self.enc1(stem)
        e2 = self.enc2(e1)
        e3 = self.mid(self.enc3(e2))
        d2 = self.dec2(torch.cat((self._up(e3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self._up(d2, e1), e1), dim=1))
        out = self.head(torch.cat((self._up(d1, stem), stem), dim=1))
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


class LinearBottleneck(nn.Module):
    """MobileNet-style inverted residual with a linear projection."""

    def __init__(self, in_channels, out_channels, stride=1, expansion=3):
        super().__init__()
        hidden = in_channels * expansion
        self.residual = stride == 1 and in_channels == out_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.layers(x)
        return out + x if self.residual else out


class FastSCNNLite(nn.Module):
    """Fast-SCNN-inspired segmentation, with global context and spatial fusion.

    Uses only standard PyTorch operations and exports at a static NCHW size.
    Global average pooling avoids input-size-dependent pyramid pooling kernels.
    Output is a single full-size logit map; no pretrained weights are bundled.
    """

    def __init__(self):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU6(inplace=True),
            DepthwiseSeparableConv(24, 32, 2),
            DepthwiseSeparableConv(32, 48, 2),
        )
        self.context_encoder = nn.Sequential(
            LinearBottleneck(48, 64, 2),
            LinearBottleneck(64, 64), LinearBottleneck(64, 64),
            LinearBottleneck(64, 96, 2),
            LinearBottleneck(96, 96), LinearBottleneck(96, 96),
            LinearBottleneck(96, 128),
        )
        # No BatchNorm on a 1x1 pooled feature: batch size one is supported.
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(128, 128, 1), nn.ReLU6(inplace=True),
        )
        self.context_projection = nn.Sequential(
            DepthwiseSeparableConv(128, 96), nn.Conv2d(96, 96, 1, bias=False),
            nn.BatchNorm2d(96),
        )
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(48, 96, 1, bias=False), nn.BatchNorm2d(96),
        )
        self.classifier = nn.Sequential(
            nn.ReLU6(inplace=True), DepthwiseSeparableConv(96, 96),
            DepthwiseSeparableConv(96, 64), nn.Dropout2d(.1), nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        spatial = self.downsample(x)
        context = self.context_encoder(spatial)
        context = context + self.global_context(context)
        context = self.context_projection(context)
        context = F.interpolate(context, size=spatial.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.classifier(context + self.spatial_projection(spatial))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


ARCHITECTURES = ("lite_road_net", "fast_scnn_lite")


def create_model(architecture="lite_road_net"):
    """Construct an untrained model. Checkpoint loading is handled separately."""
    constructors = {"lite_road_net": LiteRoadNet, "fast_scnn_lite": FastSCNNLite}
    if architecture not in constructors:
        raise ValueError(f"Unknown architecture {architecture!r}; choose from {ARCHITECTURES}")
    return constructors[architecture]()
