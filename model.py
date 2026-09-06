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
    """~0.15M parameters, output is logits (apply sigmoid outside the model)."""
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
