from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


class RoadNet(nn.Module):
    """DeepLabV3 with a single road/non-road output channel."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.net = deeplabv3_resnet50(weights=None, weights_backbone=weights, aux_loss=True)
        self.net.classifier[-1] = nn.Conv2d(256, 1, 1)
        self.net.aux_classifier[-1] = nn.Conv2d(256, 1, 1)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out if self.training else out["out"]


def load_model(checkpoint: str | Path, device: torch.device) -> RoadNet:
    data = torch.load(checkpoint, map_location=device, weights_only=True)
    model = RoadNet(pretrained=False).to(device)
    model.load_state_dict(data["model"] if "model" in data else data)
    model.eval()
    return model
