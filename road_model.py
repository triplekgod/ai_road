from pathlib import Path

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, lraspp_mobilenet_v3_large


class RoadNet(nn.Module):
    """Road segmenter: fast MobileNetV3 or accurate ResNet-50."""

    def __init__(self, architecture: str = "fast", pretrained: bool = True):
        super().__init__()
        self.architecture = architecture
        if architecture == "fast":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.net = lraspp_mobilenet_v3_large(weights=None, weights_backbone=weights)
            self.net.classifier.low_classifier = nn.Conv2d(40, 1, 1)
            self.net.classifier.high_classifier = nn.Conv2d(128, 1, 1)
        elif architecture == "accurate":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.net = deeplabv3_resnet50(weights=None, weights_backbone=weights, aux_loss=True)
            self.net.classifier[-1] = nn.Conv2d(256, 1, 1)
            self.net.aux_classifier[-1] = nn.Conv2d(256, 1, 1)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out if self.training else out["out"]


def load_model(checkpoint: str | Path, device: torch.device) -> RoadNet:
    data = torch.load(checkpoint, map_location=device, weights_only=True)
    architecture = data.get("architecture", "accurate") if isinstance(data, dict) else "accurate"
    model = RoadNet(architecture, pretrained=False).to(device)
    model.load_state_dict(data["model"] if "model" in data else data)
    model.input_size = tuple(data.get("input_size", (512, 288) if architecture == "accurate" else (320, 180)))
    model.eval()
    return model
