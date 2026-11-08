import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Baseline CNN built from scratch."""

    def __init__(self, num_classes=2):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def resnet18_tl(num_classes=2):
    """Pretrained ResNet18 fine-tuned on CT scans."""
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, num_classes))
    return m


def get_model(name):
    return SimpleCNN() if name == "cnn" else resnet18_tl()