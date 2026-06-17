import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Encoder(nn.Module):
    def __init__(self, input_channel=3, input_size=224, embedding_size=256):
        super().__init__()
        self.input_size = input_size
        self.backbone = resnet18(weights=None)
        if input_channel != 3:
            self.backbone.conv1 = nn.Conv2d(
                input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.project = nn.Sequential(
            nn.Conv2d(512, embedding_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.out_dim = embedding_size
        self.local_dim = embedding_size

    def forward_features(self, x):
        m = self.backbone
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        return self.project(x)

    def forward(self, x, return_spatial=False):
        feat_map = self.forward_features(x)
        if return_spatial:
            return feat_map
        emb = self.pool(feat_map).flatten(1)
        return self.bn(emb)
