from torch import nn as nn
from torchvision.models import resnet18


class BevEncoder(nn.Module):
    def __init__(self, in_channel):
        super(BevEncoder, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=7,
            stride=2, padding=3,
            bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x2 = self.layer3(x)

        return x2, x1
