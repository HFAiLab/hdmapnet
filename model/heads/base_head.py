from torch import nn
from model.utils.up_model import Up

class BaseHead(nn.Module):
    def __init__(self, out_dim):
        super(BaseHead, self).__init__()

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, kernel_size=1, padding=0),
        )

    def forward(self, x2, x1):
        x = self.up1(x2, x1)
        x = self.up2(x)
        return x
