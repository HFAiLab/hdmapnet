import torch
from torch import nn


class Up(nn.Module):
    def __init__(self, in_dims, out_dims, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=True
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dims, out_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)
