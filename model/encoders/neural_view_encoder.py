import torch
from torch import nn


class NeuralViewEncoder(nn.Module):
    def __init__(self, front_view_size, bev_size, num_front_view=6):
        super(NeuralViewEncoder, self).__init__()
        self.bev_size = bev_size
        self.num_front_view = num_front_view
        front_view_dim = front_view_size[0] * front_view_size[1]
        bev_dim = bev_size[0] * bev_size[1]
        self.view_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(front_view_dim, bev_dim),
                nn.ReLU(),
                nn.Linear(bev_dim, bev_dim),
                nn.ReLU()
            )
            for _ in range(self.num_front_view)
        ])

    def forward(self, x):
        B, FVN, C, H, W = x.shape
        x = x.view(B, FVN, C, H * W)
        bev_embeddings = []
        for i in range(FVN):
            output = self.view_transforms[i](x[:, i])
            bev_embeddings.append(output.view(B, C, self.bev_size[0], self.bev_size[1]))
        return torch.stack(bev_embeddings, 1)
