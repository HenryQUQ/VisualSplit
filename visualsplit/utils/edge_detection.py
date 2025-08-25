import torch
import torch.nn as nn


class SobelOperator(torch.nn.Module):
    def __init__(self):
        super(SobelOperator, self).__init__()
        self.sobel_x = nn.Parameter(
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(
                (1, 1, 3, 3)
            ),
            requires_grad=False,
        )

        self.sobel_y = nn.Parameter(
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(
                (1, 1, 3, 3)
            ),
            requires_grad=False,
        )

    def forward(self, x):
        edge_x = nn.functional.conv2d(x, self.sobel_x, padding=1)
        edge_y = nn.functional.conv2d(x, self.sobel_y, padding=1)
        edge = torch.sqrt(edge_x**2 + edge_y**2)
        return edge
