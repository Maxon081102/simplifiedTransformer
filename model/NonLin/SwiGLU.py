import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linearA = nn.Linear(size, size)
        self.linearB = nn.Linear(size, size)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)
        # self.beta = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        x_b = self.linearB(x)
        swish = x_b * torch.sigmoid(self.beta * x_b)
        return swish * self.linearA(x)
