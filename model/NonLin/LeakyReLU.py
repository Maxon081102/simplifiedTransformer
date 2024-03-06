import torch
import torch.nn as nn


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 1e-2):
        super().__init__()
        self.negative_slope = negative_slope
        self.register_buffer(
            "negative_slope",
            torch.tensor(negative_slope, requires_grad=False),
            persistent=False,
        )

    def forward(self, x):
        return torch.where(x >= 0.0, x, x * self.negative_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"
