import torch
import torch.nn as nn


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 1e-2) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.register_buffer(
            "negative_slope",
            torch.tensor(negative_slope, requires_grad=False),
            persistent=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.where(input >= 0.0, input, input * self.negative_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"
