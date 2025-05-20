import torch.nn as nn
from torch import Tensor

class LinearAdapter(nn.Module):
    """
    A simple linear adapter module that applies a linear transformation
    preserving input dimensionality.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)