from typing import Final

import torch
from torch import nn, Tensor
from torch.nn import Parameter
from torch.nn.functional import layer_norm


class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False
    """

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight: Final[Parameter] = Parameter(torch.ones(ndim)) # learnable scaling parameter gamma
        self.bias: Final[Parameter | None] = Parameter(torch.zeros(ndim)) if bias else None # learnable shift parameter beta

    def forward(self, x: Tensor) -> Tensor:

        x = layer_norm(
            input=x,
            normalized_shape=self.weight.shape,
            weight=self.weight,
            bias=self.bias,
            eps=1e-5
        )

        return x