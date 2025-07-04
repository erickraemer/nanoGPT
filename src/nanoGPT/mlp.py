from torch import Tensor
from torch.nn import Module, Linear, GELU, Dropout

from nanoGPT.gpt_config import GPTConfig


class MLP(Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc    = Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = GELU()
        self.c_proj  = Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x