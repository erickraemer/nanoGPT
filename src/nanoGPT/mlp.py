from torch import Tensor
from torch.nn import Module, Linear, GELU, Dropout

from nanoGPT.gpt_config import GPTConfig


class MLP(Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc    = Linear(cfg.model.embedding_size, 4 * cfg.model.embedding_size, bias=cfg.model.bias)
        self.gelu    = GELU()
        self.c_proj  = Linear(4 * cfg.model.embedding_size, cfg.model.embedding_size, bias=cfg.model.bias)
        self.dropout = Dropout(cfg.model.dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x