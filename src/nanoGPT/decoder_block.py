from torch import Tensor
from torch.nn import Module

from nanoGPT.causal_self_attention import CausalSelfAttention
from nanoGPT.gpt_config import GPTConfig
from nanoGPT.layer_norm import LayerNorm
from nanoGPT.mlp import MLP


class DecoderBlock(Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.model.embedding_size, bias=cfg.model.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.model.embedding_size, bias=cfg.model.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:

        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x