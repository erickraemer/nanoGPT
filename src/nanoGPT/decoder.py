from torch import Tensor
from torch.nn import Module

from nanoGPT.causal_self_attention import CausalSelfAttention
from nanoGPT.gpt_config import GPTConfig
from nanoGPT.layer_norm import LayerNorm
from nanoGPT.mlp import MLP


class Block(Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x