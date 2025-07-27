import math
from collections.abc import Callable
from typing import Final

import torch
from torch import Tensor
from torch.nn import Module, Linear, Dropout
from torch.nn.functional import softmax, scaled_dot_product_attention

from nanoGPT.gpt_config import GPTConfig

AttentionFunction: type = Final[Callable[[Tensor, Tensor, Tensor], Tensor]]

class CausalSelfAttention(Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = Dropout(config.dropout)
        self.resid_dropout = Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_active_heads = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.functional, 'scaled_dot_product_attention')
        self._attention_func: AttentionFunction = self.flash_attention if self.flash else self.manual_attention

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def flash_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        Efficient attention using Flash Attention CUDA kernels.
        """

        att = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )

        return att

    def manual_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        manual implementation of attention
        """

        T: int = query.shape[2]
        att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = softmax(att, dim=-1)
        att = self.attn_dropout(att)
        att = att @ value

        return att

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        x: Tensor = self.c_attn(x) # (B, T, 3C)
        q, k, v = torch.split(x, self.n_embd, dim=2) # ((B, T, C), (B, T, C), (B, T, C))

        # (B, T, C) -> (B, T, H, C/H) with H*C/H = C
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # consider only the active heads
        k = k[:, :self.n_active_heads, :, :]
        q = q[:, :self.n_active_heads, :, :]
        v = v[:, :self.n_active_heads, :, :]

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = self._attention_func(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y