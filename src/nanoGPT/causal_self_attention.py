import math
from collections.abc import Callable
from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Linear, Dropout

from nanoGPT.gpt_config import GPTConfig

AttentionFunction: type = Callable[[Tensor, Tensor, Tensor], Tensor]


class CausalSelfAttention(Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()

        # hyperparameter
        self._total_heads: int = cfg.model.heads
        self._active_heads: int = cfg.model.heads
        self._embedding_size: int = cfg.model.embedding_size
        self._head_size: int = self._embedding_size // self._total_heads
        self._active_embedding_size: int = self._head_size * self._active_heads
        self._dropout_rate = cfg.model.dropout_rate

        # key, query, value projections for all heads, but in a batch
        self._c_attn: Final[Linear] = Linear(cfg.model.embedding_size, 3 * cfg.model.embedding_size, bias=cfg.model.bias)

        # output projection
        self._c_proj: Final[Linear] = Linear(cfg.model.embedding_size, cfg.model.embedding_size, bias=cfg.model.bias)

        # regularization
        self._attn_dropout = Dropout(cfg.model.dropout_rate)
        self._resid_dropout = Dropout(cfg.model.dropout_rate)

        # choose whether to use flash attention or manual attention
        self._attention_func: Final[AttentionFunction] = self._get_attention_func(cfg)
        self._create_causal_mask(cfg)

    def _create_causal_mask(self, cfg: GPTConfig):
        """
        Create causal mask to ensure that attention is only applied to the left in
        the input sequence when using manual attention.
        """

        if cfg.flash:
            return

        self.register_buffer(
            "_bias",
            torch.tril(torch.ones(cfg.data.block_size, cfg.data.block_size))
            .view(1, 1, cfg.data.block_size, cfg.data.block_size)
        )

    def set_active_heads(self, active_heads: int):
        assert 0 < active_heads <= self._total_heads

        # recalculate sizes
        self._active_heads: int = active_heads
        self._active_embedding_size: int = self._head_size * active_heads

        # zero inactive heads in the projection matrix
        p_heads = self._c_proj.weight.data.view(self._total_heads, self._head_size, self._embedding_size)
        p_heads[active_heads:, :, :] = 0.0

    def _get_attention_func(self, cfg: GPTConfig) -> AttentionFunction:
        """
        Returns the attention function based on whether flash attention is supported.
        """
        return self.flash_attention if cfg.flash else self.manual_attention

    def dynamic_head_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Dynamic head attention that considers only the active heads.
        """

        # zero inactive heads
        # mask = torch.zeros_like(k, requires_grad=False)
        # mask[:, :self._active_heads, :, :] = 1.0
        # k = k * mask
        # q = q * mask
        # v = v * mask

        att = self._attention_func(q, k, v)

        return att

    def flash_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        Efficient attention using Flash Attention CUDA kernels.
        """

        att = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self._dropout_rate if self.training else 0,
            is_causal=True
        )

        return att

    def manual_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        manual implementation of attention
        """

        T: int = query.shape[2]
        att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        att = att.masked_fill(self._bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self._attn_dropout(att)
        att = att @ value

        return att

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        x: Tensor = self._c_attn(x)  # (B, T, 3C)
        q, k, v = torch.split(x, self._embedding_size, dim=2)  # ((B, T, C), (B, T, C), (B, T, C))

        # (B, T, C) -> (B, T, H, C/H) with H*C/H = C
        k = k.view(B, T, self._total_heads, self._head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self._total_heads, self._head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self._total_heads, self._head_size).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = self.dynamic_head_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self._resid_dropout(self._c_proj(y))
        return y