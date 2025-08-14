import logging
import math
from collections.abc import Callable
from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Linear, Dropout

from nanoGPT.gpt_config import GPTConfig

AttentionFunction: type = Callable[[Tensor, Tensor, Tensor], Tensor]

Logger = logging.getLogger(__file__)


class CausalSelfAttention(Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()

        # key, query, value projections for all heads, but in a batch
        self.c_attn = Linear(cfg.model.embedding_size, 3 * cfg.model.embedding_size, bias=cfg.model.bias)

        # output projection
        self.c_proj = Linear(cfg.model.embedding_size, cfg.model.embedding_size, bias=cfg.model.bias)

        # regularization
        self.attn_dropout = Dropout(cfg.model.dropout_rate)
        self.resid_dropout = Dropout(cfg.model.dropout_rate)
        self.cfg: Final[GPTConfig] = cfg

        # choose whether to use flash attention or manual attention
        self._attention_func: Final[AttentionFunction] = self._get_attention_func()
        self._create_causal_mask()

    def _create_causal_mask(self):
        """
        Create causal mask to ensure that attention is only applied to the left in
        the input sequence when using manual attention.
        """

        if self.cfg.flash:
            return

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self.cfg.data.block_size, self.cfg.data.block_size))
            .view(1, 1, self.cfg.data.block_size, self.cfg.data.block_size)
        )

    def _get_attention_func(self) -> AttentionFunction:
        """
        Returns the attention function based on whether flash attention is supported.
        """
        return self.flash_attention if self.cfg.flash else self.manual_attention

    def dynamic_head_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Dynamic head attention that considers only the active heads.
        """

        # consider only the active heads
        k = k[:, :self.cfg.model.active_heads, :, :]
        q = q[:, :self.cfg.model.active_heads, :, :]
        v = v[:, :self.cfg.model.active_heads, :, :]

        att = self._attention_func(q, k, v)

        # pad to full number of heads
        att = F.pad(att, (0, 0, 0, 0, 0, self.cfg.model.heads - self.cfg.model.active_heads), mode='constant',
                    value=0)

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
            dropout_p=self.cfg.model.dropout_rate if self.training else 0,
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
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        att = att @ value

        return att

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        hs: Final[int] = C // self.cfg.model.heads

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        x: Tensor = self.c_attn(x)  # (B, T, 3C)
        q, k, v = torch.split(x, self.cfg.model.embedding_size, dim=2)  # ((B, T, C), (B, T, C), (B, T, C))

        # (B, T, C) -> (B, T, H, C/H) with H*C/H = C
        k = k.view(B, T, self.cfg.model.heads, hs).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.cfg.model.heads, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.cfg.model.heads, hs).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = self.dynamic_head_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
