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
        ah: Final[int] = self.cfg.model.active_heads
        nh: Final[int] = self.cfg.model.heads

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        C_hat: Final[int] = ah * hs
        wa = self.c_attn.weight.reshape(3, nh, hs, C)[:, :ah, :, :].reshape(3*C_hat, C)
        ba = None if self.c_attn.bias is None else self.c_attn.bias.reshape(3, nh, hs)[:, :ah, :].reshape(3*C_hat)
        x: Tensor = F.linear(x, wa, ba)
        q, k, v = torch.split(x, C_hat, dim=2)  # ((B, T, C_hat), (B, T, C_hat), (B, T, C_hat))

        # (B, T, C_hat) -> (B, T, ah, hs)
        k = k.view(B, T, ah, hs).transpose(1, 2)  # (B, ah, T, hs)
        q = q.view(B, T, ah, hs).transpose(1, 2)  # (B, ah, T, hs)
        v = v.view(B, T, ah, hs).transpose(1, 2)  # (B, ah, T, hs)

        # causal self-attention; Self-attend: (B, ah, T, hs) x (B, ah, hs, T) -> (B, ah, T, hs)
        y = self._attention_func(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C_hat)  # re-assemble all head outputs side by side

        # output projection
        wp = self.c_proj.weight[:, :C_hat]
        y = F.linear(y, wp, self.c_proj.bias)
        y = self.resid_dropout(y)
        return y
