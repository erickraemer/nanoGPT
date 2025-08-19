import functools
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
        self._active_heads: int = cfg.model.active_heads
        self._embedding_size: int = cfg.model.embedding_size
        self._head_size: int = self._embedding_size // self._total_heads
        self._active_embedding_size: int = self._head_size * self._active_heads

        # key, query, value projections for all heads, but in a batch
        self._c_attn: Final[Linear] = Linear(cfg.model.embedding_size, 3 * cfg.model.embedding_size, bias=cfg.model.bias)

        # output projection
        self._c_proj: Final[Linear] = Linear(cfg.model.embedding_size, cfg.model.embedding_size, bias=cfg.model.bias)

        # regularization
        self._attn_dropout = Dropout(cfg.model.dropout_rate)
        self._resid_dropout = Dropout(cfg.model.dropout_rate)
        self._cfg: Final[GPTConfig] = cfg

        # choose whether to use flash attention or manual attention
        self._attention_func: Final[AttentionFunction] = self._get_attention_func()
        self._create_causal_mask()

    def _create_causal_mask(self):
        """
        Create causal mask to ensure that attention is only applied to the left in
        the input sequence when using manual attention.
        """

        if self._cfg.flash:
            return

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self._cfg.data.block_size, self._cfg.data.block_size))
            .view(1, 1, self._cfg.data.block_size, self._cfg.data.block_size)
        )

    def _get_attention_func(self) -> AttentionFunction:
        """
        Returns the attention function based on whether flash attention is supported.
        """

        return self.flash_attention if self._cfg.flash else self.manual_attention

    def _get_c_attn_weight_view(self) -> Tensor:
        """
        Returns a view for c_attn's weight based on the current active heads
        """

        weight = self._c_attn.weight.view(self._total_heads, 3, self._head_size, self._embedding_size)
        weight = weight[:self._active_heads, ...]
        weight = weight.view(3 * self._active_embedding_size, self._embedding_size)

        # check that the new tensor is a reference and not a copy
        assert self._c_attn.weight.data_ptr() == weight.data_ptr()

        return weight

    def _get_c_attn_bias_view(self) -> Tensor | None:
        """
        Returns a view for c_attn's bias based on the current active heads
        """

        if self._c_attn.bias is None:
            return None

        bias = self._c_attn.bias.view(self._total_heads, 3, self._head_size)
        bias = bias[:self._active_heads, ...]
        bias = bias.view(3 * self._active_embedding_size)

        # check that the new tensor is a reference and not a copy
        assert self._c_attn.bias.data_ptr() == bias.data_ptr()

        return bias

    def _get_c_proj_weight_view(self) -> Tensor:
        """
        Returns a view for c_proj's weight based on the current active heads
        """

        weight = self._c_proj.weight[:, :self._active_embedding_size]

        # check that the new tensor is a reference and not a copy
        assert self._c_proj.weight.data_ptr() == weight.data_ptr()

        return weight

    def _c_attn_forward(self, x: Tensor) -> Tensor:

        c_attn = F.linear(
            input=x,
            weight=self._get_c_attn_weight_view(),
            bias=self._get_c_attn_bias_view()
        )
        return c_attn

    def _c_proj_forward(self, x: Tensor) -> Tensor:

        c_proj = F.linear(
            input=x,
            weight=self._get_c_proj_weight_view(),
            bias=self._c_proj.bias
        )
        return c_proj

    def set_active_heads(self, active_heads: int):
        assert 0 < active_heads <= self._total_heads

        # recalculate sizes
        self._active_heads: int = active_heads
        self._head_size: int = self._embedding_size // self._total_heads
        self._active_embedding_size: int = self._head_size * active_heads


    def flash_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        Efficient attention using Flash Attention CUDA kernels.
        """

        att = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self._cfg.model.dropout_rate if self.training else 0,
            is_causal=True
        )

        return att

    def manual_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        manual implementation of attention
        """

        sl: int = query.shape[2]  # sequence length
        att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        att = att.masked_fill(self.bias[:, :, :sl, :sl] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self._attn_dropout(att)
        att = att @ value

        return att

    def forward(self, x: Tensor) -> Tensor:

        # Terminology
        # ---------------------------
        # batch size            (bs)
        # sequence length       (sl)
        # embedding size        (es)
        # active embedding size (aes)
        # active heads          (ah)
        # head size             (hs)

        bs, sl, _ = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim:
        # -> (bs, sl, 3 * aes)
        x = self._c_attn_forward(x)

        q: Tensor
        k: Tensor
        v: Tensor
        # (bs, sl, 3 * aes) -> 3 * (bs, sl, aes)
        q, k, v = torch.split(x, self._active_embedding_size, dim=2)

        # (bs, sl, aes) -> (bs, sl, ah, hs)
        k = k.view(bs, sl, self._active_heads, self._head_size).transpose(1, 2)
        q = q.view(bs, sl, self._active_heads, self._head_size).transpose(1, 2)
        v = v.view(bs, sl, self._active_heads, self._head_size).transpose(1, 2)

        # causal self-attention; Self-attend: -> (bs, ah, sl, hs)
        x: Tensor = self._attention_func(q, k, v)
        # re-assemble all head outputs side by side: -> (bs, sl, aes)
        x = x.transpose(1, 2).contiguous().view(bs, sl, self._active_embedding_size)

        # output projection: -> (bs, sl, es)
        x = self._c_proj_forward(x)
        x = self._resid_dropout(x)
        return x
