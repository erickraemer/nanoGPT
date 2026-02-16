import math
from collections.abc import Callable
from typing import Final, Iterable

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
        # projection head mask: true = disabled, false = enabled
        self._projection_head_mask: Tensor = torch.full((self._total_heads,), False, dtype=torch.bool)
        self._embedding_size: int = cfg.model.embedding_size
        self._head_size: int = cfg.model.head_dimension
        self._head_dim: int = cfg.model.head_dimension * cfg.model.heads
        self._dropout_rate = cfg.model.dropout_rate

        # key, query, value projections for all heads, but in a batch
        self._c_attn: Final[Linear] = Linear(cfg.model.embedding_size, 3 * self._head_dim, bias=cfg.model.bias)

        # output projection
        self._c_proj: Final[Linear] = Linear(self._head_dim, cfg.model.embedding_size, bias=cfg.model.bias)
        self._c_proj.weight.register_hook(self._mask_inactive_projection_gradients)
        self._last_c_proj_grad: Tensor | None = None

        # regularization
        # self._attn_dropout = Dropout(cfg.model.dropout_rate)
        # self._resid_dropout = Dropout(cfg.model.dropout_rate)

        # choose whether to use flash attention or manual attention
        self._attention_func: Final[AttentionFunction] = self._get_attention_func(cfg)

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def head_size(self) -> int:
        return self._head_size

    @property
    def total_heads(self) -> int:
        return self._total_heads

    def set_disabled_heads(self, disabled_heads: Iterable[int]):
        """
        Disables attention heads in the projection matrix. All other heads will be enabled.
        :param disabled_heads: an iterable of heads to disable starting at zero.
        """
        disabled_heads = set(disabled_heads)
        active_heads = (i for i in range(self.total_heads) if i not in disabled_heads)
        self.set_active_heads(active_heads)

    def set_active_heads(self, active_heads: Iterable[int]):
        """
        Enables attention heads in the projection matrix. All other heads will be disabled.
        :param active_heads: an iterable of heads to enable starting at zero.
        """

        mask = torch.full((self._total_heads,), True, dtype=torch.bool)
        for i in active_heads:
            mask[i] = False

        self._projection_head_mask = mask

        # zero inactive heads in the projection matrix
        p_heads = self.get_projection_head_view(self._c_proj.weight.data)
        p_heads[self._projection_head_mask] = 0

    def _get_attention_func(self, cfg: GPTConfig) -> AttentionFunction:
        """
        Returns the attention function based on whether flash attention is supported.
        """
        if cfg.flash:
            return self.flash_attention

        self.register_buffer(
            "_bias",
            torch.tril(torch.ones(cfg.data.block_size, cfg.data.block_size))
            .view(1, 1, cfg.data.block_size, cfg.data.block_size)
        )

        return self.manual_attention

    def _mask_inactive_projection_gradients(self, grad: Tensor) -> Tensor:
        """Hook to zero out gradients for inactive heads in the projection matrix"""

        masked_grad = self.get_projection_head_view(grad)
        self._last_c_proj_grad = masked_grad.clone().detach()
        masked_grad[self._projection_head_mask] = 0
        return grad

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

    def get_projection_head_view(self, weight: Tensor) -> Tensor:
        """
        Returns a view of the projection heads.
        """

        weight = weight.data.view(self._embedding_size, self._total_heads, self._head_size)
        weight = weight.transpose(0, 1)

        assert weight.shape == (self._total_heads, self._embedding_size, self._head_size)
        return weight

    def get_attention_head_view(self, weight: Tensor) -> Tensor:
        """
        Returns a view of the attention weight with the
        shape (total_heads, head_size, 3 (K, Q, V), embed_size).
        """

        # attention weight shape is (3xEmbedding Size, Embedding Size)
        heads = weight.view(self._total_heads, self._head_size, 3, self.embedding_size)
        # heads = heads.transpose(0, 1)

        # assert heads.shape == (self._total_heads, self._head_size, self._embedding_size)

        return heads

    def get_attention_head_gradients(self) -> Tensor | None:
        """
        Returns a view of the attention gradients with the shape (total_heads, 3 (K, Q, V), head_size, embed_size).
        """

        heads = self.get_attention_head_view(self._c_attn.weight.grad).detach()

        return heads

    def get_projection_head_gradients(self) -> Tensor | None:
        """
        Returns a view of the projection head gradients with the shape (total_heads, embed_size, head_size).
        """

        norms = self._last_c_proj_grad

        assert norms.shape == (self._total_heads, self.embedding_size, self._head_size)

        return norms

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # multiply by attention weight to get the shape (batch size, sequence length, 3 x total heads x head size)
        x: Tensor = self._c_attn(x)

        # create a view of shape (batch size, sequence length, total heads, head size, 3)
        # where: 3 x total heads x head size = 3 x head dim
        attn = x.view(B, T, self._total_heads, self._head_size, 3)

        # move dimension to get (batch size, total heads, sequence length, head size, 3)
        attn = attn.movedim(1, 2)

        # unbind to get q, k, v of shape (batch size, total heads, sequence length, head size)
        q, k, v = attn.unbind(dim=-1)

        # causal self-attention -> (batch size, total heads, sequence length, head size)
        y = self._attention_func(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, self._head_dim)  # re-assemble all head outputs side by side

        # output projection
        # y = self._resid_dropout(self._c_proj(y))
        y = self._c_proj(y)
        return y