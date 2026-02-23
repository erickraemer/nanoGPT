import math

import torch
from torch import Tensor
import torch.nn.functional as F
import pytest



def test_weights():
    embedding_size: int = 16
    batch_size: int = 1
    sequence_length: int = 16
    total_heads: int = 4
    active_heads: int = 2
    block_size: int = 16

    assert embedding_size % total_heads == 0
    head_size: int = embedding_size // total_heads
    active_embedding_size = head_size *  active_heads

    weight = torch.empty((embedding_size * 3, embedding_size), dtype=torch.float, requires_grad=False)


    weight = weight.view(total_heads, 3, head_size, embedding_size)
    for i in range(3):
        for h in range(total_heads):
            weight[h, i, ...] = h + i * 0.3

    weight = weight.view(3 * embedding_size, embedding_size)
    weight = weight[:3 * active_embedding_size, :]

    x = torch.ones((batch_size, sequence_length, embedding_size), dtype=torch.float, requires_grad=False)

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    x: Tensor = F.linear(x, weight, None) # x = (bs, sl, 3 * aes) = (bs, sl, 3 * ah * hs)
    x = x.view(batch_size, sequence_length, active_heads, 3 * head_size).transpose(1, 2)
    q, k, v = x.split(head_size, dim=-1)

    # t = torch.split(x, 3 * head_size, dim=2)
    # x = torch.stack(t, dim=1)
    # q, k, v = torch.split(x, head_size, dim=3)
    # x = x.view(batch_size, sequence_length, 3, active_heads, head_size)
    #
    # # Split into q, k, v and rearrange dimensions
    # q, k, v = x.unbind(dim=2)  # Each has shape (batch_size, sequence_length, active_heads, head_size)
    #
    # # Transpose to put heads as batch dimension

    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)

    # bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
    # sl: int = q.shape[2]  # sequence length
    # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    # att = att.masked_fill(bias[:, :, :sl, :sl] == 0, float('-inf'))
    # att = F.softmax(att, dim=-1)
    # y = att @ v

    y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, active_embedding_size)

    print()

