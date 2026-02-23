import pytest
import torch

from nanoGPT.causal_self_attention import CausalSelfAttention
from nanoGPT.gpt_config import GPTConfig

@torch.no_grad()
def test_heads():
    cfg = GPTConfig()
    n_heads = cfg.model.heads = 8
    d_head = cfg.model.head_dimension = 64
    d_model = cfg.model.embedding_size = 512
    sl = cfg.data.block_size = 512
    bs = cfg.data.batch_size = 1

    for h in range(n_heads):

        attn = CausalSelfAttention(cfg)
        v = attn.get_attention_head_view(attn._c_attn.weight)
        v[h] = 0

        def attention(query, key, value):
            nonlocal h, attn
            # (batch size, total heads, sequence length, head size)
            y = attn.flash_attention(query, key, value)
            y_ = y.transpose(0, 1)

            # test that zeroed heads result in zero attention
            assert torch.sum(torch.abs(y_[h])) == 0, f"head={h}"

            # all non zeroed heads must produce non zero attention
            for i in range(n_heads):
                if i == h:
                    continue
                assert torch.sum(torch.abs(y_[i])) != 0, f"head={h}"

            return y

        attn._attention_func = attention

        x = torch.normal(0.5, std=1.0, size=(bs, sl, d_model))
        y = attn.forward(x)

@torch.no_grad()
def test_query():
    cfg = GPTConfig()
    n_heads = cfg.model.heads = 8
    d_head = cfg.model.head_dimension = 64
    d_model = cfg.model.embedding_size = 512
    sl = cfg.data.block_size = 512
    bs = cfg.data.batch_size = 1

    for h in range(n_heads):

        attn = CausalSelfAttention(cfg)
        a = attn.get_attention_head_view(attn._c_attn.weight)
        q,k,v = torch.unbind(a, dim=1)
        q[h] = 0 # set value to zero

        def attention(query, key, value):
            nonlocal h, attn
            # (batch size, total heads, sequence length, head size)
            y = attn.flash_attention(query, key, value)
            y_ = y.transpose(0, 1)

            # all non zeroed value heads must produce non zero attention
            for i in range(n_heads):
                assert torch.sum(torch.abs(y_[i])) != 0, f"head={h}"

            return y

        attn._attention_func = attention

        x = torch.normal(0.5, std=1.0, size=(bs, sl, d_model))
        y = attn.forward(x)

@torch.no_grad()
def test_key():
    cfg = GPTConfig()
    n_heads = cfg.model.heads = 8
    d_head = cfg.model.head_dimension = 64
    d_model = cfg.model.embedding_size = 512
    sl = cfg.data.block_size = 512
    bs = cfg.data.batch_size = 1

    for h in range(n_heads):

        attn = CausalSelfAttention(cfg)
        a = attn.get_attention_head_view(attn._c_attn.weight)
        q,k,v = torch.unbind(a, dim=1)
        k[h] = 0 # set value to zero

        def attention(query, key, value):
            nonlocal h, attn
            # (batch size, total heads, sequence length, head size)
            y = attn.flash_attention(query, key, value)
            y_ = y.transpose(0, 1)

            # all non zeroed value heads must produce non zero attention
            for i in range(n_heads):
                assert torch.sum(torch.abs(y_[i])) != 0, f"head={h}"

            return y

        attn._attention_func = attention

        x = torch.normal(0.5, std=1.0, size=(bs, sl, d_model))
        y = attn.forward(x)

@torch.no_grad()
def test_values():
    cfg = GPTConfig()
    n_heads = cfg.model.heads = 8
    d_head = cfg.model.head_dimension = 64
    d_model = cfg.model.embedding_size = 512
    sl = cfg.data.block_size = 512
    bs = cfg.data.batch_size = 1

    for h in range(n_heads):

        attn = CausalSelfAttention(cfg)
        a = attn.get_attention_head_view(attn._c_attn.weight)
        q,k,v = torch.unbind(a, dim=1)
        v[h] = 0 # set value to zero

        def attention(query, key, value):
            nonlocal h, attn
            # (batch size, total heads, sequence length, head size)
            y = attn.flash_attention(query, key, value)
            y_ = y.transpose(0, 1)

            # test that zeroed value head result in zero attention
            assert torch.sum(torch.abs(y_[h])) == 0, f"head={h}"

            # all non zeroed value heads must produce non zero attention
            for i in range(n_heads):
                if i == h:
                    continue
                assert torch.sum(torch.abs(y_[i])) != 0, f"head={h}"

            return y

        attn._attention_func = attention

        x = torch.normal(0.5, std=1.0, size=(bs, sl, d_model))
        y = attn.forward(x)

@torch.no_grad()
def test_projection():
    cfg = GPTConfig()
    n_heads = cfg.model.heads = 8
    d_head = cfg.model.head_dimension = 64
    d_model = cfg.model.embedding_size = 512
    sl = cfg.data.block_size = 512
    bs = cfg.data.batch_size = 1

    attn = CausalSelfAttention(cfg)
    v = attn.get_attention_head_view(attn._c_attn.weight)
    v[:] = 0

    x = torch.normal(0.01, std=0.02, size=(bs, sl, d_model))
    y = attn.forward(x)

    # test if output is zero if all heads are zeroed
    assert torch.sum(torch.abs(y)) == 0

@torch.no_grad()
def test_projection2():
    cfg = GPTConfig()
    n_heads = cfg.model.heads = 8
    d_head = cfg.model.head_dimension = 64
    d_model = cfg.model.embedding_size = 512
    sl = cfg.data.block_size = 512
    bs = cfg.data.batch_size = 1


    for i in range(n_heads):
        attn = CausalSelfAttention(cfg)
        a = attn.get_attention_head_view(attn._c_attn.weight)
        # set all attention heads which are not i to zero
        a[[i != k for k in range(n_heads)]] = 0

        # set the projection of head i to zero
        p = attn.get_projection_head_view(attn._c_proj.weight)
        p[i] = 0

        x = torch.normal(0.5, std=1.0, size=(bs, sl, d_model))
        y = attn.forward(x)

        # output needs to be zero
        assert torch.sum(torch.abs(y)) == 0

@torch.no_grad()
def test_projection3():
    cfg = GPTConfig()
    n_heads = cfg.model.heads = 8
    d_head = cfg.model.head_dimension = 64
    d_model = cfg.model.embedding_size = 512
    sl = cfg.data.block_size = 512
    bs = cfg.data.batch_size = 1


    for i in range(n_heads):
        attn = CausalSelfAttention(cfg)
        a = attn.get_attention_head_view(attn._c_attn.weight)
        # set all attention head i to zero
        a[i] = 0

        # set the projection of all heads which are not i to zero
        p = attn.get_projection_head_view(attn._c_proj.weight)
        p[[i != k for k in range(n_heads)]] = 0

        x = torch.normal(0.5, std=1.0, size=(bs, sl, d_model))
        y = attn.forward(x)

        # output needs to be zero
        assert torch.sum(torch.abs(y)) == 0