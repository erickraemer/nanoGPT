import pytest
import torch

from nanoGPT.causal_self_attention import CausalSelfAttention
from nanoGPT.gpt_config import GPTConfig


def _make_cfg():
    cfg = GPTConfig()
    cfg.model.heads = 8
    cfg.model.head_dimension = 64
    cfg.model.embedding_size = 768
    cfg.data.block_size = 512
    cfg.data.batch_size = 2
    return cfg


def _make_attn(cfg=None):
    cfg = cfg or _make_cfg()
    return CausalSelfAttention(cfg), cfg


# ── 1. Views, not copies ────────────────────────────────────────────

@torch.no_grad()
def test_attention_head_view_is_view():
    attn, cfg = _make_attn()
    view = attn.get_attention_head_view(attn._c_attn.weight)
    original_val = attn._c_attn.weight[0, 0].item()

    # mutate through the view
    view[0, 0, 0, 0] = 999.0
    assert attn._c_attn.weight.view(
        cfg.model.heads, 3, cfg.model.head_dimension, cfg.model.embedding_size
    )[0, 0, 0, 0].item() == 999.0

    # mutate through the raw weight, observe through the view
    attn._c_attn.weight.data.view_as(attn._c_attn.weight)[0, 0] = -1.0
    assert view[0, 0, 0, 0].item() == -1.0


@torch.no_grad()
def test_projection_head_view_is_view():
    attn, cfg = _make_attn()
    view = attn.get_projection_head_view(attn._c_proj.weight)

    view[0, 0, 0] = 777.0
    raw = attn._c_proj.weight.data.view(
        cfg.model.embedding_size, cfg.model.heads, cfg.model.head_dimension
    ).transpose(0, 1)
    assert raw[0, 0, 0].item() == 777.0


# ── 2. Shape ─────────────────────────────────────────────────────────

@torch.no_grad()
def test_attention_head_view_shape():
    attn, cfg = _make_attn()
    view = attn.get_attention_head_view(attn._c_attn.weight)
    n_heads = cfg.model.heads
    d_head = cfg.model.head_dimension
    d_model = cfg.model.embedding_size
    assert view.shape == (n_heads, 3, d_head, d_model)


@torch.no_grad()
def test_projection_head_view_shape():
    attn, cfg = _make_attn()
    view = attn.get_projection_head_view(attn._c_proj.weight)
    n_heads = cfg.model.heads
    d_head = cfg.model.head_dimension
    d_model = cfg.model.embedding_size
    assert view.shape == (n_heads, d_model, d_head)


# ── 3. Full coverage (no gaps, no overlaps) ──────────────────────────

@torch.no_grad()
def test_attention_view_covers_all_elements():
    attn, cfg = _make_attn()
    n_heads = cfg.model.heads
    view = attn.get_attention_head_view(attn._c_attn.weight)

    # assign each head a distinct constant
    for h in range(n_heads):
        view[h] = float(h + 1)

    # every element of the raw weight must be one of those constants
    raw = attn._c_attn.weight.data
    unique = torch.unique(raw)
    expected = torch.arange(1, n_heads + 1, dtype=raw.dtype)
    assert torch.equal(unique.sort().values, expected.sort().values)

    # total element count must match
    assert raw.numel() == view.numel()


@torch.no_grad()
def test_projection_view_covers_all_elements():
    attn, cfg = _make_attn()
    n_heads = cfg.model.heads
    view = attn.get_projection_head_view(attn._c_proj.weight)

    for h in range(n_heads):
        view[h] = float(h + 1)

    raw = attn._c_proj.weight.data
    unique = torch.unique(raw)
    expected = torch.arange(1, n_heads + 1, dtype=raw.dtype)
    assert torch.equal(unique.sort().values, expected.sort().values)
    assert raw.numel() == view.numel()


# ── 4. Head isolation ────────────────────────────────────────────────

@torch.no_grad()
def test_attention_head_isolation():
    attn, cfg = _make_attn()
    n_heads = cfg.model.heads
    view = attn.get_attention_head_view(attn._c_attn.weight)

    # snapshot all heads
    originals = [view[h].clone() for h in range(n_heads)]

    # mutate head 0
    view[0] = 0.0

    # all other heads unchanged
    for h in range(1, n_heads):
        assert torch.equal(view[h], originals[h]), f"head {h} was corrupted"


@torch.no_grad()
def test_projection_head_isolation():
    attn, cfg = _make_attn()
    n_heads = cfg.model.heads
    view = attn.get_projection_head_view(attn._c_proj.weight)

    originals = [view[h].clone() for h in range(n_heads)]
    view[0] = 0.0

    for h in range(1, n_heads):
        assert torch.equal(view[h], originals[h]), f"head {h} was corrupted"


# ── 5. Q/K/V ordering consistency with forward ──────────────────────

@torch.no_grad()
def test_qkv_ordering_matches_forward():
    """
    Set Q weights to 1, K weights to 2, V weights to 3 via the view,
    then verify that forward's internal decomposition assigns them correctly.
    """
    cfg = _make_cfg()
    cfg.model.heads = 2
    cfg.model.head_dimension = 4
    cfg.model.embedding_size = 8
    cfg.data.block_size = 4
    attn = CausalSelfAttention(cfg)

    n_heads = cfg.model.heads
    d_head = cfg.model.head_dimension
    d_model = cfg.model.embedding_size

    view = attn.get_attention_head_view(attn._c_attn.weight)
    # zero bias to isolate weight contributions
    if attn._c_attn.bias is not None:
        attn._c_attn.bias.data.zero_()

    # assign distinct constants per component
    for h in range(n_heads):
        view[h, 0] = 1.0  # Q
        view[h, 1] = 2.0  # K
        view[h, 2] = 3.0  # V

    captured = {}

    def spy_attention(query, key, value):
        captured['q'] = query.clone()
        captured['k'] = key.clone()
        captured['v'] = value.clone()
        return attn.flash_attention(query, key, value)

    attn._attention_func = spy_attention

    x = torch.ones(1, cfg.data.block_size, d_model)
    attn.forward(x)

    # Q should reflect weight=1 applied to x, K weight=2, V weight=3
    # Since x is all-ones and bias is zero:
    #   Q_h = x @ W_q_h^T where W_q_h is all 1s -> each element = d_model * 1
    #   K_h = x @ W_k_h^T where W_k_h is all 2s -> each element = d_model * 2
    #   V_h = x @ W_v_h^T where W_v_h is all 3s -> each element = d_model * 3
    expected_q_val = d_model * 1.0
    expected_k_val = d_model * 2.0
    expected_v_val = d_model * 3.0

    assert torch.allclose(captured['q'], torch.full_like(captured['q'], expected_q_val)), \
        f"Q mismatch: expected {expected_q_val}, got unique vals {torch.unique(captured['q'])}"
    assert torch.allclose(captured['k'], torch.full_like(captured['k'], expected_k_val)), \
        f"K mismatch: expected {expected_k_val}, got unique vals {torch.unique(captured['k'])}"
    assert torch.allclose(captured['v'], torch.full_like(captured['v'], expected_v_val)), \
        f"V mismatch: expected {expected_v_val}, got unique vals {torch.unique(captured['v'])}"


# ── 6. Round-trip: view slice matches actual linear projection ───────

@torch.no_grad()
def test_attention_view_roundtrip():
    """
    For each head h, manually compute x @ W_h^T for Q/K/V using the view slice
    and compare against the corresponding portion of _c_attn(x) reshaped the
    same way forward() does.
    """
    cfg = _make_cfg()
    attn = CausalSelfAttention(cfg)

    n_heads = cfg.model.heads
    d_head = cfg.model.head_dimension
    d_model = cfg.model.embedding_size
    sl = cfg.data.block_size

    x = torch.randn(1, sl, d_model)
    full_out = attn._c_attn(x)  # (1, sl, 3 * head_dim)

    # reshape the same way forward does
    decomposed = full_out.view(1, sl, n_heads, 3, d_head).transpose(1, 2)
    # (1, n_heads, sl, 3, d_head)

    view = attn.get_attention_head_view(attn._c_attn.weight)
    bias = attn._c_attn.bias

    for h in range(n_heads):
        for comp in range(3):  # Q=0, K=1, V=2
            W_hc = view[h, comp]  # (d_head, d_model)
            manual = x @ W_hc.T  # (1, sl, d_head)

            if bias is not None:
                # extract the matching bias slice
                bias_view = bias.view(n_heads, 3, d_head)
                manual = manual + bias_view[h, comp]

            actual = decomposed[0, h, :, comp, :]  # (sl, d_head)
            assert torch.allclose(manual.squeeze(0), actual, atol=1e-5), \
                f"Mismatch at head={h}, comp={comp}"


@torch.no_grad()
def test_projection_view_roundtrip():
    """
    For each head, verify that _c_proj applied to a one-hot head pattern
    matches manual matmul with the view slice.
    """
    cfg = _make_cfg()
    attn = CausalSelfAttention(cfg)

    n_heads = cfg.model.heads
    d_head = cfg.model.head_dimension
    d_model = cfg.model.embedding_size

    # zero bias to isolate weight
    if attn._c_proj.bias is not None:
        attn._c_proj.bias.data.zero_()

    view = attn.get_projection_head_view(attn._c_proj.weight)  # (n_heads, d_model, d_head)

    for h in range(n_heads):
        # construct input that is nonzero only in head h's slot
        head_input = torch.zeros(1, 1, n_heads * d_head)
        head_val = torch.randn(d_head)
        head_input[0, 0, h * d_head:(h + 1) * d_head] = head_val

        actual = attn._c_proj(head_input).squeeze()  # (d_model,)
        manual = view[h] @ head_val  # (d_model,)

        assert torch.allclose(actual, manual, atol=1e-5), \
            f"Projection roundtrip mismatch at head={h}"


# ── 7. Composition: attention view + projection view agree on head identity

@torch.no_grad()
def test_attention_projection_composition():
    """
    Activate only head i (zero all others in attention), set projection for head i
    to a known matrix, and verify the end-to-end output matches manual computation.
    """
    cfg = _make_cfg()
    cfg.model.heads = 4
    cfg.model.head_dimension = 16
    cfg.model.embedding_size = 64
    cfg.data.block_size = 8
    attn = CausalSelfAttention(cfg)

    n_heads = cfg.model.heads
    d_head = cfg.model.head_dimension
    d_model = cfg.model.embedding_size
    sl = cfg.data.block_size

    # zero all biases
    if attn._c_attn.bias is not None:
        attn._c_attn.bias.data.zero_()
    if attn._c_proj.bias is not None:
        attn._c_proj.bias.data.zero_()

    target_head = 1

    # zero all attention heads except target
    a_view = attn.get_attention_head_view(attn._c_attn.weight)
    for h in range(n_heads):
        if h != target_head:
            a_view[h] = 0.0

    # set a known projection for the target head, zero others
    p_view = attn.get_projection_head_view(attn._c_proj.weight)
    for h in range(n_heads):
        if h != target_head:
            p_view[h] = 0.0

    # record the target head's projection matrix
    proj_matrix = p_view[target_head].clone()  # (d_model, d_head)

    # capture the attention output for the target head
    captured = {}

    def spy(query, key, value):
        y = attn.flash_attention(query, key, value)
        captured['y'] = y.clone()
        return y

    attn._attention_func = spy

    x = torch.randn(1, sl, d_model)
    output = attn.forward(x)

    # the attention output for head target_head
    attn_out_h = captured['y'][:, target_head, :, :]  # (1, sl, d_head)

    # manual projection
    expected = attn_out_h @ proj_matrix.T  # (1, sl, d_model)

    assert torch.allclose(output, expected, atol=1e-4), \
        "Composition mismatch: attention view and projection view disagree on head identity"