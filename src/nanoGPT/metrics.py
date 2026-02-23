import torch

from nanoGPT.causal_self_attention import CausalSelfAttention


@torch.no_grad()
def get_attention_head_norms(attn: CausalSelfAttention, optimizer: torch.optim.Optimizer, layer: int) -> dict[str, float]:
    norms: dict[str, float] = {}

    attention_heads = attn.get_attention_head_gradients()
    attention_heads = attention_heads.reshape(len(attention_heads), -1)  # flatten

    c_attn_opt_state = optimizer.state[attn._c_attn.weight]
    exp_avg_sq = c_attn_opt_state["exp_avg_sq"].detach().clone()
    eps: float = optimizer.param_groups[0]['eps']

    v_sq = torch.sqrt(exp_avg_sq + eps)
    v_sq = attn.get_attention_head_view(v_sq)
    v_sq = v_sq.reshape(len(attention_heads), -1)  # flatten

    head_norms = torch.linalg.norm(attention_heads, dim=1)  # copy
    transformed_norms = torch.linalg.norm(attention_heads / v_sq, dim=1)  # copy

    for i in range(attn.total_heads):
        norms[f"gradient_norm/layer{layer:02}/c_attn/head{i:02}"] = head_norms[i].item()
        norms[f"transformed_norm/layer{layer:02}/c_attn/head{i:02}"] = transformed_norms[i].item()

    layer_norm = torch.linalg.norm(attention_heads)
    transformed_layer_norm = torch.linalg.norm(attention_heads / v_sq)
    norms[f"gradient_norm/layer{layer:02}/c_attn/total"] = layer_norm.item()
    norms[f"transformed_norm/layer{layer:02}/c_attn/total"] = transformed_layer_norm.item()

    return norms


@torch.no_grad()
def get_projection_head_norms(attn: CausalSelfAttention, optimizer: torch.optim.Optimizer, layer: int) -> dict[str, float]:
    norms: dict[str, float] = {}

    projection_heads = attn.get_projection_head_gradients()
    projection_heads = projection_heads.reshape(len(projection_heads), -1)  # flatten

    c_proj_opt_state = optimizer.state[attn._c_proj.weight]
    exp_avg_sq = c_proj_opt_state["exp_avg_sq"].detach().clone()
    eps: float = optimizer.param_groups[0]['eps']

    v_sq = torch.sqrt(exp_avg_sq + eps)
    v_sq = attn.get_projection_head_view(v_sq)
    v_sq = v_sq.reshape(len(projection_heads), -1)

    projection_head_norms = torch.linalg.norm(projection_heads, dim=1)  # copy
    transformed_norms = torch.linalg.norm(projection_heads / v_sq, dim=1)  # copy

    for i in range(attn.total_heads):
        norms[f"gradient_norm/layer{layer:02}/c_proj/head{i:02}"] = projection_head_norms[i].item()
        norms[f"transformed_norm/layer{layer:02}/c_proj/head{i:02}"] = transformed_norms[i].item()

    layer_norm = torch.linalg.norm(projection_heads)
    transformed_layer_norm = torch.linalg.norm(projection_heads / v_sq)
    norms[f"gradient_norm/layer{layer:02}/c_proj/total"] = layer_norm.item()
    norms[f"transformed_norm/layer{layer:02}/c_proj/total"] = transformed_layer_norm.item()

    return norms


@torch.no_grad()
def get_head_distributions(attn: CausalSelfAttention, layer: int) -> dict[str, float]:
    distributions: dict[str, float] = {}

    c_attn = attn._c_attn.weight.data.detach().clone()
    heads = attn.get_attention_head_view(c_attn)
    heads = heads.reshape(len(heads), -1)  # flatten

    mean = torch.mean(heads, dim=1)
    variance = torch.std(heads, dim=1)

    for i in range(attn.total_heads):
        distributions[f"mean/layer{layer:02}/c_attn/head{i:02}"] = mean[i].item()
        distributions[f"variance/layer{layer:02}/c_attn/head{i:02}"] = variance[i].item()

    distributions[f"mean/layer{layer:02}/c_attn/total"] = torch.mean(c_attn, dim=(0, 1)).item()
    distributions[f"variance/layer{layer:02}/c_attn/total"] = torch.std(c_attn, dim=(0, 1)).item()

    return distributions

@torch.no_grad()
def get_attention_entropy(attn: CausalSelfAttention, layer: int) -> dict[str, float]:
    entropies: dict[str, float] = {}

    ent = attn.get_attention_entropy()

    for i in range(attn.total_heads):
        entropies[f"attn_entropy/layer{layer:02}/head{i:02}"] = ent[i].item()

    entropies[f"attn_entropy/layer{layer:02}/total"] = torch.mean(ent).item()

    return entropies