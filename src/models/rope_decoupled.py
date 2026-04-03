import torch
import torch.nn as nn

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embeddings"""
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

def rotate_half(x):
    """Rotate half the hidden dims"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

class DecoupledRoPE(nn.Module):
    """Decoupled RoPE for MLA"""

    def __init__(self, dim: int, max_position_embeddings: int = 32768):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position_ids):
        seq_len = x.shape[1]
        freqs = torch.outer(position_ids.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin
