import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MLAAttention(nn.Module):
    """Multi-head Latent Attention with KV compression"""

    def __init__(self, hidden_size: int, num_heads: int, latent_dim: int, rope_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim

        # Q projection (standard)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # KV compression: down-project to latent space
        self.kv_down_proj = nn.Linear(hidden_size, latent_dim * 2, bias=False)

        # KV decompression: up-project from latent
        self.k_up_proj = nn.Linear(latent_dim, hidden_size, bias=False)
        self.v_up_proj = nn.Linear(latent_dim, hidden_size, bias=False)

        # RoPE for decoupled position encoding
        self.q_rope_proj = nn.Linear(hidden_size, rope_dim, bias=False)
        self.k_rope_proj = nn.Linear(latent_dim, rope_dim, bias=False)

        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = self.head_dim ** -0.5
        self._init_mla_linear_weights()

    def _init_mla_linear_weights(self):
        """Xavier with reduced gain so new MLA blocks start closer to transformer scale."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:

        bsz, seq_len, _ = hidden_states.shape

        # Q projection
        q = self.q_proj(hidden_states)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # KV compression to latent space
        kv_latent = self.kv_down_proj(hidden_states)
        k_latent, v_latent = kv_latent.chunk(2, dim=-1)

        # Handle KV cache
        if past_key_value is not None:
            k_latent = torch.cat([past_key_value[0], k_latent], dim=1)
            v_latent = torch.cat([past_key_value[1], v_latent], dim=1)

        if use_cache:
            past_key_value = (k_latent, v_latent)

        # Decompress KV from latent
        k = self.k_up_proj(k_latent)
        v = self.v_up_proj(v_latent)

        k = k.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (decoupled)
        if position_ids is not None:
            q_rope = self.q_rope_proj(hidden_states)
            k_rope = self.k_rope_proj(k_latent)
            # Simplified RoPE application (actual implementation needs cos/sin)
            q = q + q_rope.view(bsz, seq_len, self.num_heads, -1).transpose(1, 2)[..., :self.head_dim]
            k = k + k_rope.view(bsz, -1, self.num_heads, -1).transpose(1, 2)[..., :self.head_dim]

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value
