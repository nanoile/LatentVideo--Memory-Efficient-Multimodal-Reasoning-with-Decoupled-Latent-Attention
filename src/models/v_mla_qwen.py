import os
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText
from .mla_attention import MLAAttention

class VMLAQwen(nn.Module):
    """Qwen3-VL with MLA attention replacement"""

    def __init__(self, base_model_name_or_model, latent_dim: int = 256, rope_dim: int = 64):
        super().__init__()
        # Support both model path and pre-loaded model
        if isinstance(base_model_name_or_model, str):
            self.base_model = AutoModelForImageTextToText.from_pretrained(
                base_model_name_or_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto"
            )
        else:
            self.base_model = base_model_name_or_model
        self.config = self.base_model.config
        self.latent_dim = latent_dim
        self._replace_attention_layers(latent_dim, rope_dim)

    def _replace_attention_layers(self, latent_dim: int, rope_dim: int):
        """Replace all attention modules with MLA"""
        # Qwen3-VL structure: model.layers (language model)
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            for layer in self.base_model.model.layers:
                if hasattr(layer, 'self_attn'):
                    hidden_size = layer.self_attn.hidden_size
                    num_heads = layer.self_attn.num_heads
                    layer.self_attn = MLAAttention(hidden_size, num_heads, latent_dim, rope_dim)
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"✓ Replaced {len(self.base_model.model.layers)} language layers with MLA")

        # Vision encoder
        if hasattr(self.base_model, 'visual') and hasattr(self.base_model.visual, 'blocks'):
            for block in self.base_model.visual.blocks:
                if hasattr(block, 'attn'):
                    hidden_size = block.attn.embed_dim
                    num_heads = block.attn.num_heads
                    block.attn = MLAAttention(hidden_size, num_heads, latent_dim, rope_dim)
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"✓ Replaced {len(self.base_model.visual.blocks)} vision layers with MLA")

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    def generate(self, **kwargs):
        return self.base_model.generate(**kwargs)
