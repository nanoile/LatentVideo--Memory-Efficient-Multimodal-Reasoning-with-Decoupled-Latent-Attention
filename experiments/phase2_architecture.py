import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from src.models.v_mla_qwen import VMLAQwen

_DEFAULT_MODEL = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen3-VL-8B-Instruct")


def test_forward():
    print("Testing MLA architecture...")
    model = VMLAQwen(_DEFAULT_MODEL, latent_dim=256)
    
    dummy_input = torch.randn(1, 100, 4096).to(model.base_model.device).to(torch.bfloat16)
    with torch.no_grad():
        output = model.base_model.model(inputs_embeds=dummy_input, use_cache=True)
    
    print(f"✓ Forward pass successful")
    print(f"✓ Output shape: {output[0].shape}")
    
if __name__ == "__main__":
    test_forward()
