import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
from transformers import AutoModelForImageTextToText
from src.models.v_mla_qwen import VMLAQwen
from deepspeed.runtime.zero import Init
import deepspeed

_DEFAULT_MODEL = os.environ.get(
    "QWEN_VL_MODEL", "Qwen/Qwen3-VL-8B-Instruct"
)


def test_model_loading():
    print("=" * 50)
    print("Test 1: Loading Teacher to CPU")
    print("=" * 50)
    
    teacher = AutoModelForImageTextToText.from_pretrained(
        _DEFAULT_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": "cpu"}
    )
    print("✓ Teacher loaded successfully")
    print(f"  Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    print("\n" + "=" * 50)
    print("Test 2: Loading Student base model")
    print("=" * 50)
    
    student_base = AutoModelForImageTextToText.from_pretrained(
        _DEFAULT_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    print("✓ Student base loaded")
    print(f"  Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    print("\n" + "=" * 50)
    print("Test 3: Wrapping with MLA (no DeepSpeed)")
    print("=" * 50)
    
    student = VMLAQwen(student_base, latent_dim=256)
    print("✓ MLA wrapped")
    print(f"  Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)

if __name__ == "__main__":
    test_model_loading()
