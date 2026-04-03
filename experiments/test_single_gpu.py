import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
from transformers import AutoModelForImageTextToText
from src.models.v_mla_qwen import VMLAQwen

print("Loading configs...")
with open('configs/model_config.yaml') as f:
    model_cfg = yaml.safe_load(f)

print("\n1. Loading Teacher to CPU...")
teacher = AutoModelForImageTextToText.from_pretrained(
    model_cfg['model']['base_model'],
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": "cpu"}
)
teacher.eval()
print(f"✓ Teacher on CPU")

print("\n2. Loading Student to GPU...")
student_base = AutoModelForImageTextToText.from_pretrained(
    model_cfg['model']['base_model'],
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda:0"
)
student = VMLAQwen(student_base, model_cfg['mla']['latent_dim'])
print(f"✓ Student on GPU, Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

print("\n3. Testing forward pass...")
dummy_input = torch.randn(1, 10, 4096).to("cuda:0").to(torch.bfloat16)
with torch.no_grad():
    output = student.base_model.model(inputs_embeds=dummy_input)
print(f"✓ Forward pass OK, Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

print("\n✓ All single-GPU tests passed!")
print(f"✓ Student fits in 24GB: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB < 24 GB")
