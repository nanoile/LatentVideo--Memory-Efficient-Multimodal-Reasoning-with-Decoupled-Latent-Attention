# V-MLA Implementation Guide

## Quick Start

```bash
# 1. Setup environment
git clone https://github.com/nanoile/LatentVideo--Memory-Efficient-Multimodal-Reasoning-with-Decoupled-Latent-Attention.git
cd LatentVideo--Memory-Efficient-Multimodal-Reasoning-with-Decoupled-Latent-Attention
pip install -r requirements.txt
bash setup.sh

# 2. Run baseline profiling
bash scripts/run_baseline.sh

# 3. Test architecture
python experiments/phase2_architecture.py

# 4. Run distillation (requires dataset)
bash scripts/run_distillation.sh

# 5. Evaluate
bash scripts/run_evaluation.sh
```

## Architecture Details

### MLA Compression
```
Input: X ∈ R^(B×L×1152)
↓
KV Latent: [K_lat, V_lat] ∈ R^(B×L×256)  [4.5× compression]
↓
Decompress: K = K_lat @ W_uk, V = V_lat @ W_uv
↓
Attention: softmax(QK^T/√d)V
```

### Memory Savings
- Original KV: `(B, L, 16, 72)` = 1152d per token
- MLA KV_lat: `(B, L, 256)` = 256d per token
- **Compression: 4.5×**

## Expected Timeline

| Phase | Duration | Hardware |
|-------|----------|----------|
| Phase 1: Baseline | 2-4 hours | 4×3090 |
| Phase 2: Architecture | 1 day | Single GPU |
| Phase 3: Distillation | 3 days | 4×3090 |
| Phase 4: Evaluation | 1 day | 4×3090 |

## Key Files

- `src/models/mla_attention.py` - Core MLA implementation
- `src/models/v_mla_qwen.py` - Modified Qwen model
- `src/distillation/losses.py` - Distillation loss functions
- `kernels/mla_fused.py` - Triton kernels (optional optimization)

## Troubleshooting

**OOM during baseline?**
- Reduce video length
- Enable gradient checkpointing
- Use FP8 quantization

**Distillation not converging?**
- Check layer alignment indices
- Reduce learning rate
- Increase warmup steps

**Accuracy drop >1%?**
- Increase distillation epochs
- Add more alignment layers
- Use larger latent_dim (e.g., 512)
