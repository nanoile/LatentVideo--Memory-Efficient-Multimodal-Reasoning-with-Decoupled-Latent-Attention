# V-MLA: Visual Multi-head Latent Attention for Long Video Understanding

**Optimizing Qwen 3.5-VL for Extended Video Processing via MLA Architecture**

## 🎯 Project Overview

This project reimplements Qwen 3.5-VL's attention mechanism using **Multi-head Latent Attention (MLA)** to achieve 4-5× KV cache compression, enabling processing of 30+ minute videos on 4×3090 GPUs (96GB VRAM).

### Key Innovation
- Replace GQA/MHA with MLA in both vision and language layers
- Latent space compression: 1152d → 256d (4.5× reduction)
- Structural distillation instead of full pretraining
- Custom Triton kernels for fused operations

## 📊 Expected Results

| Metric | Baseline (GQA) | V-MLA | Improvement |
|--------|---------------|-------|-------------|
| KV Cache @ 1hr video | ~180GB | ~45GB | 75% reduction |
| Max video length (4×3090) | ~8 min | ~35 min | 4.4× longer |
| Accuracy drop (MVBench) | - | <1% | Minimal loss |

## 🏗️ Architecture

```
Original: X → [Q, K, V] → Attention → Output
V-MLA:    X → Q, [K_lat, V_lat] → UpProj(K_lat), UpProj(V_lat) → Attention → Output
          
KV Cache: Store only K_lat, V_lat (256d instead of 1152d)
```

## 📁 Repository Structure

```
qwen_mla/
├── src/
│   ├── models/
│   │   ├── v_mla_qwen.py          # MLA-modified Qwen model
│   │   ├── mla_attention.py       # Core MLA attention module
│   │   └── rope_decoupled.py      # Decoupled RoPE for MLA
│   ├── distillation/
│   │   ├── trainer.py             # Structural distillation trainer
│   │   └── losses.py              # Layer-wise alignment losses
│   └── utils/
│       ├── profiler.py            # Memory & compute profiling
│       └── data_loader.py         # Video dataset handling
├── kernels/
│   └── mla_fused.py               # Triton fused kernels
├── experiments/
│   ├── phase1_baseline.py         # Baseline profiling
│   ├── phase2_architecture.py     # Architecture testing
│   ├── phase3_distillation.py     # Distillation experiments
│   └── phase4_evaluation.py       # Long-video benchmarks
├── configs/
│   ├── model_config.yaml          # Model hyperparameters
│   └── distill_config.yaml        # Distillation settings
├── scripts/
│   ├── run_baseline.sh
│   ├── run_distillation.sh
│   └── run_evaluation.sh
└── requirements.txt
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/nanoile/LatentVideo--Memory-Efficient-Multimodal-Reasoning-with-Decoupled-Latent-Attention.git
cd LatentVideo--Memory-Efficient-Multimodal-Reasoning-with-Decoupled-Latent-Attention
pip install -r requirements.txt
```

Set `configs/model_config.yaml` → `base_model` to a local path if you use ModelScope/HF cache layouts other than the default Hub id `Qwen/Qwen3-VL-8B-Instruct`. Optional: `export QWEN_VL_MODEL=...` for `experiments/test_init.py` and `phase2_architecture.py`.

### Phase 1: Baseline Profiling
```bash
bash scripts/run_baseline.sh
# Output: baseline_metrics.json with OOM thresholds
```

### Phase 2: Architecture Validation
```bash
python experiments/phase2_architecture.py
# Verify MLA module correctness
```

### Phase 3: Structural Distillation
```bash
bash scripts/run_distillation.sh
# 4×3090 DeepSpeed Stage 2, ~3 days on 50k video clips
```

### Phase 4: Evaluation
```bash
python experiments/phase4_evaluation.py --benchmark mvbench
# Test on MVBench, EgoSchema, Long-Video Needle
```

## 📈 Experiment Tracking

All experiments logged to WandB:
- Hidden state alignment curves
- Memory vs video length plots
- Accuracy comparison tables

## 🔬 Technical Details

### MLA Compression Ratio
- Original KV: `(batch, seq_len, num_heads, head_dim)` = `(1, 100k, 16, 72)` ≈ 11GB
- MLA KV_lat: `(batch, seq_len, d_lat)` = `(1, 100k, 256)` ≈ 2.5GB

### Distillation Strategy
- Teacher: Frozen Qwen 3.5-VL
- Student: V-MLA variant
- Loss: `MSE(hidden_states) + KL(logits)`
- Data: 50k clips from WebVid-10M
- Training: ~72 hours on 4×3090

### Custom Kernels
Triton kernel fuses:
1. Latent uprojection: `K_lat @ W_uk → K`
2. Scaled dot-product attention
3. Avoids materializing full K, V in HBM

## 📝 Citation

```bibtex
@misc{vmla2026,
  title={V-MLA: Scaling Video Understanding via Multi-head Latent Attention},
  author={Your Name},
  year={2026},
  note={Research project for PhD applications}
}
```

## 🎓 For PhD Applications

This project demonstrates:
- ✅ Deep understanding of Transformer architectures
- ✅ Systems optimization (memory, compute)
- ✅ Novel architecture design (MLA adaptation)
- ✅ Efficient training strategies (distillation)
- ✅ Custom kernel development (Triton)
- ✅ Rigorous experimental methodology

## 📧 Contact

For questions about this research: [your-email]

## 🙏 Acknowledgments

- Qwen Team for the base model
- DeepSeek for MLA architecture inspiration
- Community for 4×3090 optimization tips
