# V-MLA Quick Start Guide

## Installation

```bash
git clone https://github.com/nanoile/LatentVideo--Memory-Efficient-Multimodal-Reasoning-with-Decoupled-Latent-Attention.git
cd LatentVideo--Memory-Efficient-Multimodal-Reasoning-with-Decoupled-Latent-Attention
pip install -r requirements.txt
pip install -e .
```

## Running Experiments

### Phase 1: Baseline (Find OOM threshold)
```bash
bash scripts/run_baseline.sh
# Check results/baseline/results.json for OOM point
```

### Phase 2: Test MLA Architecture
```bash
python experiments/phase2_architecture.py
# Verify MLA forward pass works
```

### Phase 3: Distillation (3 days on 4×3090)
```bash
bash scripts/run_distillation.sh
# Monitor with: wandb login && check dashboard
```

### Phase 4: Evaluation
```bash
bash scripts/run_evaluation.sh
# Results in results/evaluation/
```

## Expected Timeline

- Phase 1: 2 hours
- Phase 2: 4 hours  
- Phase 3: 72 hours (3 days)
- Phase 4: 8 hours

Total: ~4 days for complete experiment
