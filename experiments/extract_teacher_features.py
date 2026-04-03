#!/usr/bin/env python3
"""
Phase 1: Extract Teacher features offline
Run Teacher inference on all samples and cache hidden states to disk
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _name, _subdir in (
    ("TRITON_CACHE_DIR", ".triton_cache"),
    ("TORCH_EXTENSIONS_DIR", ".torch_extensions"),
):
    _path = os.path.join(_repo_root, _subdir)
    os.makedirs(_path, exist_ok=True)
    os.environ.setdefault(_name, _path)
_tmp = os.path.join(_repo_root, ".tmp")
os.makedirs(_tmp, exist_ok=True)
for _k in ("TMPDIR", "TMP", "TEMP"):
    os.environ.setdefault(_k, _tmp)

import torch
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForImageTextToText

from src.utils.data_loader import create_dataloader
from src.utils.hf_model_dims import inputs_embeds_hidden_size


def extract_features(args):
    # Load configs
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.distill_config) as f:
        distill_cfg = yaml.safe_load(f)

    # Create output directory
    feature_dir = Path(args.output_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Load Teacher model
    print("Loading Teacher model...")
    teacher = AutoModelForImageTextToText.from_pretrained(
        model_cfg['model']['base_model'],
        dtype=torch.bfloat16,
        device_map="auto"  # Use all available GPUs
    )
    teacher.eval()
    teacher_dtype = next(teacher.parameters()).dtype
    print(f"✓ Teacher loaded on GPU ({teacher_dtype})")

    embed_dim = inputs_embeds_hidden_size(teacher.config)
    print(f"✓ inputs_embeds hidden size (from config): {embed_dim}")

    # Create data loader
    print("Creating data loader...")
    train_loader = create_dataloader(
        dataset_name=distill_cfg['data']['dataset'],
        batch_size=args.batch_size,
        num_workers=distill_cfg['data']['num_workers'],
        num_samples=distill_cfg['distillation']['num_samples'],
        hidden_size=embed_dim,
        seq_len=distill_cfg['data'].get('dummy_seq_len', 10),
        max_seq_len=distill_cfg['data'].get('max_seq_len'),
    )
    print(f"✓ Data loader created: {len(train_loader)} batches")

    # Extract features
    layers_to_match = distill_cfg['distillation']['layers_to_match']
    print(f"\nExtracting features from layers: {layers_to_match}")

    sample_idx = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Extracting")):
            # Teacher forward
            outputs = teacher(
                inputs_embeds=batch['inputs_embeds'].to(teacher.device, dtype=teacher_dtype),
                output_hidden_states=True,
                return_dict=True
            )

            # Extract and save features for each sample in batch
            for i in range(batch['inputs_embeds'].size(0)):
                features = {
                    'hidden_states': [outputs.hidden_states[layer][i].cpu() for layer in layers_to_match],
                    'logits': outputs.logits[i].cpu()
                }

                # Save to disk
                feature_path = feature_dir / f"sample_{sample_idx:06d}.pt"
                torch.save(features, feature_path)
                sample_idx += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {sample_idx} samples")

    print(f"\n✓ Feature extraction complete!")
    print(f"✓ Saved {sample_idx} feature files to {feature_dir}")

    # Save metadata
    metadata = {
        'num_samples': sample_idx,
        'layers_to_match': layers_to_match,
        'model': model_cfg['model']['base_model']
    }
    torch.save(metadata, feature_dir / "metadata.pt")
    print(f"✓ Metadata saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--distill_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    extract_features(args)
