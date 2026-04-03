#!/usr/bin/env python3
"""
Phase 2: Train Student with cached Teacher features (loads sample_*.pt from disk).
Does not run the teacher — run scripts/run_extract_features.sh first.
"""
import os
import sys

# Before `import torch`: single arch + thin parallel compile. Do not leave empty — DS JIT briefly clears this env
# during its own builds; an empty value makes nvcc target every arch (huge RAM, often SIGKILL under 4 ranks).
_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
os.environ["TORCH_CUDA_ARCH_LIST"] = _arch if _arch else "8.6"
os.environ.setdefault("MAX_JOBS", "1")
os.environ.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

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
_hf_home = os.path.join(_repo_root, ".cache", "huggingface")
os.makedirs(_hf_home, exist_ok=True)
os.environ.setdefault("HF_HOME", _hf_home)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_hf_home, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_hf_home, "transformers"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_repo_root, ".cache", "xdg"))
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(_repo_root, ".cache", "modelscope"))
os.makedirs(os.environ["MODELSCOPE_CACHE"], exist_ok=True)

import gc
from pathlib import Path

import torch
import yaml
import argparse
import deepspeed
from transformers import AutoModelForImageTextToText
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from src.models.v_mla_qwen import VMLAQwen
from src.distillation.losses import DistillationLoss
from src.utils.data_loader import create_cached_feature_dataloader
from src.utils.hf_model_dims import inputs_embeds_hidden_size, set_model_use_cache_false
from src.utils.deepspeed_config_utils import normalize_deepspeed_config_dict_inplace
from src.utils.qwen_vl_hidden_hooks import register_qwen3_vl_text_layer_hooks


def _is_rank0():
    return int(os.environ.get("RANK", "0")) == 0


def _rank0_print(*args, **kwargs):
    if _is_rank0():
        print(*args, **kwargs)


def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)

    # Load configs
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.distill_config) as f:
        distill_cfg = yaml.safe_load(f)

    feature_root = Path(args.feature_dir).resolve()
    if not feature_root.is_dir():
        raise FileNotFoundError(
            f"Feature directory does not exist: {feature_root}\n"
            "This entry point does NOT run teacher feature extraction.\n"
            "Run Phase 1 first:  bash scripts/run_extract_features.sh"
        )
    n_feat_files = len(list(feature_root.glob("sample_*.pt")))
    if n_feat_files == 0:
        raise FileNotFoundError(
            f"No sample_*.pt under {feature_root}.\n"
            "Training expects offline caches from experiments/extract_teacher_features.py.\n"
            "Run:  bash scripts/run_extract_features.sh\n"
            "Then point --feature_dir to the same directory (default: cached_features)."
        )
    num_samples_cfg = int(distill_cfg["distillation"]["num_samples"])
    if num_samples_cfg > n_feat_files:
        raise ValueError(
            f"distillation.num_samples={num_samples_cfg} but only {n_feat_files} feature file(s) exist. "
            f"Set num_samples <= {n_feat_files} in configs/distill_config.yaml or extract more features."
        )

    # Keep alive: HF uses a weakref so `from_pretrained` can detect ZeRO-3 and load via
    # `_load_state_dict_into_zero3_model` (manual Init() alone leaves that off and breaks loading).
    hf_ds_config = HfDeepSpeedConfig(args.deepspeed_config)

    deepspeed.init_distributed()

    # After distributed init, some stacks merge these into zero_optimization; ZeRO pydantic rejects them there.
    normalize_deepspeed_config_dict_inplace(hf_ds_config.config)

    _rank0_print("Loading Student with HuggingFace ZeRO-3 integration...")
    student_base = AutoModelForImageTextToText.from_pretrained(
        model_cfg['model']['base_model'],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    student = VMLAQwen(student_base, model_cfg['mla']['latent_dim'])
    _rank0_print("✓ Student initialized with ZeRO-3")

    set_model_use_cache_false(student.base_model)

    # Enable gradient checkpointing
    student.base_model.gradient_checkpointing_enable()
    _rank0_print("✓ Gradient checkpointing enabled")

    # Clear GPU cache
    torch.cuda.empty_cache()
    _rank0_print("✓ GPU cache cleared")

    loss_fn = DistillationLoss(**distill_cfg['distillation']['loss'])

    # Optimizer from deepspeed_config (required for ZeRO CPU offload path).
    _rank0_print("Initializing DeepSpeed...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=student,
        model_parameters=student.parameters(),
        config=hf_ds_config.config,
    )
    _rank0_print("✓ DeepSpeed initialized")

    embed_dim = inputs_embeds_hidden_size(student.base_model.config)
    _rank0_print(f"✓ inputs_embeds hidden size (from student config): {embed_dim}")

    # DataLoader: load each .pt in Dataset (del dict immediately); num_workers=0, pin_memory=False.
    _rank0_print("Creating data loader (CachedFeatureDataset, workers=0)...")
    train_loader = create_cached_feature_dataloader(
        feature_dir=str(feature_root),
        num_samples=num_samples_cfg,
        batch_size=distill_cfg['distillation']['batch_size'],
        hidden_size=embed_dim,
        seq_len=distill_cfg['data'].get('dummy_seq_len', 10),
        max_seq_len=distill_cfg['data'].get('max_seq_len'),
        distributed=True,
    )
    _rank0_print(f"✓ Data loader created: {len(train_loader)} batches")

    _student_mod = getattr(model_engine, "module", model_engine)
    student_dtype = next(_student_mod.parameters()).dtype

    layers_to_match = distill_cfg["distillation"]["layers_to_match"]
    logit_w = float(distill_cfg["distillation"]["loss"].get("logit_weight", 0))

    hook_bufs = None
    hook_handles = None
    try:
        hook_bufs, hook_handles = register_qwen3_vl_text_layer_hooks(
            _student_mod.base_model, layers_to_match
        )
        _rank0_print("✓ LM layer hooks on (no output_hidden_states) to cut activation memory.")
    except Exception as exc:
        _rank0_print(f"⚠ LM hooks unavailable ({exc}); using output_hidden_states=True.")

    try:
        _training_loop(
            model_engine,
            train_loader,
            distill_cfg,
            loss_fn,
            layers_to_match,
            student_dtype,
            logit_w,
            hook_bufs,
            args,
        )
    finally:
        if hook_handles:
            for _h in hook_handles:
                _h.remove()


def _training_loop(
    model_engine,
    train_loader,
    distill_cfg,
    loss_fn,
    layers_to_match,
    student_dtype,
    logit_w,
    hook_bufs,
    args,
):
    global_step = 0
    for epoch in range(distill_cfg['distillation']['num_epochs']):
        _rank0_print(f"\n=== Epoch {epoch+1}/{distill_cfg['distillation']['num_epochs']} ===")
        model_engine.train()
        if train_loader.sampler is not None and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            teacher_hidden_batch = [
                h.to(model_engine.device) for h in batch["teacher_hidden_batch"]
            ]
            teacher_logits_batch = batch["teacher_logits_batch"].to(model_engine.device)

            if hook_bufs is not None:
                for _b in hook_bufs:
                    _b.clear()

            fwd_kw = dict(return_dict=True, use_cache=False)
            if hook_bufs is not None:
                fwd_kw["output_hidden_states"] = False
                if logit_w == 0:
                    fwd_kw["logits_to_keep"] = 1
            else:
                fwd_kw["output_hidden_states"] = True

            student_outputs = model_engine(
                inputs_embeds=batch["inputs_embeds"].to(model_engine.device, dtype=student_dtype),
                **fwd_kw,
            )
            if hook_bufs is not None:
                student_hidden = [b[0] for b in hook_bufs]
            else:
                student_hidden = [student_outputs.hidden_states[i] for i in layers_to_match]
            student_logits = student_outputs.logits

            # Calculate loss
            loss = loss_fn(
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden_batch,
                student_logits=student_logits,
                teacher_logits=teacher_logits_batch
            )

            # Backward
            model_engine.backward(loss)
            model_engine.step()

            global_step += 1

            # Logging
            if global_step % distill_cfg['logging']['log_interval'] == 0:
                _rank0_print(f"Step {global_step} | Loss: {loss.item():.4f}")

            # Checkpoint
            if global_step % distill_cfg['logging']['eval_interval'] == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                model_engine.save_checkpoint(ckpt_path)
                _rank0_print(f"✓ Saved checkpoint: {ckpt_path}")

            del (
                teacher_hidden_batch,
                teacher_logits_batch,
                student_hidden,
                student_logits,
                student_outputs,
                loss,
            )
            if global_step % 10 == 0:
                gc.collect()

            if args.max_steps is not None and global_step >= args.max_steps:
                _rank0_print(f"Stopping early (--max_steps={args.max_steps}).")
                break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    if args.max_steps is not None:
        _rank0_print("\n=== Training stopped (max_steps) ===")
    else:
        _rank0_print("\n=== Training Complete ===")
        final_path = os.path.join(args.output_dir, "final_model")
        model_engine.save_checkpoint(final_path)
        _rank0_print(f"✓ Final model saved: {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--distill_config", type=str, required=True)
    parser.add_argument("--deepspeed_config", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help=(
            "Stop after this many dataloader iterations (each calls backward+step). "
            "For a meaningful smoke test, use max_steps >= gradient_accumulation_steps in the DeepSpeed JSON "
            "so at least one ZeRO optimizer step runs."
        ),
    )
    args = parser.parse_args()
    main(args)
