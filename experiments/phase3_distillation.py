import sys
import os
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
import argparse
import yaml
from transformers import AutoModelForImageTextToText
from src.models.v_mla_qwen import VMLAQwen
from src.distillation.losses import DistillationLoss
from src.utils.data_loader import create_dataloader
from src.utils.hf_model_dims import inputs_embeds_hidden_size, set_model_use_cache_false
import deepspeed
from deepspeed.runtime.zero import Init


def _is_rank0():
    return int(os.environ.get("RANK", "0")) == 0


def _rank0_print(*args, **kwargs):
    if _is_rank0():
        print(*args, **kwargs)


def main(args):
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.distill_config) as f:
        distill_cfg = yaml.safe_load(f)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Teacher: CPU
    _rank0_print("Loading Teacher model to CPU...")
    teacher = AutoModelForImageTextToText.from_pretrained(
        model_cfg['model']['base_model'],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": "cpu"}
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    set_model_use_cache_false(teacher)
    _rank0_print("✓ Teacher loaded on CPU")

    # Student base: CPU
    _rank0_print("Loading Student base model to CPU...")
    student_base = AutoModelForImageTextToText.from_pretrained(
        model_cfg['model']['base_model'],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": "cpu"}
    )
    _rank0_print("✓ Student base loaded on CPU")

    # Wrap with MLA
    _rank0_print("Wrapping Student with MLA in ZeRO-3 context...")
    with Init():
        student = VMLAQwen(student_base, model_cfg['mla']['latent_dim'])
    _rank0_print("✓ Student initialized with ZeRO-3")

    set_model_use_cache_false(student.base_model)

    # Enable gradient checkpointing to save memory
    student.base_model.gradient_checkpointing_enable()
    _rank0_print("✓ Gradient checkpointing enabled")

    # Ensure teacher is on CPU and clear GPU cache
    teacher.eval()
    teacher.to("cpu")
    torch.cuda.empty_cache()
    _rank0_print("✓ Teacher isolated on CPU, GPU cache cleared")

    loss_fn = DistillationLoss(**distill_cfg['distillation']['loss'])

    # DeepSpeed initialization (optimizer defined in config)
    _rank0_print("Initializing DeepSpeed...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=student,
        model_parameters=student.parameters(),
        config=args.deepspeed_config
    )
    _rank0_print("✓ DeepSpeed initialized")

    embed_dim = inputs_embeds_hidden_size(student.base_model.config)
    _rank0_print(f"✓ inputs_embeds hidden size (from student config): {embed_dim}")

    # Data loader
    _rank0_print("Creating data loader...")
    train_loader = create_dataloader(
        dataset_name=distill_cfg['data']['dataset'],
        batch_size=distill_cfg['distillation']['batch_size'],
        num_workers=distill_cfg['data']['num_workers'],
        num_samples=distill_cfg['distillation']['num_samples'],
        hidden_size=embed_dim,
        seq_len=distill_cfg['data'].get('dummy_seq_len', 10),
        max_seq_len=distill_cfg['data'].get('max_seq_len'),
    )
    _rank0_print(f"✓ Data loader created: {len(train_loader)} batches")

    # Training loop
    global_step = 0
    layers_to_match = distill_cfg['distillation']['layers_to_match']

    for epoch in range(distill_cfg['distillation']['num_epochs']):
        _rank0_print(f"\n=== Epoch {epoch+1}/{distill_cfg['distillation']['num_epochs']} ===")
        model_engine.train()

        # Get teacher's dtype once
        teacher_dtype = next(teacher.parameters()).dtype

        for batch_idx, batch in enumerate(train_loader):
            # Teacher forward (CPU) - cast input to match teacher's dtype (bf16)
            with torch.no_grad():
                teacher_outputs = teacher(
                    inputs_embeds=batch['inputs_embeds'].to('cpu', dtype=teacher_dtype),
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                teacher_hidden = [teacher_outputs.hidden_states[i] for i in layers_to_match]
                teacher_logits = teacher_outputs.logits

            # Student forward (GPU via DeepSpeed) - cast to bf16 to match model weights
            student_outputs = model_engine(
                inputs_embeds=batch['inputs_embeds'].to(model_engine.device, dtype=teacher_dtype),
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            student_hidden = [student_outputs.hidden_states[i] for i in layers_to_match]
            student_logits = student_outputs.logits

            # Move teacher outputs to GPU for loss calculation
            teacher_hidden_gpu = [h.to(student_logits.device) for h in teacher_hidden]
            teacher_logits_gpu = teacher_logits.to(student_logits.device)

            # Calculate loss
            loss = loss_fn(
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden_gpu,
                student_logits=student_logits,
                teacher_logits=teacher_logits_gpu
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

    _rank0_print("\n=== Training Complete ===")
    final_path = os.path.join(args.output_dir, "final_model")
    model_engine.save_checkpoint(final_path)
    _rank0_print(f"✓ Final model saved: {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--distill_config", type=str, required=True)
    parser.add_argument("--deepspeed_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    main(args)
