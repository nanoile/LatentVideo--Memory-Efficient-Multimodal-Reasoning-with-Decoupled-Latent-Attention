import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
from transformers import AutoModelForImageTextToText
from src.utils.profiler import MemoryProfiler
import json

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    profiler = MemoryProfiler()
    
    print(f"Loading model: {args.model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Use visual encoder output dimension
    hidden_size = 1280  # Qwen3-VL visual hidden size
    print(f"Visual hidden size: {hidden_size}")
    
    results = {}
    for length in map(int, args.video_lengths.split(',')):
        print(f"\nTesting {length}s video...")
        profiler.snapshot(f"before_{length}s")
        try:
            num_tokens = length * 30 * 16
            dummy_input = torch.randn(1, num_tokens, hidden_size).to(model.device).to(torch.bfloat16)
            with torch.no_grad():
                _ = model.model(inputs_embeds=dummy_input, use_cache=True)
            profiler.snapshot(f"after_{length}s")
            peak = profiler.get_peak_memory()
            results[length] = {"status": "success", "peak_memory_mb": peak}
            print(f"✓ {length}s passed, peak: {peak}")
        except RuntimeError as e:
            results[length] = {"status": "OOM", "error": str(e)[:100]}
            print(f"✗ {length}s OOM")
            break
    
    profiler.save(f"{args.output_dir}/baseline_profile.json")
    with open(f"{args.output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--video_lengths", type=str, default="60,120,300,600")
    parser.add_argument("--output_dir", type=str, default="results/baseline")
    main(parser.parse_args())
