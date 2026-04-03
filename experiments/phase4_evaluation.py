import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
from src.models.v_mla_qwen import VMLAQwen
import json

def evaluate_long_video(model, video_length_minutes):
    num_tokens = video_length_minutes * 60 * 30 * 16
    dummy_input = torch.randn(1, num_tokens, 1152).to(model.base_model.device)
    
    try:
        with torch.no_grad():
            _ = model(inputs_embeds=dummy_input, use_cache=True)
        return "success"
    except RuntimeError as e:
        return f"failed: {str(e)[:100]}"

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model = VMLAQwen.from_pretrained(args.checkpoint)
    
    results = {}
    for length in [5, 10, 20, 30, 60]:
        print(f"Testing {length} min video...")
        results[f"{length}min"] = evaluate_long_video(model, length)
    
    with open(f"{args.output_dir}/eval_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default="mvbench")
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    main(parser.parse_args())
