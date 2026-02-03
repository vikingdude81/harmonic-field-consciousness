"""
Example: math correctness clean vs corrupted prompts, activation patch sweep.

This script:
- Runs clean and corrupted prompts.
- Sweeps residual patching across layers to see impact on a correctness metric.
- Prints layer-wise metrics.

Prereqs: torch, transformers, your model + tokenizer.
"""
import argparse
from typing import Any, Dict

import torch

from consciousness_circuit.patching import residual_patch_sweep


def correctness_metric(logits: torch.Tensor, encoded_inputs: Dict[str, torch.Tensor]) -> float:
    # Simple: probability of the correct answer token for the last position
    # User should adapt this to their task
    target_id = encoded_inputs.input_ids[0, -1]
    probs = torch.softmax(logits[0, -1], dim=-1)
    return probs[target_id].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-clean", required=True)
    parser.add_argument("--prompt-corrupt", required=True)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")

    layers = None
    if args.layer_end is not None:
        layers = list(range(args.layer_start, args.layer_end))

    results = residual_patch_sweep(
        model,
        tok,
        prompt_clean=args.prompt_clean,
        prompt_corrupt=args.prompt_corrupt,
        target_fn=correctness_metric,
        layer_indices=layers,
        token_pos=-1,
    )

    print("Layer -> patched metric (higher means closer to clean)")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
