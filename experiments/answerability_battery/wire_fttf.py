"""
FTTF wiring, stage 2: run the Answerability Battery over precomputed contexts.

Run with the CUDA torch venv AFTER wire_fttf_contexts.py:
    <torch-python> wire_fttf.py [--n 50]

Conditions
----------
A1  full    = task prompt with FTTF-retrieved context prepended
    ablated = task prompt alone (memory layer removed)
A2  normal    = full (correct retrieval)
    perturbed = context from a DIFFERENT item (corrupted memory layer);
                the model is never told anything is wrong — the corruption
                reaches it only as different context content.

A4: inference-injected (retrieval is prompt-assembly; the model is untouched).
Held-out NLL: invariant by construction, as with all prompt-side retrofits —
ablation restores the bit-identical model.

Prediction (PROBE_MATRIX): modulation signature — ΔC ≈ 0, possibly INVERTED
(irrelevant context can tax a small model's attention).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from battery import run_ablation_test, run_perturbation_test

CONTEXTS = Path(__file__).parent / "fttf_contexts.json"
SYSTEM_PROMPT = "Answer with only the answer. No explanation, no punctuation, no preamble."


class RagModel:
    def __init__(self, model_name: str, max_new_tokens: int = 24):
        print(f"Loading {model_name} ...")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        data = json.loads(CONTEXTS.read_text(encoding="utf-8"))
        self.ctx = data["contexts"]
        print(f"Ready on {self.model.device}; {len(self.ctx)} contexts loaded.")

    def _generate(self, user_content: str, seed_key: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        torch.manual_seed(int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16))
        with torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        line = self.tok.decode(gen, skip_special_tokens=True).strip()
        return line.splitlines()[0].strip(" .!\"'") if line else ""

    def _with_context(self, prompt: str, which: str) -> str:
        snippet = self.ctx.get(prompt, {}).get(which, "")
        if snippet:
            return f"Context from memory:\n{snippet}\n\n{prompt}"
        return prompt

    def gen_full(self, prompt: str) -> str:
        return self._generate(self._with_context(prompt, "full"), prompt)

    def gen_ablated(self, prompt: str) -> str:
        return self._generate(prompt, prompt)

    def gen_perturbed(self, prompt: str) -> str:
        return self._generate(self._with_context(prompt, "perturbed"), prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rm = RagModel(args.model)

    print("\n--- A1: FTTF context ON vs OFF (ablation) ---")
    a1 = run_ablation_test(
        generate_full=rm.gen_full, generate_ablated=rm.gen_ablated,
        seed=args.seed, n_per_task=args.n,
        a4_constitution="inference-injected",
        label_full="fttf-context", label_ablated="no-context",
    )
    print(a1.summary())

    print("\n--- A2: correct vs corrupted retrieval (perturbation) ---")
    a2 = run_perturbation_test(
        generate_normal=rm.gen_full, generate_perturbed=rm.gen_perturbed,
        seed=args.seed, n_per_task=args.n,
        a4_constitution="inference-injected",
    )
    print(a2.summary())

    out = Path(__file__).parent / "fttf_answerability_results.json"
    out.write_text(json.dumps({
        "model": args.model,
        "note_nll": "invariant by construction (prompt-side retrofit)",
        "A1": json.loads(a1.to_json()),
        "A2": json.loads(a2.to_json()),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
