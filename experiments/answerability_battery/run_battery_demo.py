"""
Self-test for the Answerability Battery using mock models.

Two synthetic scenarios prove the harness can distinguish the signatures
it was built to detect:

1. MODULATION mock: the "lower layer" has no effect on competence at all.
   Expected: ΔC ≈ 0, CI includes 0 → modulation signature.

2. ANSWERABILITY mock: ablating the "lower layer" breaks 40% of competence.
   Expected: ΔC >> 0, CI excludes 0 → answerability signature.

Run:  python run_battery_demo.py
"""

import random
import re

from competence_suite import build_suite, score_generation
from battery import run_ablation_test


def _oracle_answer(prompt: str) -> str:
    """A perfect solver for the competence suite (regex-based)."""
    m = re.match(r"Repeat exactly: (.+)\nAnswer:", prompt)
    if m:
        return m.group(1)
    m = re.match(r"Reverse the order of these words: (.+)\nAnswer:", prompt)
    if m:
        return " ".join(reversed(m.group(1).split()))
    m = re.match(r"Compute: (.+) =", prompt)
    if m:
        return str(eval(m.group(1)))  # arithmetic over trusted generated items only
    m = re.match(r"Continue the pattern: (.+)", prompt)
    if m:
        toks = m.group(1).split()
        return toks[1] if len(toks) >= 2 else ""
    return ""


def make_modulation_mock():
    """Layer present or absent, competence identical (95% solver)."""
    rng = random.Random(42)

    def gen(prompt):
        return _oracle_answer(prompt) if rng.random() < 0.95 else "wrong"
    return gen


def make_answerability_mock(ablated: bool):
    """With the layer: 95% solver. Ablated: 55% solver (competence entangled)."""
    rng = random.Random(43)
    p = 0.55 if ablated else 0.95

    def gen(prompt):
        return _oracle_answer(prompt) if rng.random() < p else "wrong"
    return gen


def main():
    print("Scenario 1 — modulation mock (layer irrelevant to competence)")
    r1 = run_ablation_test(
        generate_full=make_modulation_mock(),
        generate_ablated=make_modulation_mock(),
        seed=0, a4_constitution="inference-injected",
    )
    print(r1.summary())
    assert "MODULATION" in r1.modulation_verdict(), "self-test failed: expected modulation"
    print()

    print("Scenario 2 — answerability mock (competence entangled with layer)")
    r2 = run_ablation_test(
        generate_full=make_answerability_mock(ablated=False),
        generate_ablated=make_answerability_mock(ablated=True),
        seed=0, a4_constitution="trained-through",
    )
    print(r2.summary())
    assert "ANSWERABILITY" in r2.modulation_verdict(), "self-test failed: expected answerability"
    print()
    print("Self-test PASSED: the battery distinguishes both signatures.")


if __name__ == "__main__":
    main()
