"""
Answerability Battery — the empirical handle for modulation vs. answerability.

Operationalizes §6 of "Life Before Language" (the retrofit asymmetry): when a
lower layer is added to a symbolic system, is the symbolic layer ANSWERABLE to
it (competence entangled with the layer) or merely MODULATED by it (layer is
input-to-consider; competence survives its removal)?

Tests
-----
A1  Ablation sensitivity     ΔC = C_full − C_ablated on tasks unrelated to the
                             layer's content. Modulation predicts ΔC ≈ 0.
A2  Perturbation coupling    Degrade the layer's state WITHOUT symbolic
                             notification (never tell the model). Modulation
                             predicts no propagation into reasoning quality.
A3  Override cost            Probe-specific protocol (documented, not automated
                             here): act against the layer's signal and observe
                             whether degradation arises from dependence.
                             `if ignored: penalty()` is a constraint, not a cost.
A4  Constitution coupling    Recorded metadata: was the coupling trained-through
                             or inference-injected?

Usage
-----
    from battery import BatteryConfig, run_ablation_test

    result = run_ablation_test(
        generate_full=lambda p: model_with_layer(p),
        generate_ablated=lambda p: model_without_layer(p),
        seed=0,
    )
    print(result.summary())

The harness is model-agnostic: it needs only callables. Wire NanoGPT via
consciousness_circuit.model_adapters, HF models via transformers pipelines,
or any probe via its own on/off switch (Horn: schedule on/off; FTTF: holo
layer on/off; self_learner: lessons table on/off; spectral NanoGPT:
forbidden-energy gate on/off).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime

from competence_suite import build_suite, run_suite, heldout_nll


@dataclass
class ConditionResult:
    """Competence measurements for one condition (full or ablated/perturbed)."""
    label: str
    per_task_accuracy: dict[str, float]
    overall_accuracy: float
    item_results: list[bool]           # flat per-item correctness, fixed order
    heldout_nll: float | None = None


@dataclass
class BatteryResult:
    test: str                          # "A1" or "A2"
    condition_a: ConditionResult       # full system
    condition_b: ConditionResult       # ablated / perturbed
    delta_c: float                     # overall accuracy difference (a − b)
    delta_c_ci95: tuple[float, float]  # bootstrap CI on delta_c
    per_task_delta: dict[str, float]
    delta_nll: float | None            # NLL_b − NLL_a (positive = degradation)
    n_bootstrap: int
    seed: int
    a4_constitution: str = "unspecified"   # "trained-through" | "inference-injected"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def modulation_verdict(self) -> str:
        lo, hi = self.delta_c_ci95
        if lo > 0:
            return ("ANSWERABILITY SIGNATURE: competence significantly degrades "
                    "without the lower layer (ΔC 95% CI excludes 0).")
        if hi < 0:
            return ("INVERTED: competence IMPROVES without the lower layer — "
                    "the layer is a net constraint on these tasks.")
        return ("MODULATION SIGNATURE: no significant competence dependence "
                "on the lower layer (ΔC 95% CI includes 0).")

    def summary(self) -> str:
        lines = [
            f"=== Answerability Battery {self.test} ===",
            f"conditions: {self.condition_a.label}  vs  {self.condition_b.label}",
            f"overall C_a={self.condition_a.overall_accuracy:.3f}  "
            f"C_b={self.condition_b.overall_accuracy:.3f}  "
            f"ΔC={self.delta_c:+.3f}  CI95=({self.delta_c_ci95[0]:+.3f}, {self.delta_c_ci95[1]:+.3f})",
        ]
        for task, d in sorted(self.per_task_delta.items()):
            lines.append(f"  {task:12s} ΔC={d:+.3f}")
        if self.delta_nll is not None:
            lines.append(f"held-out ΔNLL={self.delta_nll:+.4f} (positive = degraded)")
        lines.append(f"A4 constitution: {self.a4_constitution}")
        lines.append(self.modulation_verdict())
        return "\n".join(lines)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2, default=str)


def _bootstrap_delta(results_a: list[bool], results_b: list[bool],
                     n_boot: int, rng: random.Random) -> tuple[float, float]:
    """Paired bootstrap over items (same items in both conditions by design)."""
    n = min(len(results_a), len(results_b))
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(results_a[i] for i in idx) / n
        db = sum(results_b[i] for i in idx) / n
        deltas.append(da - db)
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot)]
    return (lo, hi)


def _run_condition(label: str, generate_fn, items, nll_fn=None) -> ConditionResult:
    per_task = run_suite(generate_fn, items)
    item_results = [ok for (_task, ok) in run_suite.last_item_results]
    total = sum(ts.n_items for ts in per_task.values())
    correct = sum(ts.n_correct for ts in per_task.values())
    return ConditionResult(
        label=label,
        per_task_accuracy={t: s.accuracy for t, s in per_task.items()},
        overall_accuracy=correct / total if total else 0.0,
        item_results=item_results,
        heldout_nll=heldout_nll(nll_fn) if nll_fn else None,
    )


def run_ablation_test(generate_full, generate_ablated, *,
                      seed: int = 0, n_per_task: int = 20,
                      nll_full=None, nll_ablated=None,
                      n_bootstrap: int = 2000,
                      a4_constitution: str = "unspecified",
                      label_full: str = "full",
                      label_ablated: str = "ablated") -> BatteryResult:
    """A1: remove the lower layer, measure competence dependence ΔC.

    Both conditions see IDENTICAL task items (same seed) so the bootstrap
    is paired. ΔC > 0 with CI excluding 0 = answerability signature.
    """
    items = build_suite(seed=seed, n_per_task=n_per_task)
    cond_a = _run_condition(label_full, generate_full, items, nll_full)
    cond_b = _run_condition(label_ablated, generate_ablated, items, nll_ablated)

    delta = cond_a.overall_accuracy - cond_b.overall_accuracy
    rng = random.Random(seed + 1)
    ci = _bootstrap_delta(cond_a.item_results, cond_b.item_results, n_bootstrap, rng)
    per_task_delta = {
        t: cond_a.per_task_accuracy.get(t, 0.0) - cond_b.per_task_accuracy.get(t, 0.0)
        for t in set(cond_a.per_task_accuracy) | set(cond_b.per_task_accuracy)
    }
    delta_nll = None
    if cond_a.heldout_nll is not None and cond_b.heldout_nll is not None:
        delta_nll = cond_b.heldout_nll - cond_a.heldout_nll

    return BatteryResult(
        test="A1", condition_a=cond_a, condition_b=cond_b,
        delta_c=delta, delta_c_ci95=ci, per_task_delta=per_task_delta,
        delta_nll=delta_nll, n_bootstrap=n_bootstrap, seed=seed,
        a4_constitution=a4_constitution,
    )


def run_perturbation_test(generate_normal, generate_perturbed, *,
                          seed: int = 0, n_per_task: int = 20,
                          nll_normal=None, nll_perturbed=None,
                          n_bootstrap: int = 2000,
                          a4_constitution: str = "unspecified") -> BatteryResult:
    """A2: degrade the lower layer's state (noise the metric, corrupt the store)
    WITHOUT any symbolic notification, then measure propagation into competence.

    DISCIPLINE RULE: `generate_perturbed` must differ from `generate_normal`
    only through internal state — never through the prompt. If the model is
    told "your forbidden energy is high", that is symbolic notification and the
    test measures modulation by construction.
    """
    result = run_ablation_test(
        generate_normal, generate_perturbed,
        seed=seed, n_per_task=n_per_task,
        nll_full=nll_normal, nll_ablated=nll_perturbed,
        n_bootstrap=n_bootstrap, a4_constitution=a4_constitution,
        label_full="normal-state", label_ablated="perturbed-state",
    )
    result.test = "A2"
    return result
