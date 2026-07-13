# Answerability Battery

Operationalizes the modulation-vs-answerability distinction from *Life Before
Language* §6 (the retrofit asymmetry) — the paper's own flagged "sharpest
empirical question": when a lower layer is added to a symbolic system, does the
symbolic layer become **answerable** to it, or is it merely **modulated** by it?

See `PROBE_MATRIX.md` (repo root) for the full experimental program this serves.

## The four tests

| # | Test | Implemented | Modulation predicts | Answerability predicts |
|---|---|---|---|---|
| A1a | **Scaffold ablation** — remove the layer *mechanism*, measure ΔC on unrelated tasks | `run_ablation_test` | ΔC ≈ 0 (or <0 tax) | — (scaffold survival ≠ answerability) |
| A1b | **Sediment ablation** — disrupt the internal structure *produced by* development under the layer | protocol + occupancy diagnostic (see NanoGPT `run_sediment_occupancy.py`) | little/no residue | disrupting residue costs competence |
| A2 | Perturbation coupling — degrade layer state *without symbolic notification* | `run_perturbation_test` | no propagation | competence degrades ("exhaustion" signature) |
| A3 | Override cost — act against the layer's signal | protocol only (probe-specific) | free | costly *from dependence*, not programmed penalty |
| A4 | Constitution coupling — trained-through vs inference-injected | recorded metadata | inference-injected | trained-through |

**A1 split into A1a/A1b (forced by the C3 flagship result).** A grown-in model
can survive *scaffold* ablation (the regulator is removable) precisely because
development changed the reasoner — the effect is now *sediment* in the weights.
So two ablations, not one: A1a removes the mechanism that shaped development;
A1b removes the structure development produced. Internalization = surviving A1a
while failing A1b. See PROBE_MATRIX.md for the B1/B2 bearer-change distinction
this implements.

## Discipline rules (violating these voids the result)

1. **Tasks must be unrelated to the layer's content.** Ablating a memory layer
   obviously hurts recall *of its own content* — that proves nothing. A1/A2 ask
   whether *general* competence depends on the layer, the way human reasoning
   depends on a body it isn't "about."
2. **A2 must not notify.** Telling the model "your forbidden energy is high" is
   symbolic input → modulation by construction. Perturbation must reach the
   model only through architecture/state.
3. **A3 must not be gamed.** `if ignore_signal: apply_penalty()` is a
   constraint, not a cost. Only degradation arising from genuine dependence counts.
4. **Same seed across conditions.** Both conditions see identical task items;
   the bootstrap is paired.

## Wiring the probes

The harness needs only callables — `generate_fn(prompt) -> str`, optionally
`nll_fn(text) -> float`:

| Probe | Toggle for the ablated condition |
|---|---|
| Gabriel's Horn (`horn-experiment`) | entropy-decay schedule on/off — **DONE, see repo's `run_answerability.py`: modulation confirmed (A1 ΔC=+0.010 CI +0.000..+0.025; A2 ΔC=+0.015 CI −0.045..+0.070)** |
| FTTF-Holographic-RAG | holographic layer bypassed |
| self_learner (AI_Command_Center) | lessons table not injected |
| NanoGPT spectral (flagship C2a/C2b/C3) | forbidden-energy gate / spectral layer ablated |

For NanoGPT/HF models, wire `generate_fn` through
`consciousness_circuit.model_adapters.create_adapter`.

## Expected results (the paper's predictions)

Per PROBE_MATRIX.md: every existing retrofit probe should show the
**modulation signature** (ΔC ≈ 0). The open row is the grown-in spectral
condition (flagship experiment C3): if it shows the **answerability signature**
while the retrofit conditions (C2a/C2b) do not, construction order is
empirically load-bearing. Either outcome is publishable.

## Files

- `competence_suite.py` — scale-appropriate tasks (copy, reverse, arithmetic
  chains, pattern completion) + held-out NLL. Numpy-free, stdlib only.
- `battery.py` — A1/A2 runners, paired bootstrap CIs, verdict logic, JSON export.
- `run_battery_demo.py` — self-test with mock models; proves the harness
  distinguishes both signatures. Run: `python run_battery_demo.py`
