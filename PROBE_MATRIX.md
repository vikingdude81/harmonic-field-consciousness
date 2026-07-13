# The Probe Matrix

**Version:** 1.1 — 2026-07-12
**Status:** Canonical project reference (alongside the thesis map, LGI technical
reference, and ECOSYSTEM_MAP.md)
**Companion to:** ECOSYSTEM_MAP.md, BRIDGE_AUDIT_JUL12_2026.md
**Purpose:** Map every implemented architecture against the structural variables the
*Life Before Language* framework says matter: bearer identity, sedimentation,
ablatability, construction order, continuity, and answerability. Instruments
(which measure) are listed separately from probes (which implement-and-fall-short).

**The evidence claim, stated precisely:** not "five independent systems converged,"
but *five design-diverse systems, built within one research lineage, repeatedly
reproduce the same bearer/change separation.* That is what the evidence supports,
and it is the claim a hostile reviewer cannot easily knock down.

**Methodological order** (formalization comes last, not first):

```
concept → operational definition → behavioral battery → empirical regularities → formalization
```

## The Matrix

| Probe | What accumulates / registers | What object is changed | Is the reasoner the thing changed? | Ablate lower layer → competence loss? (prediction) | Construction order | Continuity type | Verdict | Precise shortfall |
|---|---|---|---|---|---|---|---|---|
| **Gabriel's Horn** (github.com/vikingdude81/horn-experiment) | Entropy of sampling distribution, per CoT step | The output distribution (via temperature) | No — weights untouched, schedule is designer-set | **No — MEASURED** (A1 ΔC=+0.010, CI +0.000..+0.025; A2 ΔC=+0.015, CI −0.045..+0.070; Qwen2.5-0.5B, n=200/condition, 2026-07-12). NLL invariant by construction. | Retrofit (inference-time wrapper) | Per-generation only | Constraint — **modulation signature confirmed empirically** | Regulation **without stakes**: nothing in the system is worse off for violating the schedule |
| **LGI constellation** (lgi-executive, motion-lgi, lattice-grid-interface) | Forbidden-energy / curvature events in state space | The inference trajectory (ALLOW/DAMPEN/BLOCK) | No — the scored worker sees zones; the reasoner is gated, not changed | No (remove gate → raw generation persists) | Retrofit (governs a finished model) | Per-run | Constraint (state-dependent — sharpest constraint form) | Sensing **without consequence**: violation costs the system nothing it bears |
| **FTTF-Holographic-RAG** | Traces superposed into 10,000D project vectors (Stage 1); discrete 768D chunks (Stage 2) | The routing index vector — genuinely modified by each trace | No — the LLM queries the changed object; the LLM is unchanged | **No — MEASURED, and INVERTED** (A1 ΔC=−0.165, CI −0.225..−0.105: the memory layer *taxes* general competence; A2 ΔC=+0.005, CI −0.040..+0.050: corrupting the memory's *content* is undetectable on unrelated tasks. Qwen2.5-0.5B, n=200/cond, 2026-07-12) | Retrofit (external memory service) | Persistent, but of state not of a subject | Constraint — **modulation confirmed; the layer's presence costs attention while its content carries no weight** | Sedimentation **of the index, not the reasoner** — and the superposition has a measured capacity knee (`fourier_capacity_curve.py`) biological sedimentation doesn't share |
| **self_learner** (AI_Command_Center) | Distilled lessons w/ confidence + usage counts (SQLite) | The future system prompt | No — lessons reach the reasoner as input | No (drop lessons table → same model, same competence) | Retrofit (post-session loop) | Persistent, cross-session | Constraint (dispositional-shaped) | Development **without a developed subject**: accumulation changes inputs, not the accumulating system |
| **Evolutionary machinery** (ACC `quantum_genetic_algorithm.py` + LOCAL_Ai `QuantumOracleAgent`) → to be extracted as **Continuant Probe** | Per-individual fitness; population states (held in HRR memory) | The population distribution across generations | No individual is changed by its own consequence within its life | N/A (the "lower layer" *is* the selection loop) | Selection over instances | Lineage — not any continuing individual | **Closest to consequence** — real differential outcome, wrong bearer | Consequence **to lineage, not life**: individuals that fail don't propagate, but no individual bears its own failure |
| **NanoGPT spectral** (submodule) | Spectral geometry / forbidden_energy present in the architecture | Potentially the weights themselves, *during training* | **Open — this is the experiment** | **Open — this is the measurement** | **Grown-in candidate** (the only one) | Across training | TBD | The one system where the wall's key variable — construction order — can be manipulated |

**Instruments (not probes):**

| Instrument | Measures | Angle |
|---|---|---|
| consciousness_circuit v3.5.1 | 7-dimension activation profile in hidden states | Representation-level |
| consciousness_testbed (ACC) | Φ surrogate, neural complexity, metastability | Integration-dynamics-level |

Two theoretically unrelated instruments → cross-validation: run both on the same
system (e.g., each NanoGPT experimental condition). Agreement or precise disagreement
is Workstream 2 data either way.

## Reading the matrix

Every probe row shows the same signature, which the feedback correctly compressed:

```
function exists → structural analogue works → information crosses an interface
→ ✗ THE BEARER IS NOT THE THING CHANGED
```

Column-wise, the pattern is stark:
- **"Is the reasoner the thing changed?"** — No, five times. The only "open" is the grown-in candidate.
- **"Ablate → competence loss?"** — predicted No, five times; **now measured No for Gabriel's Horn** (first battery result, 2026-07-12: modulation signature on both A1 and A2). Four predictions remain to be measured. If any probe surprises us here, that is front-page news for the thesis in either direction.

One methodological note from the first measurement: for inference-time
processors, the modulation verdict is *architectural* — ablation restores the
bit-identical model, so held-out NLL cannot move. A1/A2 for this probe class
quantify output-shaping strength, and the A2 per-task deltas showed real
redistribution (copy −0.28, pattern +0.32) with zero net competence change —
a crisp empirical illustration of "shapes what the system does, not what it is."

The second measurement (FTTF, 2026-07-12) added a sharper observation: the
retrofit memory layer is not merely non-constitutive, it is a **net tax** —
prepended memory context significantly degrades unrelated competence
(ΔC=−0.165), while corrupting the memory's *content* changes nothing
(ΔC=+0.005). The reasoner pays for the layer's presence and is indifferent to
its truth. Contrast the biological case, where memory corruption devastates
reasoning and memory's presence is free — the retrofit signature is the exact
mirror image of the constitutive one. This "presence-tax / content-indifference"
pair may be the cleanest single-table statement of the modulation regime yet.

## Operationalizing answerability (the A-battery)

Four measurable components, straight from the framework's modulation-vs-answerability
checklist. Each is a concrete experiment, not a metaphysical claim.

| # | Test | Operationalization | Modulation predicts | Answerability predicts |
|---|---|---|---|---|
| **A1** | Ablation sensitivity | Remove the lower layer; measure ΔC = C_full − C_ablated on tasks *unrelated* to that layer's content | ΔC ≈ 0 | ΔC substantially > 0 |
| **A2** | Perturbation coupling | Change the lower layer's state **without symbolic notification** — noise the metric, corrupt the memory, but never *tell* the model; measure whether reasoning trajectory changes | Unaffected | Degrades (the "exhaustion" signature) |
| **A3** | Override cost | Have the higher layer act against the lower layer's signal; measure whether anything breaks or costs | Free (signal is advisory; model routes around it) | Costly — **but the cost must arise from dependence, not from a programmed penalty** |
| **A4** | Constitution coupling | Was the coupling present during weight formation (trained-through) or injected at inference? | Inference-injected | Trained-through |

Three discipline notes on the battery:

- **A1 measures *competence dependence*, not answerability.** Don't over-name it.
  A retrofit layer may alter outputs while leaving core competence intact; a
  constitutive layer should predict deeper degradation. The battery accumulates
  regularities; "answerability" is what we call the pattern *if* it shows up.
- **A2 is the experimental form of "input-to-consider vs. condition-of-operation."**
  Telling the model "forbidden energy is high" is symbolic notification — that's
  modulation by construction. The test only counts if the state change reaches
  reasoning through the architecture, not through the prompt.
- **A3 must not be gamed.** `if ignore_signal: apply_penalty()` is just another
  constraint and proves nothing. The interesting result is degradation that occurs
  because the reasoning process itself depends on the ignored structure.

A first-pass **answerability index**: A1–A3 as normalized effect sizes, A4 as a
binary moderator. Nothing fancier is needed to start benchmarking — the point is a
common yardstick across probes, not a final theory.

**Scale caveat for competence measures:** at NanoGPT scale there is little
"competence" to lose on realistic tasks, so ΔC must be defined against
scale-appropriate benchmarks — held-out perplexity, small synthetic reasoning
suites (arithmetic chains, copy/reverse tasks, structure completion), and the
tier-2 behavioral suites already in the spectral experiments. Otherwise A1/A2
return noise and the experiment falsely reports "no dependence."

## The flagship experiment: Construction-Order Dependence in Spectrally Governed Language Models

**Research question (stated conservatively):** Does integrating a regulatory geometry
during model formation produce stronger dependence of later reasoning competence on
that geometry than attaching the same or functionally similar regulatory structure
after training?

The hypothesis is deliberately narrow. It is **not** "grown-in AI is more conscious"
and not even "grown-in AI is more human-like." It is:

> **Construction order affects the degree of functional dependence between
> regulatory structure and symbolic competence.**

This gives the thesis the property it needs most: **it can lose.** If all conditions
behave identically under ablation, retrofit asymmetry (for this regulatory layer) is
weakened. If grown-in shows markedly greater ablation sensitivity and perturbation
coupling, construction history is empirically load-bearing. Either result matters.

| Condition | Description | Status |
|---|---|---|
| C1 | **Vanilla** — standard training, no spectral architecture | exists |
| C2a | **Inference retrofit** — vanilla trained, spectral/governance layer attached at inference only | exists (current LGI usage) |
| C2b | **Adapted retrofit** — vanilla trained, then *fine-tuned with the spectral layer attached* | near-term (one fine-tuning run) |
| C3 | **Grown-in** — spectral geometry present from initialization, entire training trajectory through it | spectral_transformer — near-term |
| C4 | C3 + continual state across episodes | phase 2 (engineering) |
| C5 | C3 + HRR sedimentation inside the loop | phase 2 (engineering) |

**Why C2b exists:** it operationalizes the thesis's own permanence-vs-convergence
question. The convergence view says deep-enough retrofit might collapse the
grown-in/bolted-on distinction; C2b *is* a deeper retrofit. If C2b behaves like C2a,
depth of retrofit doesn't close the gap. If C2b approaches C3, convergence gains
support. Without C2b the experiment can't distinguish "construction order matters"
from "amount of joint training matters."

**Confound control:** C2b and C3 must use the *same spectral architecture and
hyperparameters* — the only difference is when it enters the training trajectory.
Otherwise construction order is confounded with architecture.

**Measurement:** run the full A-battery on C1–C3 (both C2 variants). Same
scale-appropriate competence benchmarks across conditions; both instruments
(consciousness_circuit + testbed Φ/complexity/metastability) on every condition —
cross-instrument agreement or precise disagreement is Workstream 2 data either way.

**The paper's prediction:** C2a shows the modulation signature (ΔC ≈ 0 under
ablation). If C3 shows ΔC substantially > 0 with A2 coupling — competence *entangled*
with the grown-in geometry — then construction order measurably converts modulation
toward answerability. If C3 also shows ΔC ≈ 0, the asymmetry may sit deeper than
architecture — equally publishable, and honest either way.

### MEASURED RESULTS (2026-07-12) — three regimes, not two

Char-level TinyStories, 4-layer/256-dim SpectralTransformer, shared rank-16
P_bad, paired per-sample val-NLL ablation (P_bad → 0), n=1000 val samples,
bootstrap CI:

| Condition | NLL (full) | NLL (ablated) | ΔNLL | CI95 | Regime |
|---|---|---|---|---|---|
| C1 vanilla | 0.5164 | 0.5164 | +0.00000 | exact | control |
| C2a inference retrofit | 0.5264 | 0.5164 | −0.01002 | (−0.0108, −0.0092) | **TAX** — layer costs; ablation restores baseline exactly |
| C2b adapted retrofit (10 ep) | 0.5106 | 0.5143 | +0.00374 | (+0.0033, +0.0041) | **CRUTCH** — weights co-adapted; removal disturbs |
| C3 grown-in (20 ep) | 0.5149 | 0.5153 | +0.00042 | (+0.0002, +0.0006) | **INTERNALIZED** — near-vestigial at runtime |

**The naive prediction was wrong in the most informative way.** Grown-in did
NOT show the largest runtime dependence — the adapted retrofit did (≈9× C3,
non-overlapping CIs). Reading: the C3 model *never had access to* the
forbidden subspace, so it developed entirely within the allowed geometry —
the explicit regulator became nearly removable because the system grew into
the kind of thing that doesn't need it. The C2b model, formed *before* the
constraint, adapted around it and now leans on it. And C2a never adapted at
all, so the constraint is pure runtime cost.

This is §3's internalization phenomenon at the architecture level: external
regulation, present during formation, disappears into constitution — you can
remove the scaffold because the shaping now lives in the weights. Construction
order didn't change *how much* the system depends on the layer; it changed
**where the constraint lives** — outside as ongoing operation (retrofit) vs
inside as form (grown-in). That is §6's "direction of constitution," measured.

**Honest caveats before this goes near the paper:**
1. Effect sizes are small in absolute terms (10⁻³–10⁻² NLL); statistically
   solid (CIs exclude 0) but char-level toy scale.
2. ~~Budget confound~~ — resolved by the dose-response run below.
3. Single seed per condition; replicate before publishing numbers.
4. "Internalized" here concerns a *removable* regulator whose shaping persists
   in weights — it does not license any claim about stakes or consequence
   (the wall is untouched, exactly as the framework requires).

### DOSE-RESPONSE RESULT (2026-07-12): convergence with a permanent residue

C2b re-run at increasing fine-tuning doses, early stopping disabled, same
baseline + geometry + protocol throughout:

| Dose (epochs) | ΔNLL | CI95 |
|---|---|---|
| 10 | +0.00396 | (+0.00357, +0.00437) — replicates flagship C2b |
| 20 | +0.00203 | (+0.00175, +0.00232) |
| 40 | +0.00105 | (+0.00084, +0.00126) |
| 80 | +0.00112 | (+0.00093, +0.00132) |
| **C3 grown-in (reference)** | **+0.00042** | (+0.00023, +0.00063) |

**Reading: partial convergence, measurable floor.** Retrofit dependence halves
per doubling of adaptation until ~40 epochs, then flattens. The asymptote
(~+0.0011) sits ≈2.6× above the grown-in signature with non-overlapping CIs.
So on §6's permanence-vs-convergence question, the measurement answers *both,
quantified*: deep retrofit adaptation converges most of the way toward
internalization, and a residue of construction order persists at the
asymptote that further adaptation does not close. Workstream 2's deliverable
was "characterize the asymptotic limit" — this is a literal measured
asymptote, at toy scale, for one regulatory layer.

Additional caveats specific to this curve: (a) single seed per dose; (b) the
80-epoch model is well into overfitting (train 0.347 vs val 0.552 rising),
so the 40-epoch point is the cleanest deep-dose measurement — a 160-epoch
point plus seed replicates would firm up the floor; (c) toy scale as above.

**Follow-ups in value order:** seed replicates of {40, 80, C3}; a 160-epoch
point; the same sweep at word-level/GPT-2 scale; and the same protocol on a
*different* layer type (e.g., memory instead of geometry) to test whether the
residue generalizes.

## FTTF: feature the knee, don't hide it

The accurate claim: **FTTF implements sedimentation of the routing index, not of the
full memory substrate or the reasoner.** And the capacity knee makes the probe
*better*, not worse, because FTTF then delivers two results in Workstream 2 form:

1. A formal analogue of accumulation-by-superposition **can be implemented** (the
   10,000D HRR trace genuinely modifies the accumulating object).
2. The analogue has **empirically measurable limits** (the recovery knee from
   `fourier_capacity_curve.py`) *and* still leaves the downstream reasoner unchanged.

Build the closest formal analogue possible, then carefully describe the residue —
that is the project's whole discipline, and FTTF executes it exactly.

## The Continuant Probe: where L3 and L5 may be one target

The evolutionary machinery gives the **control case**:

```
individual acts → fitness evaluated → population changes → future individual differs
```

The missing architecture — the probe's actual target — is:

```
individual acts → outcome alters that individual → the altered individual continues
→ future reasoning develops from its own altered condition
```

The second loop bridges the thesis's Level 3 and Level 5 arguments, which may be two
views of one engineering target:

> **a persistent individual whose own outcomes alter the substrate from which its
> future actions arise.**

L5 asks: can accumulation change the accumulating system? L3 asks: can the same
entity bear the consequence of its action? A Continuant Probe that closes the second
loop would be a qualitatively different probe from everything currently built —
still not subjectivity (the framework makes no such promise), but the first system
on the far side of every "No" in the matrix's two key columns.

## Priority ranking

1. **NanoGPT construction-order experiment** (C1/C2a/C2b/C3 + A-battery)
2. **Formalize and run the four-part Answerability Battery** across existing probes
3. **Correct the FTTF claim** to "sedimentation of the routing index" everywhere it appears
4. **Rename/extract the evolutionary system as the Continuant Probe**
5. **Use this Probe Matrix as the central experimental map** linking codebase to thesis

## Required renames / extractions before publication

1. **Extract the evolutionary machinery** (`quantum_genetic_algorithm.py` +
   `QuantumOracleAgent`) into a clean standalone repo named **Continuant Probe**
   (or similar) — its thesis role: *differential consequence shapes a lineage; does any
   architecture transfer consequence from the lineage into the continuing individual?*
   Retire "Oracle Prime" from the paper; the name collides three ways locally and its
   cited `LayeredFitness` implementation does not resolve.
2. **Qualify the FTTF claim**: sedimentation of the routing *index*, not the store;
   note the capacity knee as a disanalogy with biological memory.
3. **Frame convergence as design diversity, not independence**: five architecturally
   unrelated systems hitting the same wall is the claim; "independent" invites a
   selection-effect objection the paper doesn't need to expose itself to.
4. **Keep the "repos followed the retrofit arrow" narrative out of the argument** —
   preface color only; the chronology doesn't survive a hostile audit.
