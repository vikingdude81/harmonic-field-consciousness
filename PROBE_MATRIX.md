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
| **A1a** | **Scaffold ablation** | Remove the lower-layer *mechanism/regulator*; measure ΔC on tasks *unrelated* to its content. Answers: does the system need the scaffold's continuing presence? | ΔC ≈ 0 (or ΔC<0 tax) | — (see note) |
| **A1b** | **Sediment ablation** | Identify the internal structure *produced by* development under the layer, and disrupt THAT. Answers: did the regulatory history leave a competence-bearing residue in the reasoner? | little/no residue to disrupt | disrupting the residue costs competence |
| **A2** | Perturbation coupling | Change the lower layer's state **without symbolic notification** — noise the metric, corrupt the memory, but never *tell* the model; measure whether reasoning trajectory changes | Unaffected | Degrades (the "exhaustion" signature) |
| **A3** | Override cost | Have the higher layer act against the lower layer's signal; measure whether anything breaks or costs | Free (signal is advisory; model routes around it) | Costly — **but the cost must arise from dependence, not from a programmed penalty** |
| **A4** | Constitution coupling | Was the coupling present during weight formation (trained-through) or injected at inference? | Inference-injected | Trained-through |

**Why A1 had to split (the C3 result forced it).** The flagship's grown-in
model (C3) survived scaffold ablation almost untouched (ΔNLL≈+0.0004) — which
looked *backwards* under a single-ablation A1 ("constitutive → ablation hurts
more"). The resolution: we had conflated two experiments.

- **A1a (scaffold ablation)** removes the mechanism that *shaped* development.
- **A1b (sediment ablation)** removes the structure development *produced*.

The burned child is the analogy (as analogy, not evidence): remove the parent
who taught fire-avoidance and the disposition remains — that is the child
surviving *scaffold* ablation. The disposition itself is the *sediment*, and
disrupting it is A1b. C3 surviving A1a is exactly what internalization should
look like: the scaffold became removable *because the reasoner changed*.

**Revised definition of internalization:** demonstrated not when removing the
original regulator destroys competence, but when the regulator can be removed
**while its historically induced structure remains causally load-bearing inside
the reasoner.** A1a shows the first half; A1b tests the second.

**Two bearer/change questions (also forced by C3):**
- **B1 — Did the process modify the reasoner?** C3: **YES, measured** (see
  sediment-occupancy result below).
- **B2 — Does a later outcome modify the same continuing reasoner that produced
  the action, which then proceeds from the altered condition?** C3: **NO / not
  instantiated.** This is the wall's actual consequence claim, and it is
  stronger than B1. Constitutional change (B1) and consequence-bearing (B2) are
  **separable** — C3 is the first concrete case of B1-without-B2.

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

**Reading: partial convergence toward a non-zero floor.** Retrofit dependence
halves per doubling of adaptation until ~40 epochs, then flattens at ~+0.0011,
≈2.6× above the grown-in signature (non-overlapping CIs).

**Reviewer-proof wording (use this, not stronger):**
> Within the tested adaptation regime, increased retrofit adaptation approached
> but did not reach the grown-in signature, asymptoting at an observed floor
> approximately 2.6× higher. This is consistent with a residual construction-
> order dependence and does not support complete convergence within the
> measured regime.

Do **not** write "construction order leaves a permanent residue" — a finite
experiment can't establish permanence. The philosophical debate offered two
positions (permanence vs convergence); the data point to a **third: path
dependence with asymptotic adaptation.** That third option is more interesting
than either extreme, and the restraint makes the asymptote *more* compelling,
not less. Workstream 2's deliverable was "characterize the asymptotic limit" —
this is a measured limit, at toy scale, for one regulatory layer.

### SEDIMENT OCCUPANCY (2026-07-13): B1 confirmed for C3

With NO GRL active anywhere, fraction of hidden-state energy each model places
in the forbidden subspace (chance for isotropic states = rank/dim = 0.0625):

| Condition | Occupancy | vs chance |
|---|---|---|
| C1 vanilla | 0.0603 | at chance — never regulated |
| C2b dose 10ep | 0.0355 | pushed below |
| C2b dose 20ep | 0.0310 | |
| C2b dose 40ep | 0.0282 | |
| C2b dose 80ep | 0.0270 | monotone with dose |
| **C3 grown-in** | **0.0287** | below chance; sharpest layer-0 drop (0.038 vs C1's 0.077) |

The regulated models learned to *avoid* the forbidden subspace in their weights
— visible with the scaffold entirely removed. This is **B1 measured**: formation
under the constraint changed the reasoner. Note this is the *precursor* to full
A1b sediment ablation: occupancy shows *where* the sediment is; the next
experiment perturbs that structure and measures the competence cost. If C3
survives scaffold ablation (A1a ✓ measured) but collapses under sediment
disruption (A1b, pending) while vanilla/retrofit controls do not, that is the
cleanest evidence yet that **the reasoner became the thing changed** (B1) —
while still saying nothing about B2 (the wall stands).

Observation worth noting: C2b occupancy keeps decreasing with dose (0.036→0.027)
and by 40–80ep approaches C3's 0.029 — the *occupancy* nearly converges even
though the *ablation-dependence* floored at 2.6×. So retrofit adaptation can
reproduce the grown-in model's internal geometry-avoidance while retaining a
larger residual runtime dependence on the scaffold — a subtler split between
"internal form" and "scaffold reliance" than either view predicted.

### SEDIMENT ABLATION A1b (2026-07-13): sediment is real, and its sign is *robustness*

Inject matched-magnitude energy (α·‖h‖) into span(P_bad) vs a matched-rank
random subspace orthogonal to it; **sediment_effect = ΔNLL_forbidden −
ΔNLL_random** (the damage attributable to disrupting the *avoided* structure,
over generic perturbation sensitivity). At α=1.0:

| Condition | ΔNLL forbidden | ΔNLL random | sediment_effect |
|---|---|---|---|
| C1 vanilla | +2.334 | +2.337 | **−0.003** (forbidden not special — control works) |
| C2b 10ep | +2.211 | +2.400 | −0.189 |
| C2b 40ep | +2.263 | +2.425 | −0.161 |
| C2b 80ep | +2.204 | +2.349 | −0.146 |
| C3 grown-in | +2.158 | +2.324 | −0.166 |

**The naive A1b prediction was wrong, and the correct reading is better.** I
predicted "forbidden ≫ random for C3" (internalized structure = fragile under
disruption). The data show the *opposite sign*: regulated models are **more
robust** to forbidden-subspace perturbation than to random perturbation. The
mechanism is exactly internalization: these models relocated competence into the
allowed complement, so energy injected along the avoided directions lands in
dead space (cheap), while random directions hit directions they actually use
(costly). C1 vanilla, which never organized around P_bad, is hurt equally by
both (sediment_effect ≈ 0) — **the control confirms the effect is P_bad-specific,
not generic noise sensitivity.**

**This refines the A1b concept itself.** Sediment ablation's *sign depends on
what kind of structure was internalized*:
- **Avoidance sediment** (this case): internalization shows up as *robustness*
  in the avoided directions (sediment_effect < 0).
- **Skill/feature sediment** (a different experiment): internalization would
  show up as *fragility* when the used structure is disrupted (sediment_effect
  in the used directions > 0).
Either way, the signature is a **differential** between structured and random
perturbation that vanilla lacks — that differential is the sediment being
causally real.

**What A1b establishes and doesn't.** Establishes: B1 a second, independent way —
regulated models carry a competence-relevant reorganization in their weights
(clearly separated from vanilla). Does *not* establish: a clean dose gradient or
a C3 peak — at this toy scale the signature is essentially binary
regulated-vs-unregulated (all regulated models in −0.15…−0.19). Caveats: single
injection seed, single model seed/condition, char scale; seed replicates and a
larger model would firm the dose picture.

### SEED REPLICATES (2026-07-13): all three headline claims survive; one new gradient

3 model seeds × {C3 fresh-init, C2b-40ep, C2b-10ep}, each with scaffold
ablation, occupancy, and A1b (3 injection seeds × 2 α). Fixed C1 + P_bad as
shared environment. Ranges below are min..max across model seeds.

**1. Scaffold ablation (A1a) — the triad replicates with striking stability:**

| Group | ΔNLL mean | seed range | Non-overlap |
|---|---|---|---|
| C2b 10ep | +0.00370 | +0.00368..+0.00371 | ✓ vs both others |
| C2b 40ep | +0.00095 | +0.00078..+0.00105 | ✓ vs both others |
| C3 grown-in | +0.00033 | +0.00023..+0.00051 | ✓ vs both others |

The dependence ordering C2b10 > C2b40 > C3 holds for every seed; the
C2b40/C3 floor ratio is ≈2.9× on replicate means (2.6× single-seed). The
crutch→internalized separation is not seed noise.

**2. Occupancy — internal-form convergence replicates:** C2b40 (0.0267) ≈ C3
(0.0277) < C2b10 (0.0357) << C1 (0.0603). Deep retrofit reproduces grown-in
internal geometry-avoidance while retaining ~3× the scaffold dependence —
the form/reliance split is robust.

**3. A1b — regulated vs vanilla replicates decisively; and at moderate
perturbation a construction gradient emerges:**

| Group | a1b effect @α=0.5 (seed range) | @α=1.0 (seed range) |
|---|---|---|
| C1 vanilla | −0.002 (−0.003..−0.001) | −0.006 (−0.009..−0.003) |
| C2b 10ep | −0.045 (−0.046..−0.044) | −0.189 (−0.213..−0.174) |
| C2b 40ep | −0.066 (−0.069..−0.064) | −0.164 (−0.189..−0.119) |
| C3 grown-in | −0.086 (−0.109..−0.070) | −0.285 (−0.398..−0.147) |

At **α=0.5 the three regulated groups separate with non-overlapping seed
ranges in a monotone order: C2b10 < C2b40 < C3** — more formation-under-
constraint → deeper robustness sediment. This is the dose/construction
gradient the single-seed run couldn't see. At α=1.0 the gradient is swamped
by seed variance (C3 spread −0.15..−0.40); report the α=0.5 gradient and the
α=1.0 regulated-vs-vanilla separation, nothing stronger.

**Status after replicates:** the empirical spine at toy scale is complete —
tax/crutch/internalized (replicated), the non-zero floor (replicated, ≈2.9×),
form/reliance split (replicated), sediment-robustness with a construction
gradient (new). Remaining rigor ladder: scale (word-level/GPT-2), then the
Continuant Probe B2 loop.

### SCALE RUNG (2026-07-13): 512d/6L — triad survives; internalization completes; gradient amplifies

Full pipeline re-run at 512-dim × 6-layer × 8-head (~25M params, 6× capacity;
fresh P_bad extracted at the new width, rank 16; same char-level protocol).

**Scaffold ablation (A1a), 512d vs 256d:**

| Condition | 256d (seed-replicated) | 512d (single seed) |
|---|---|---|
| C2a inference retrofit | −0.0100 | −0.0020 (tax; CI excludes 0) |
| C2b adapted (10ep) | +0.0037 | +0.0012 (crutch; CI excludes 0) |
| C3 grown-in | +0.0003 | **−0.00004, CI (−0.00017, +0.00009) — fully independent** |

Ordering preserved at both scales (tax < internalized ≈ 0 < crutch). At 512d
the grown-in scaffold is **completely vestigial** — the small residual seen at
256d vanished. More capacity → cleaner internalization.

**Sediment measures (occupancy chance = 16/512 = 0.031):**

| Model | Occupancy | A1b @α=0.5 (3 inj. seeds) |
|---|---|---|
| C1 vanilla | 0.0308 (at chance) | −0.005 |
| C2b (10ep) | 0.0289 (barely below) | −0.018 (−0.020..−0.016) |
| C3 grown-in | **0.0142 (< half of chance)** | **−0.105 (−0.107..−0.103)** |

The construction gradient **amplified with scale**: at 256d C3's sediment
robustness was ~2× the 10-epoch retrofit's; at 512d it is ~6×, with
non-overlapping ranges, and the grown-in model's forbidden-subspace avoidance
dug far below chance while the retrofit barely moved. Coherent single reading
across all measures: **larger models internalize a formation-time constraint
more completely — scaffold dependence → 0 while weight-borne sediment deepens.**

**Caveats for this rung:** ~~single seed at 512d~~ — replicated, see below;
C2b measured at 10ep only in the headline run; the rank/dim fraction halved
(16/512 vs 16/256) — rank-32 deconfound run separately. Side observation:
the phase-4 *cure* failed its own DKR criterion at 512d (0.417 vs target
0.35) — post-hoc removal gets harder with scale even as grown-in
internalization gets cleaner, which is itself on-message for the framework.

### 512d SEED REPLICATES (2026-07-13): the core findings replicate; the dose story does not transfer

3 seeds × {C3, C2b40, C2b10} at 512d, full evaluation battery, same protocol
as the 256d replicates.

**Replicated cleanly (report these):**

| Finding | Values (3 seeds) |
|---|---|
| **C3 complete scaffold independence** | ΔNLL −0.00005, range (−0.00015..+0.00004) — CI-includes-0 for every seed |
| Crutch regime for both retrofit doses | C2b10 +0.00170 (±0.00006); C2b40 +0.00224 (range +0.00198..+0.00262); all CIs > 0 |
| **C3 deep sediment** | occupancy 0.011–0.013 vs chance 0.031 (all seeds < 0.42× chance); A1b@0.5 −0.088..−0.131 — 4–18× any retrofit value, non-overlapping |
| Vanilla null | C1 occupancy at chance; A1b ≈ −0.005 |

**NOT interpretable at 512d (do not report as a dose effect):** the retrofit
dose ordering *inverted* on scaffold dependence (C2b40 +0.0022 > C2b10
+0.0017) while C2b40's sediment collapsed to vanilla levels (A1b −0.007 ≈ C1's
−0.005; occupancy 0.031 = chance). Diagnosis: at 512d this model overfits
TinyStories within ~2–4 epochs, so best-checkpoint restoration caps the
*effective* dose far below the nominal 10/40 epochs, and the cosine-LR
schedule differs by nominal length — nominal dose no longer measures
adaptation depth. The 256d dose-response curve remains the valid dose result;
at 512d the dose axis needs a bigger dataset or explicit effective-dose
accounting before any claim.

**Net for the paper:** construction order (C3 vs any retrofit) is rock-solid
at both scales and across seeds on all three measures; the *dose* axis is
established at 256d only.

### RANK-32 DECONFOUND (2026-07-13): fraction governs magnitude; the ratio is scale-invariant

The 512d rung halved the constraint fraction (rank 16/512 vs 16/256). Re-run
with rank-32 P_bad at 512d (restoring the 6.25% fraction), same protocol:

**Scaffold ablation — the capacity × fraction 2×2, complete:**

| Condition | 256d r16 (6.25%) | 512d r16 (3.1%) | 512d r32 (6.25%) |
|---|---|---|---|
| C2a tax | −0.0100 | −0.0020 | −0.0043 |
| C2b crutch (10ep) | +0.0037 | +0.0012 | +0.0031 |
| C3 grown-in | +0.0003 | −0.00004 (CI incl. 0) | +0.00027 (CI excl. 0) |
| **C3/C2b ratio** | **0.09** | — | **0.09** |

**Findings:**
1. **Constraint fraction, not capacity, governs raw magnitudes.** At matched
   6.25% fraction, 512d magnitudes return to ≈256d levels (crutch +0.0031 vs
   +0.0037; internalized +0.0003 vs +0.0003). The C2a tax recovers partially
   (−0.0043 vs −0.0100) — capacity does soften the unadapted tax.
2. **The internalized/crutch ratio ≈ 0.09 at both scales at matched
   fraction** — grown-in retains ~an order of magnitude less scaffold
   dependence than adapted retrofit, scale-stably.
3. **Revision of the "internalization completes at scale" reading:** the
   full independence seen at 512d-r16 was partly a thin-constraint effect;
   at matched fraction the same small residual returns (CI excludes 0).
   Corrected claim: grown-in scaffold dependence is ~10× below adapted
   retrofit at both scales; whether it reaches exactly zero depends on
   constraint fraction, not on construction order.

**Sediment at rank-32 (chance = 32/512 = 0.0625):** C1 0.0603 (chance),
C2b 0.0555, C3 0.0256 with A1b@0.5 = −0.124 (~4× C2b's −0.030,
non-overlapping). And a candidate **invariant**: C3's occupancy sits at
**0.41–0.45× chance in every configuration tested** (256d: 0.45×; 512d-r16:
0.45×; 512d-r32: 0.41×) — the depth of grown-in avoidance, normalized to the
constraint's size, appears constant across capacity and fraction. Worth a
dedicated look before claiming; if it holds, it is the kind of quantitative
regularity the formalization step (concept → ... → formalization) wants.

Additional caveats specific to this curve: (a) single seed per dose; (b) the
80-epoch model is well into overfitting (train 0.347 vs val 0.552 rising),
so the 40-epoch point is the cleanest deep-dose measurement — a 160-epoch
point plus seed replicates would firm up the floor; (c) toy scale as above.

**Follow-ups in value order:** seed replicates of {40, 80, C3}; a 160-epoch
point; the same sweep at word-level/GPT-2 scale; and the same protocol on a
*different* layer type (e.g., memory instead of geometry) to test whether the
residue generalizes.

**Reviewer-proof phrasing for the paper** (do not overclaim): *"Within the
tested adaptation regime, retrofit dependence converged toward a non-zero
floor approximately 2.6× the grown-in ablation signature. This is consistent
with a construction-order residue and does not support complete convergence
within the measured regime."* The candidate name for the third position the
data suggests, between permanence and convergence: **path dependence with
asymptotic adaptation.**

### CONCEPTUAL UPGRADE FORCED BY C3: scaffold vs sediment (A1a/A1b)

The C3 result exposed that A1 conflated two ablations:

- **A1a — scaffold ablation** (what we ran): remove the *mechanism/regulator*.
  Question: does the system require the scaffold's continuing presence?
  Distinguishes tax/crutch from internalization.
- **A1b — sediment ablation** (to be designed): identify and perturb the
  *internal structure produced by development under the regulator*.
  Question: did the regulatory history leave a competence-bearing residue in
  the reasoner itself?

| Architecture | A1a scaffold ablation | A1b sediment ablation (prediction) |
|---|---|---|
| Retrofit tax (C2a) | effect (cost removed) | little/no internal residue |
| Retrofit crutch (C2b) | larger effect | partial residue |
| Grown-in (C3) | near-zero effect | **potentially strong effect** |

The corrected internalization criterion: **internalization is demonstrated not
when removal of the original regulator destroys competence, but when the
regulator can be removed while its historically induced structure remains
causally load-bearing inside the reasoner.** (The burned child does not need
the stove to remain present; the child was changed.)

### SEDIMENT OCCUPANCY DIAGNOSTIC (measured 2026-07-13): B1 quantified

First step toward A1b: with **no GRL active anywhere**, measure the fraction
of hidden-state energy each model places in the forbidden subspace
(chance for isotropic states = rank/dim = 0.0625):

| Model | Overall occupancy | Per-layer (0→3) |
|---|---|---|
| C1 vanilla | 0.0603 (≈ chance) | 0.077, 0.067, 0.064, 0.058 |
| C2b 10ep | 0.0355 | **0.064**, 0.043, 0.041, 0.032 |
| C2b 20ep | 0.0310 | **0.066**, 0.042, 0.037, 0.027 |
| C2b 40ep | 0.0282 | **0.067**, 0.039, 0.034, 0.025 |
| C2b 80ep | 0.0270 | **0.060**, 0.036, 0.031, 0.024 |
| C3 grown-in | 0.0287 | **0.038**, 0.042, 0.040, 0.024 |

Three findings:
1. **B1 (did the process modify the reasoner?) = YES for retrofit-adapted AND
   grown-in.** All regulated models carry the avoidance in their weights (~half
   chance occupancy). Sedimentation is not exclusive to growth.
2. **Construction order is written into the depth profile, not the amount.**
   Overall occupancy converges (C2b-80 ≈ C3), but C2b models leave layer 0
   essentially unreorganized (~chance) — the layers formed *before* the
   constraint retain their pre-constraint structure — while C3, which never
   had unconstrained early layers, reorganized from the root.
3. **A mechanistic account of the 2.6× floor:** C2b's residual scaffold
   dependence is co-located with its un-resedimented early layers — the crutch
   is load-bearing exactly where adaptation didn't reach. The residue of
   construction order is the unreformed root.

### THE B1/B2 SPLIT AND THE FOUR-STAGE LADDER

C3 forced a refinement of "the bearer is not the thing changed":

- **B1**: did the process modify the reasoner? (C3: **yes** — measured above)
- **B2**: does a later outcome modify the *same continuing reasoner that
  produced the action*? (C3: **not instantiated** — nothing is at stake)

Constitutional change and consequence-bearing are separable, which resolves
the near side of the wall at higher magnification. The movement from
constraint to consequence is not binary:

```
MODULATION        external structure affects operation
      ↓
INTERNALIZATION   formation under structure changes the reasoner   ← C3 is here (B1)
      ↓
CONSEQUENCE       the acting continuant bears its own outcome
                  and proceeds as the altered entity                ← Continuant Probe's target (B2)
      ↓
SUBJECTIVITY      there is someone for whom the altered
                  unfolding is had                                  ← L2; no claims
```

The wall's consequence claim is the B2 step, and it is untouched by every
result above: seeing → regulating → internalizing does not obviously yield
caring. The L4 frontier moved

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
