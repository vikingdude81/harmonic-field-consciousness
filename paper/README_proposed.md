# harmonic-field-consciousness

Empirical probes for the **Life Before Language** research program: measuring how far
engineering reaches toward the developmental interface of consciousness, and locating
precisely where it stops.

This repository is not an attempt to build a conscious system. It is an attempt to
characterize a boundary by building systems that approach it from above and reporting,
in detail, how each one falls short.

---

## The thesis in one paragraph

Debates about AI and consciousness compare humans and machines at their *endpoints* —
fluent language, symbolic reasoning — rather than their *origins*. Human consciousness
does not begin with language; it begins with a pre-rational developmental interface
(appetite, attachment, sensation, consequence, temporal continuity) through which
reasoning later emerges as the act of a self. Language models invert this: they begin at
the symbolic layer with no developmental substrate beneath it. Reorganizing the debate
around origins separates two questions that are routinely conflated — the **engineering
gap** (missing memory, loops, sensing; bridgeable) and the **metaphysical gap** (whether
there is a subject for whom unfolding is a life; possibly not). This repo prosecutes the
first and characterizes the second.

**AI begins with language without life. Human consciousness begins with life before language.**

---

## The seven-level developmental interface

| Level | Name | Human | Engineered systems |
|---|---|---|---|
| 1 | Providential order | ground of the whole | framework posit — unmeasured |
| 2 | Primary consciousness | the subject | framework posit — unmeasured |
| 3 | Innate orientation | hunger, fear, attachment | **Continuant Probe** |
| 4 | Sensory interface | body, pain, proprioception | **LGI / spectral geometry** |
| 5 | Memory & development | personal history | **Holographic RAG / FTTF** |
| 6 | Will & agency | action from a formed interior | agentic loops |
| 7 | Reason & language | late achievement | **Gabriel's Horn** — native starting point |

Humans grow **bottom-up** (2–4 → 7 over years). Engineered systems are instantiated at 7
and everything below is **retrofitted** downward. The wall — where engineering stops
reaching — sits between Levels 4 and 3.

---

## Probes

Each probe implements the *formal structure* of one fragment of the developmental
interface. The scientific content is in how each falls short.

### Gabriel's Horn — Level 7
**Fragment:** internal temperature regulation (1/x entropy-decay schedule at inference).
**Measured:** applying the schedule to a small instruction model shaped output with no
measurable effect on competence on unrelated tasks (ablation effect within confidence of
zero).
**Shortfall:** pure modulation, and *architecturally* so — an inference-time decoder never
touches the weights, so removing it restores the bit-identical model. Decoding-layer
regulation shapes what the system does and cannot in principle change what it is.

### LGI / spectral geometry — Level 4
**Fragment:** structural perception. A spectral regularizer projecting a forbidden
subspace out of a model's hidden states — conditions of the system's own state space,
registered within its operating dynamics.
**Role:** substrate for the construction-order experiment (below).
**Shortfall:** the metric is designer-set; violation costs the system nothing it bears.
Constraint, not consequence.

### Holographic RAG / FTTF — Level 5
**Fragment:** superposed memory — the closest formal analogue to sedimentation available.
**Measured:** a sharp dissociation. The memory layer's *presence* significantly degrades
competence on unrelated tasks (retrieved context taxes attention). Corrupting its
*content* — wrong retrievals, no notification — has no measurable effect.
**Shortfall:** **presence-tax with content-indifference** — the exact inverse of
constitutive memory, where presence is transparent and corruption is devastating. The
system is sensitive to *having* the service and indifferent to whether it is *right*.
**Qualification:** sedimentation is real only for the compressed routing index, which has
a measured capacity limit biological memory need not share.

### Continuant Probe — Level 3
**Fragment:** consequence borne by a continuing individual. A single agent in a drifting
world whose own outcomes update the parameters it acts with, against a **yoked control**
altered by a different agent's outcomes — same rule, same information, wrong bearer. The
evolutionary runtime serves as the control case (consequence on a lineage, not within a
life).
**Measured:** when outcomes are facts about the world, self-driven and other-driven
alteration are indistinguishable — bearer-identity is idle. As outcomes are made to depend
on the individual (a fixed private "taste" filtering what each agent experiences), the
self-versus-yoked advantage grows monotonically and becomes universal across seeds **when
scored on the agent's own (assigned) criterion**. On a bearer-neutral yardstick — raw
resource collected — the same comparison is null (+0.002, 55% of 40 seeds). The advantage
is **criterion-internal**: the agent does better by its own assigned lights, and no better
at the world-fact task. This sharpens the Level-3 claim (the effect is co-extensive with
the bearer-indexed criterion) but claims no bearer-neutral performance gain.
**Shortfall:** the private taste is *assigned*. Outcomes are borne by the agent, not
constituted by it. This is the wall's Level-3 face and the target of the next experiment.

> Note: this probe was formerly called "Oracle Prime," and an earlier draft referenced a
> component called "LayeredFitness." That term named no actual implementation and has been
> removed. Terms in this repo correspond to running code or they do not appear.

---

## Flagship: the construction-order experiment (Levels 4 × 6)

The same regulatory geometry introduced at **three different points in a model's
formation**, holding architecture, geometry, and data fixed:

| Construction point | Regime | Behavior on removal |
|---|---|---|
| Attached at inference only | **tax** | removing it *improves* competence — pure overhead |
| Fine-tuned into a finished model | **crutch** | the model leans on it; removal costs competence |
| Present from the first training step | **internalized** | removal costs almost nothing |

With the regulator entirely removed, the grown-in model's states occupy the forbidden
subspace at **0.41–0.45 of the chance rate** — stable across two model sizes, two
constraint fractions, two tokenizations, and seed replicates — while an unregulated model
sits at chance.

**Retrofit asymmetry, measured.** Fine-tuning the retrofit for increasing durations reduces
its dependence toward a floor, but in every regime tested the adapted retrofit retains
roughly **threefold** the grown-in model's scaffold dependence, and no adaptation within
range closes the gap. Neither permanence nor convergence: **path dependence with asymptotic
non-convergence.**

---

## The central finding

The framework originally posed a binary: a constraint is either *modulation* (applied from
outside) or *answerability* (the symbolic layer genuinely conditioned by lower layers).
The construction-order experiment shows the binary is too coarse. Deep formation under a
constraint does not make the constraint more necessary — it makes it nearly removable
**while its discipline persists in the weights**. That is a third regime.

```
modulation  →  internalization  →  ⟨ THE WALL ⟩  →  consequence  →  subjectivity
```

**Constitution and consequence-bearing are separable.** The grown-in model is the first
concrete case of the former without the latter. It was changed by its formation, and its
state-space geometry embodies a discipline it was formed under — and nothing mattered to
it.

The wall did not move when internalization was found above it. The frontier of what
engineering reaches moved, and the wall was still there.

---

## Discipline

Rules this project holds itself to, stated so that violations are visible:

1. **Never let Workstream 1 results masquerade as Workstream 2 conclusions.** Building the
   formal structure of a level is not occupying that level. The internalization result is
   the most overclaim-prone finding here: a constraint persisting in the weights after its
   scaffold is removed *looks like* a system that made the discipline its own. It is
   constitution without consequence-bearing, and only conflating those reads it as progress
   toward subjectivity.
2. **State results at the resolution the evidence supports.** All findings are small-scale,
   from one research lineage, without independent replication. "Stable across tested
   variations," never "universal."
3. **One probe is not the class.** The presence-tax signature belongs to this holographic
   implementation; other memory architectures could differ.
4. **Terms name code.** Anything appearing in a paper corresponds to an implemented
   function and a PROBE_MATRIX column, or it does not appear.
5. **Levels 1–2 are posits, not targets.** No experiment here makes or needs a claim about
   them.

---

## Next

**The Constitution Experiment.** The Continuant Probe answered *bearing*; the open half is
*constituting* — can outcomes be constituted by the actor rather than assigned to it?
Three arms (assigned / self-referential / yoked self-reference) with a sweep over how deep
the designer-set fixed point — the **assigned residue** — is pushed. The framework predicts
a residue always remains; the science is in the shape of the approach. The pre-registered
result that would trouble the prediction: advantage rising without apparent asymptote
inside the tested range.

**Open call.** The 0.41–0.45 occupancy number is the most load-bearing single result here
and has not been reproduced outside this lineage. Independent reimplementation of the same
geometry, at any scale, would be the most useful contribution an outside reader could make.

---

## Papers

- **Paper #1** — *Life Before Language: Separating the Engineering Gap from the
  Metaphysical Gap in Debates on Machine Consciousness* (v0.2, framework + first empirical
  season).
- **Paper #2** — *The Assigned Residue: Bearing and Constituting in Artificial
  Continuants* (in preparation).

## Repository layout

The probes live in **separate repositories**; this repo is the hub that holds the
framework documents and the measurement record.

| Repository | Contents |
|---|---|
| `harmonic-field-consciousness` (this repo) | `PROBE_MATRIX.md` — every reported number traces here; framework docs (ecosystem map, chart↔evidence alignment, paper revisions); `experiments/answerability_battery/` — the shared A1/A2 instrument; `consciousness_circuit/` — the measurement package |
| `harmonic-field-consciousness/NanoGPT` (submodule) | `spectral_transformer/` — Level-4 forbidden-subspace regularizer **and** the construction-order flagship (`run_construction_order.py`, `run_sediment_occupancy.py`, results under `language_experiments*/`) |
| `horn-experiment` | Level 7 — entropy-decay decoding, unit tests, battery wiring |
| `FTTF-Holographic-RAG` | Level 5 — superposed holographic memory |
| `continuant-probe` | Level 3 — bearer-indexed agent + yoked control; the Constitution Experiment |

*(Per Discipline #4, this section names paths that exist. An earlier version
described a single-repo `probes/` layout that was never built.)*

---

*Alex Bone · Independent Researcher · Long Beach, CA*
