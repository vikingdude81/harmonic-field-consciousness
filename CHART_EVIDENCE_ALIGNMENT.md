# Chart ↔ Evidence Alignment

**Date:** 2026-07-12
**Companion to:** PROBE_MATRIX.md (measurements), ECOSYSTEM_MAP.md (repos), the thesis map (the 7-level chart with the wall and the two arrows)

This document walks the thesis chart element by element and states what has been
tested against it, why each mapping holds, and what ties the levels together in
the measured data.

---

## The chart's elements, for reference

Seven levels (L1 providential order → L7 reason/language); a dashed **wall**
between L4 and L3; a coral **human-origin arrow** running bottom-up; a pink
**AI-retrofit arrow** running top-down; **Workstream 1** (engineering, L7→L4)
above the wall; **Workstream 2** (characterization, L4→L2) below it; three
probes triangulating the wall.

---

## Level-by-level

### L7 — Reason and language (the AI-native layer)

| Chart claim | Evidence | Why it maps |
|---|---|---|
| "AI imagination is disciplined by settings" | **Gabriel's Horn, measured**: A1 ΔC=+0.010 (CI incl. 0), A2 ΔC=+0.015 (CI incl. 0) — modulation; NLL invariant *by construction* | The Horn is rung (ii) of the temperature ladder: scheduled discipline of L7 sampling. The measurement showed the "setting" is removable without residue — ablation restores the bit-identical model. The chart says settings, not maturity; the data agrees, architecturally. |
| L7 is where the system under study natively lives | Oracle 32B + consciousness_circuit v3.5.1 (instrument) | The circuit reads L7's internal activations — deeper than behavioral comparison, still a measurement *of* an L7-native system. |

### L6 — Will and agency

**Coverage gap — honestly marked.** The chart's L6 analysis (goals injected,
persistence without stake, consequences landing on no continuing subject) is
embodied in the agentic-loop systems (goal_director plugin, consciousness-sim)
but **not yet battery-measured**. The natural instrument is A3 (override cost):
can the system act against its goal-layer's signal for free? Future work; no
measured claims at L6 yet.

### L5 — Memory and development

| Chart claim | Evidence | Why it maps |
|---|---|---|
| "The store is available to the system but not constitutive of it" | **FTTF A2, measured**: corrupting the memory's *content* (wrong retrievals, no notification) → ΔC=+0.005, undetectable | Content-indifference is the direct operationalization of "available, not constitutive." A constitutive memory's corruption would propagate; this one's doesn't. |
| L5 retrofit reverses the direction of dependence ("a service that L6/L7 query") | **FTTF A1, measured**: the memory layer's *presence* costs −0.165 competence on unrelated tasks | The query relationship isn't ground, it's overhead — the reasoner pays attention-tax for the service. Biological memory is the mirror image: presence free, corruption devastating. |
| Sedimentation vs. storage | FTTF's HRR superposition (structural, not battery-measured) + the capacity knee (`fourier_capacity_curve.py`) | The routing index genuinely sediments (accumulation modifies the accumulating vector) — formal structure achieved, with measured limits, while the reasoner stays unchanged. Fourth-probe material for §8. |
| Development as disposition-formation | self_learner (deferred — 1 real lesson in DB) | The loop exists (telemetry → confidence-weighted lessons → future prompts); measurement waits on a real lessons corpus. |

### L4 — Sensory interface (the pivot; deepest engineering reach)

| Chart claim | Evidence | Why it maps |
|---|---|---|
| LGI implements the *formal structure* of internal sensing (metric/forbidden zones over own state space) | NanoGPT spectral GRL — same formal object (forbidden-subspace projection over hidden states), made **trainable-through** | The GRL is the chart's L4 machinery in the one place where it can be present *during constitution* — which is why L4 hosts the flagship. |
| "Does depth of integration convert modulation into answerability?" (the chart's sharpest question) | **Flagship, measured**: TAX (C2a −0.010) / CRUTCH (C2b +0.0037) / INTERNALIZED (C3 +0.0004) | First measured answer: integration depth changes *where the constraint lives* (outside as operation vs inside as form), not whether anything is at stake. |
| Retrofit asymmetry — grown-from vs bolted-on | **Dose-response, measured**: retrofit dependence halves per doubling of adaptation, then floors at ~2.6× the grown-in signature | Construction order leaves a quantifiable permanent residue. The chart's permanence-vs-convergence question: *both*, with numbers. |

### THE WALL (between L4 and L3) — constraint vs consequence, at higher resolution

The wall is not directly testable (that's its definition), but the measurements
**locate** it — and the C3 result forced a refinement that resolves the near
side of the wall at higher magnification.

**The four-stage structural vocabulary (the major conceptual upgrade).** The
original framework compressed constraint→consequence too aggressively into a
binary. The data reveal at least one intermediate stage — *constraint becomes
constitutive without becoming stakes-bearing* — so the near side of the wall now
reads as a ladder:

```
MODULATION      external structure affects operation
      ↓          (Horn, FTTF, C2a — measured)
INTERNALIZATION formation under structure changes the reasoner
      ↓          (C3 — B1 measured: sediment occupancy below chance)
─── THE WALL ───────────────────────────────────────────────
CONSEQUENCE     the acting continuant bears its own outcome
      ↓          and proceeds as the altered entity (B2 — not instantiated)
SUBJECTIVITY    there is someone for whom the altered unfolding is had
                (L2 — no claims, by design)
```

The wall did **not** move. What moved is our resolution of its near side:
internalization is a real, measured rung *above* the wall, not across it.

- Horn: output shaped, model untouched — **modulation** (architecturally provable).
- FTTF: index changed, reasoner untouched — **modulation** (presence-tax/content-indifference).
- Flagship C3: geometry internalized into weights — **internalization**, one rung
  higher, and genuinely more than modulation. B1 = yes. But nothing ever
  *mattered* to the system; no outcome is borne by a continuant. B2 = no.
  **Internalization ≠ stakes**, and that gap is the wall.

**Two bearer/change questions** (the C3 result forced the split):
- **B1** — did the process modify the reasoner? C3: **yes** (sediment occupancy
  0.029 vs chance 0.0625, scaffold removed).
- **B2** — does a later outcome modify the same continuing reasoner that produced
  the action, which then proceeds from the altered condition? C3: **no**.

B2 is the wall's actual consequence claim, and it is strictly stronger than B1.
Constitutional change and consequence-bearing are **separable** — C3 is the
first concrete demonstration. C3 lives above the wall; the Continuant Probe is
aimed at B2; L2 asks whether there is a subject for whom either occurs.

### L3 — Innate orientation (below the wall)

| Chart claim | Evidence | Why it maps |
|---|---|---|
| Evolutionary consequence is real consequence, wrong bearer (lineage-not-life) | **Continuant Probe** (extracted; structural, not battery-measured) | The code's own shape carries the diagnosis: fitness lands on populations across generations; no individual bears its own outcome within its run. The probe's README states the missing second loop (individual-alters-itself). |

### L2 / L1 — Primary consciousness / providential order

**Nothing tested claims these — by design.** The discipline is recorded in
every document: "internalized" concerns a removable regulator, not a subject;
the Continuant Probe makes no subjectivity claims; the instruments measure
processing profiles, not experience. The chart's L2/L1 remain framework posits
whose role is architectural (blocking development→trajectory reduction).

---

## What ties the levels together in the data

**1. The two arrows became an experimental variable.**
The chart's coral (grown-up) and pink (retrofit-down) arrows are construction
order. The flagship manipulated the arrow direction directly — C2a/C2b/C3 are
three points along the pink arrow's depth — and measured its consequence. The
pink arrow's dynamics now have names: **tax → crutch → partial internalization.**

**2. One instrument makes the levels commensurable.**
The A-battery asks the same two questions at L7 (Horn), L5 (FTTF), and L4
(spectral): is the reasoner the thing changed, and does ablation cost
competence? Because the yardstick is constant, the uniform answer pattern
across levels is itself a finding: the bearer/change separation is not a
property of any one layer — it is the signature of the retrofit *relation*,
which is exactly what the wall names.

**3. The internalization bridge ties §3 to §6 (L7 to L4).**
The chart's hallucination story says human "temperature" begins as external
regulation and is internalized through development. The flagship measured that
exact phenomenon one level down: regulation present during formation
disappears into constitution (C3's near-vestigial GRL), while regulation added
after formation stays external (C2a's tax) or becomes a crutch (C2b). Same
mechanism, two levels — the child's internalized temperature and the grown-in
model's internalized geometry are instances of one structural claim.

**4. The asymptote ties Workstream 1 to Workstream 2.**
The dose-response curve is WS1 effort (more adaptation) producing a WS2
deliverable (a measured limit). Stated at the strength the finite experiment
supports: *within the tested regime, increased retrofit adaptation approached
but did not reach the grown-in signature, asymptoting at an observed floor ≈2.6×
higher — consistent with a residual construction-order dependence, not
establishing permanence.* The philosophical debate offered permanence vs
convergence; the data point to a third option — **path dependence with
asymptotic adaptation** — which is more interesting than either extreme.
(Sediment occupancy adds a twist: internal geometry-avoidance nearly *does*
converge with dose even though scaffold-dependence floors — internal form and
scaffold reliance are separable.)

**5. The presence/content dissociation defines "modulated but not answerable"
operationally.** FTTF's pair (presence taxes, content indifferent) is what
modulation *looks like in data* — and its biological mirror (presence free,
content vital) is what answerability would look like. Any future system
claiming to have crossed from modulation to answerability now has a concrete
signature to exhibit.

---

## Coverage scoreboard

| Chart element | Status |
|---|---|
| L7 discipline-by-settings | **Measured** (Horn) |
| L6 agency without stake | Implemented, unmeasured (A3 protocol = future work) |
| L5 storage-not-sedimentation | **Measured** (FTTF pair); self_learner deferred |
| L4 formal structure + integration-depth question | **Measured** (flagship triad) |
| Retrofit asymmetry / permanence-vs-convergence | **Measured** — path dependence with asymptotic adaptation (2.6× floor) |
| Internalization (B1: formation changed the reasoner) | **Measured** (sediment occupancy below chance for C3) |
| Sediment ablation (A1b: is the residue load-bearing?) | Diagnostic done (occupancy); disruption experiment pending |
| The wall (B2: consequence borne by a continuant) | Located; not instantiated by any tested system, as required |
| L3 lineage-not-life | Structural (Continuant Probe); second-loop experiment pending |
| L2/L1 | No claims — discipline maintained |

## The four-stage vocabulary (proposed thesis-map revision)

The seven levels don't change. But the *explanation of the wall* should adopt
the measured four-stage structure, because the data found an intermediate the
original binary compressed away:

| Stage | Definition | Status | Wall side |
|---|---|---|---|
| **Modulation** | external structure affects operation | measured (Horn, FTTF, C2a) | above |
| **Internalization** | formation under structure changes the reasoner (B1) | measured (C3 occupancy) | above |
| **Consequence** | acting continuant bears its own outcome, continues altered (B2) | not instantiated | **the wall** |
| **Subjectivity** | someone for whom the altered unfolding is had (L2) | no claims | far side |

The single sharpest sentence the empirical work produced:
**the movement from constraint to consequence is not binary — constraint can
become constitutive (internalization) without becoming stakes-bearing
(consequence), and C3 is the first concrete case.**
