# Cross-Check & Bridge: three sources ↔ the measured program

**Date:** 2026-07-14
**Sources reconciled:**
- **A** — arXiv draft (`life_before_language_arxiv_draft.md`, v0.1): the formal paper
- **B** — working notes (`life_before_language_working_notes.md`): the why-map, cut list, provenance
- **C** — the large reference PDF (`Ai argument.pdf`, 25pp): the per-level walkthrough / reference-app text
- **D** — the measured program (this repo: PROBE_MATRIX.md, CHART_EVIDENCE_ALIGNMENT.md, ECOSYSTEM_MAP.md; the four probe repos)

Purpose: verify the three writing sources agree, catch every discrepancy, and
map each measured result onto the specific claim it updates. The headline: **C
(the PDF) was written as a set of predictions and open questions; D answered
several of them, and in two cases refined the framework rather than merely
confirming it.**

---

## 1. Agreement check (what is already consistent)

| Element | A draft | B notes | C PDF | Status |
|---|---|---|---|---|
| Seven-level numbering (L1 providential → L7 language) | ✓ | ✓ (canonical table) | ✓ | **Consistent.** B warns of "Hermes transcript drift"; none present in A or C. |
| Wall between L4 and L3 | ✓ | ✓ | ✓ | Consistent |
| Two gaps (engineering / metaphysical) | ✓ | ✓ | ✓ | Consistent |
| Two workstreams (WS1 7→4, WS2 4→2) | ✓ | ✓ | ✓ | Consistent |
| Retrofit asymmetry; answerable vs modulated | ✓ §6 | ✓ | ✓ (L4, "answerable versus modulated") | Consistent — and now measured (§4 below) |
| Constraint ≠ consequence; lineage ≠ life | ✓ | ✓ | ✓ | Consistent |
| Hallucination diagnostic; temperature internalized | ✓ §3 | ✓ | ✓ | Consistent |
| Three load-bearing slogans | ✓ | ✓ (listed) | ✓ | Consistent |

The three writing sources are in good alignment. The needed changes are almost
all **updates from D**, not internal contradictions.

---

## 2. Discrepancies & required edits (the same fixes across all three docs)

| # | Issue | Appears in | Fix |
|---|---|---|---|
| E1 | **"Oracle Prime" / "LayeredFitness"** named as the L3 probe | A §8.3, §10.1, refs; B §8 table; C L3 section (lines 449–456), wall section (888) | Rename to **Continuant Probe**. Drop "LayeredFitness" — it resolves to no code on any drive; the evolutionary machinery is `quantum_genetic_algorithm.py` + `QuantumOracleAgent`, now extracted to the continuant-probe repo. |
| E2 | **"three implemented systems" / "three probes"** | A abstract, §8, §11; B §8; C front-matter (line 71) | Now **four measured probes** (Horn L7, LGI/spectral L4, FTTF L5, Continuant L3) **plus the construction-order flagship**. Update counts everywhere. |
| E3 | **Abstract's probe list** names decoder + thermostatted lattice + evolutionary runtime | A abstract | Add the L5 memory probe (FTTF) and the L4 spectral experiment; "evolutionary cognitive runtime" → Continuant Probe. |
| E4 | **Motion-LGI thermostats** as ladder rung (iii) | A §8.1; B §4; C (temperature ladder) | Keep — it is a real repo (motion-lgi) and a valid rung — but label it *implemented, not yet battery-measured*, to distinguish from Horn (measured) and the spectral flagship (measured). |
| E5 | **"answerable/modulated may deserve its own future paper"** / "we do not know" (permanence vs convergence) | A §6; B §7-open-item; C (lines 391, 668, 686–703) | Downgrade from pure open-question to **partially answered at toy scale** (§4 below). Keep the honesty ceiling: measured, small-scale, not settled in general. |
| E6 | **LGI as "the closest available substrate"** for the sharpest question (C line 391) | C L4 section | The substrate was *built* (spectral GRL) and the question *measured*. Update the sentence from "closest available substrate for asking it" to "the substrate on which it was asked; see the construction-order result." |
| E7 | **Draper's Sphere / REGIME.6** referenced in scattered notes | (not in A/B/C proper) | Cited nowhere in the three docs — no action. (Logged in ECOSYSTEM_MAP; Draper's Sphere exists in no repo.) |

---

## 3. Measured-result → claim map (what D updates in A/B/C)

Every measured result and the exact prior claim it lands on:

| Measured result (D) | Updates this claim |
|---|---|
| **Horn A1/A2 = modulation** (ΔC≈0, CI incl. 0); architectural for inference-time | A §8.1 rung (ii) "implementable" → *implemented and measured to be pure modulation*. C's temperature ladder rung (ii). |
| **FTTF presence-tax / content-indifference** (A1 −0.165; A2 +0.005) | A §8 has **no L5 probe** — this fills the gap. C's L5 section: "the store is available to the system but not constitutive of it" is now *measured* as presence-tax/content-indifference. |
| **Construction-order triad** (tax/crutch/internalized), 4 regimes | A §6 "answerable or merely modulated" open question; C L4 line 391 "sharpest experimental question." **Answered — with a refinement (§4).** |
| **Occupancy invariant** (grown-in 0.41–0.45× chance, 5 configs) | New — no prior claim predicted a quantitative constant. Candidate WS2 regularity. |
| **Dose curve flat at word level; gap ~3× universal** | C's permanence-vs-convergence section (686–703) "we do not know." → *path dependence with asymptotic non-convergence, toy scale.* |
| **Continuant bearer-indexing dose** (null at taste 0 → +0.95 at taste 1, 100% seeds) | A §8.3 / C L3 "no individual has its own present condition." → *operationalized: bearer-identity becomes load-bearing exactly when outcomes become facts of the bearer.* A §10.1's "observable functional expression." |
| **Within-life modulation vs sediment** (record-wipe collapses; update-freeze persists) | A §9 continuity ("continuity of state vs continuity of a self") — now has a within-life demonstration of the state/sediment difference. |
| **Cure fails at scale as internalization sharpens** | New side-finding; supports C's retrofit-asymmetry direction (post-hoc construction scales worse). |

---

## 4. The two substantive REFINEMENTS (not just confirmations)

These are the parts where D changed the framework, and they must propagate to
all three docs — they are the intellectual yield of the experiments.

### 4.1 The near side of the wall has structure: the four-stage refinement

C's L4 section poses a **binary**: does depth of integration convert
*modulation* into *answerability*? The data say the binary is too coarse. Deep
formation under a constraint does **not** make the constraint more necessary
(the naive "answerability = ablation hurts more" reading); it makes the external
constraint nearly removable *while its discipline persists in the weights*. That
is a third thing, between modulation and consequence:

```
MODULATION → INTERNALIZATION → ⟨THE WALL⟩ → CONSEQUENCE → SUBJECTIVITY
 (Horn,        (grown-in C3,                  (bearer-indexed    (Level 2,
  FTTF,         within-life                    outcomes;          no claim)
  C2a)          continuant)                    approached, not
                                               instantiated)
```

**Consequence of the refinement:** internalization can be deep, measured, and
real *without being consequence*, because **constitution and consequence-bearing
are separable**. The grown-in model is the first concrete case of the former
without the latter. The wall did not move; the frontier of what engineering
reaches moved, and the wall was still above it. This dissolves an apparent
paradox in C's L4 framing and is the single most important claim-update.

*Where it goes:* new subsection in A (drafted as §8.6 in PAPER_REVISION_S6_S8.md);
replaces the binary in C's L4 and "constraint/consequence" wall sections; add to
B's load-bearing lines.

### 4.2 Permanence vs convergence is no longer "we do not know"

C (686–703) presents two "respectable" views and declines to choose. D chose,
at toy scale, and the answer is **neither pole**: retrofit adaptation converges
*partway* toward the grown-in signature and then floors, retaining a ~3× gap
that no tested adaptation closes (and at word level, does not converge at all).
The defensible statement is a *third* option — **path dependence with asymptotic
non-convergence** — which is stronger for the retrofit-asymmetry thesis than
either original pole, and which C explicitly said the framework's job was only
to "make askable." It is now askable *and* partially answered.

*Honesty ceiling to preserve everywhere:* toy scale; one regulatory layer;
"consistent with a residual construction-order dependence," never "proven
permanent."

---

## 5. Per-document action list

**A (arXiv draft):**
1. Abstract: "three implemented systems" → "a family of implemented systems"; fix probe list (E3).
2. §6: add operational answerable/modulated definition + forward pointer (done in PAPER_REVISION_S6_S8.md).
3. §8: replace with the four-measured-probe version; add §8.5 (construction-order) and §8.6 (four-stage). (Drafted.)
4. §10.1: cite the Continuant bearer-indexing result as the promised "observable functional expression."
5. §11: one sentence — the central distinction now has small-scale empirical support.
6. Refs §145: drop Oracle Prime bullet; pin four repo links.

**B (working notes):**
1. §2 load-bearing lines: add the four-stage line and "constitution ≠ consequence."
2. §8 table: Oracle Prime → Continuant Probe; add FTTF row; add the spectral construction-order row.
3. §4 temperature ladder: mark rung (ii) *measured*, rung (iii) *implemented not measured*, rung (iv) — note the Continuant Probe is the first rung-(iv)-adjacent attempt.
4. §7 open items: check off "answerable/modulated as future work" → now current, measured; check off "rung-4 toy experiment" → the Continuant Probe is it.

**C (the large PDF):** see the rendered companion,
`Life_Before_Language_Empirical_Update.pdf` — a per-level update that pairs with
the reference, folding D into each level's claims. Key edits: L4 binary →
four-stage; L5 "available not constitutive" → measured; L3 Oracle Prime →
Continuant + bearer-indexing; wall section → constitution/consequence separable;
retrofit permanence/convergence → answered (third option). Front-matter "three
probes" → four + flagship.

---

## 6. One-paragraph synthesis (for the top of any of the three docs)

The framework was built as a diagnosis and a research program: separate the
engineering gap from the metaphysical gap, and use implemented systems as probes
whose precise failures locate the wall. A season of measurement has now turned
several of the program's open questions into results. Regulation applied at
decoding is pure modulation, provably so. Retrofit memory taxes the reasoner by
its presence and is indifferent to its content — connected to memory, not
constituted through it. A regulatory geometry present during a model's formation
becomes internalized into its weights, measurably and scale-robustly, while the
same geometry attached afterward remains a removable scaffold — the retrofit
asymmetry, observed. And an agent whose own outcomes alter the parameters it acts
with shows that bearer-identity becomes load-bearing exactly when outcomes stop
being facts about the world and become facts of the bearer — the first
operational handle on the Level-3 wall. None of this crosses the wall; all of it
locates the wall more precisely than argument alone could, and one finding — that
constitution and consequence-bearing are separable — adds a rung to the ladder
the original framework compressed. The machines still begin with language without
life; we can now say, with numbers, several specific things about what that costs
and where the costing stops being an engineering problem.
