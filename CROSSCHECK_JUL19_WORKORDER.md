# Cross-Check & Work Order — 2026-07-19

**Cross-checked:** the six documents in `Downloads/filesggg` (arXiv draft v0.2,
working notes v2, paper #2 outline, README, reference.html, constitution build
spec) against the measured program (PROBE_MATRIX) **and** against the
Constitution Experiment v0.4 results run today.

Three outputs: **(A)** corrections that must land before anything is published,
**(B)** what the *fixed* constitution experiment (v0.5) needs, **(C)** the
capstone experiment that ties the whole program together — currently missing.

---

## A. CORRECTIONS REQUIRED

### A1 — The v0.2 bearer-indexing claim is over-stated in four documents

**What changed.** The Constitution Experiment's Arm A (apparatus check, 40 seeds)
reproduced the v0.2 self-versus-yoked advantage on the agent's **own (assigned)
criterion** — `+0.909, 100% of seeds` — and found it **null on a bearer-neutral
yardstick** (raw resource collected): `+0.002, 55% of seeds`.

The effect is real and reproducible on the metric v0.2 used. But it is
**criterion-internal**: an agent altered by its own outcomes does better *by its
own assigned lights*, and no better at the world-fact task than one altered by a
donor's outcomes. This is the assigned residue surfacing one level earlier than
the spec anticipated.

**Locations to fix (exact):**

| # | File | Location | Current text |
|---|---|---|---|
| 1 | `life_before_language_arxiv_draft_v0_2.md` | §8.4, line 118 | "the self-versus-yoked advantage grows monotonically and becomes universal across seeds" |
| 2 | `README.md` | Continuant Probe → **Measured**, lines 87–90 | same claim |
| 3 | `reference.html` | Level 3 panel | same claim |
| 4 | `paper2_outline_assigned_residue.md` | §2.2 (background recap) **and** §5.1 (Arm A validity expectation) | same claim; §5.1 must say A reproduces *on own-criterion only* |
| 5 | `PROBE_MATRIX.md` | Continuant section | ✅ **already corrected** (committed today) |

**Replacement wording (drop-in):**

> …the self-versus-yoked advantage grows monotonically and becomes universal
> across seeds **when scored on the agent's own (assigned) criterion**. On a
> bearer-neutral yardstick — raw resource collected — the same comparison is
> null (+0.002, 55% of 40 seeds). The advantage is therefore **criterion-
> internal**: an agent altered by its own outcomes does better by its own
> assigned lights, and no better at the world-fact task. This sharpens rather
> than weakens the Level-3 claim, since the effect is exactly co-extensive with
> the bearer-indexed criterion — but no bearer-neutral performance gain is
> claimed.

Note the *philosophical* reading survives intact and arguably improves: the
advantage exists precisely to the extent that the criterion is the bearer's.
What must go is any implication of a world-measurable performance gain.

### A2 — The README's repository layout does not exist

`README.md` §"Repository layout" describes `probes/gabriels_horn/`,
`probes/spectral_geometry/`, `probes/holographic_rag/`, `probes/continuant/`,
`experiments/construction_order/`, `experiments/constitution/`,
`docs/reference.html`. **None of these paths exist.** Verified 2026-07-19.

Reality: the probes are in **four separate repositories** —
`horn-experiment`, `FTTF-Holographic-RAG`, `continuant-probe`, and the spectral
geometry inside `harmonic-field-consciousness/NanoGPT/spectral_transformer/`
(where the construction-order results also live, under `language_experiments*/`).

This violates the README's own **Discipline #4 ("Terms name code")** — it is the
LayeredFitness error one level up: a described structure with no referent. Either
restructure the repo to match, or rewrite the section to name the actual repos.
Recommend the latter (the separate repos are working well).

### A3 — Smaller items

- **Working notes v2** (line 45 table, line 90 provenance) are consistent; add
  the A1 qualification to the notes' overclaim watchlist.
- The **0.41–0.45 occupancy invariant** remains un-reproduced outside this
  lineage. The README's "Open call" is the right framing; keep it prominent.

---

## B. THE FIXED EXPERIMENT — Constitution v0.5

v0.4 was **spec-faithful** in design (criterion = f(own current parameters),
swept over levels of self-reference) and its apparatus check passed. But the
depth sweep supports **none of the three pre-registered shapes**, and it is
missing controls the paper #2 outline requires.

### B1 — Controls required by the outline §4.3 that v0.4 lacks

| Control | Outline's words | Status in v0.4 |
|---|---|---|
| **Parameter-matched assigned arm** at each depth | "the most likely reviewer objection… or the curve measures capacity, not constitution" | ❌ **missing — blocking** |
| **Drift-rate sweep (≥2 rates)** | "a result that holds only at one drift schedule is a result about that schedule" | ❌ missing (single rate 0.25) |
| **Yoking fidelity** — donor trajectory distribution-matched | "a yoked control accidentally easier or harder invalidates the discriminator" | ❌ unverified |
| **Per-seed reporting / universality** | "report per-seed, not just aggregate" | ⚠️ partial (ranges + positive-fraction only) |

### B2 — A confound the spec and outline did **not** anticipate (new, from v0.4)

> **The yardstick problem.** When the outcome-criterion is self-generated, there
> is no bearer-neutral way to score "doing well by its own lights," because the
> yardstick is part of what varies. Mean own-criterion value inflates from
> **1.67 at depth 0 to ~18 at depth 1** as the self-referential loop concentrates
> the criterion onto wherever the agent already is. **A self-constituting system
> can trivially satisfy itself.**

This is more fundamental than the capacity confound: it makes the primary metric
uninterpretable at depth ≥ 1, and it is why v0.4's sweep flips sign with seed
ranges of ±18. It should be added to the outline's §4.3 confound list, and it is
itself a small contribution — the formal echo, inside the toy, of why constituted
stakes are hard to *measure*, not merely hard to build.

**Three candidate fixes** (v0.5 must adopt at least one):

1. **Budget-constrained criterion** — hold the criterion's scale/entropy fixed so
   self-generated tastes cannot inflate their own scores. Cheapest; keeps the
   own-criterion metric comparable across depths.
2. **Persistence-based competence** — the agent must maintain a threshold to keep
   operating; competence = how long it continues. Uninflatable and
   bearer-neutral. *Caveat to state explicitly:* the threshold is designer-set,
   so it is itself an assigned residue — consistent with the framework's
   prediction, but it must be named, not sold as stakes.
3. **Common third-party yardstick** — score both arms by a fixed reference
   criterion. Comparable, but no longer "by its own lights," so it answers a
   weaker question.

**Recommendation:** implement (1) as the primary fix and (2) as a second
reported measure. Together they give one comparable within-criterion metric and
one uninflatable external one — which is exactly the dual-yardstick discipline
that made today's Arm A result legible.

### B3 — Until v0.5 lands

**No shape may be claimed.** Paper #2 §5 stays unwritten; §3 and §4 (the
distinction and the design) can be finalized now, per the outline's own writing
sequence.

---

## C. THE CAPSTONE — the experiment that ties it together

### The gap nobody has named yet

The program's central claim is:

> **Constitution and consequence-bearing are separable.** (README, "The central
> finding"; paper #1 §8.6; the refined ladder.)

But that claim currently rests on **two experiments in two different substrates
that never touch**:

- **Internalization (B1)** was measured in a *transformer* — real weights, a
  geometric regulator, no consequence loop anywhere in it.
- **Bearer-specificity (B2-adjacent)** was measured in a *numpy foraging toy* —
  a real within-life consequence loop, but the "agent" is a 50-number softmax
  vector, not a system that reasons.

So the separability claim is an **inference across substrates**, not a
measurement. A reviewer will say: *you showed internalization in a transformer
and bearer-specificity in a bandit; you never showed both, or their separation,
in one system.* That objection is currently correct.

### The capstone design

**One substrate exhibiting the full near-side ladder, measured with one
instrument set.** A small recurrent/neural agent in the drifting world, carrying:

| Measurement | Arms | What it establishes |
|---|---|---|
| **Internalization (B1)** | regulator grown-in vs retrofitted vs inference-only | occupancy + scaffold-ablation, exactly as the construction-order experiment — but now in an agent that *acts* |
| **Bearer-specificity (B2)** | self vs yoked outcome-alteration | the Continuant discriminator, in a system with real weights |
| **Constitution** | assigned vs self-referential criterion, depth-swept | the paper #2 discriminator |
| **Competence** | persistence-based (uninflatable) | fixes B2's yardstick problem and is bearer-neutral |

**Predicted result** (state before running): the system shows **internalization
(B1 = yes, occupancy below chance, scaffold removable)** *and* **no constituted
consequence (B2 = no, advantage criterion-internal or null on persistence)** —
demonstrating separability **in one place, with one instrument**, rather than
inferring it across two toys.

**Why it is the right capstone:** it converts the program's headline claim from
convergent-evidence-across-probes into a single measured fact; it retires the
substrate objection; it forces the persistence metric that v0.5 needs anyway; and
it makes paper #1's ladder and paper #2's residue two readings of one experiment.

**Cost:** the largest build in the program — a neural agent plus both batteries.
Days, not hours. Worth it only after v0.5 settles the yardstick.

---

## Priority order

1. **A1 + A2 corrections** — before anything is shown to anyone. Cheap, and A1
   corrects a published claim.
2. **v0.5 constitution** — add parameter-matched arm, drift sweep, yoking
   fidelity, and the budget-constrained + persistence yardsticks. Then, and only
   then, claim a shape.
3. **Capstone** — after v0.5. It is the difference between "five probes point
   the same way" and "the central claim is demonstrated in one system."
4. **Independent reproduction of the 0.41–0.45 occupancy number** — the open
   call. Still the single most load-bearing unreplicated result.
