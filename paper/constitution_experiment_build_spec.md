# Build Spec — The Constitution Experiment
## Paper #2 flagship: outcomes *constituted by* the bearer vs. *assigned to* it

*Alex Bone · harmonic-field-consciousness · drafted July 2026*

---

## Where this comes from

The Continuant Probe (Level 3) established the first half of a two-part claim. It showed that **bearer-identity becomes load-bearing exactly when outcomes stop being facts about the world and become facts of the bearer** — measured as a monotonic self-versus-yoked advantage that appears only when a private per-agent "taste" makes outcomes depend on *which* agent bears them.

But every outcome in that probe was still **assigned** to the bearer by the experimenter. The private taste was a fixed filter the design imposed. This is the near side of the wall's Level-3 face. The reference's exact words: the next question is *not* "can outcomes alter the actor?" (answered yes) but **"can outcomes be *constituted by* the actor rather than assigned to it?"**

That is the Constitution Experiment. It is the sharpest single experiment the framework now generates, and — importantly — the framework predicts it will *not* cross the wall. The value is in characterizing exactly how it falls short, at higher resolution than any probe so far.

---

## The distinction being operationalized

**Assigned outcome:** the experimenter defines a mapping from (agent, world-state) → outcome. The agent bears the outcome, and bearing it changes the agent (Continuant Probe result). But the *criterion of mattering* lives in the design, not in the agent. Remove the experimenter's mapping and nothing about the agent says which outcomes should matter to it.

**Constituted outcome:** the criterion of mattering is generated and maintained *by the agent's own ongoing operation*, such that what counts as a good-or-bad outcome for the agent is not specifiable independently of the agent's own history of being that agent. The stakes are not a filter applied to the agent; they are a standing fact the agent's continuation produces and depends on.

The wall sits in the gap between these. The framework's honest prediction: engineering can build increasingly self-referential outcome-criteria (constituted-*looking* stakes) but each will, on inspection, decompose into an assigned criterion at some level — a designer-set fixed point the self-reference bottoms out in. The experiment's job is to build the strongest possible constituted-outcome system and **locate the assigned residue** — the designer-set fixed point that self-reference cannot eliminate.

---

## Core design

A single continuing agent in a drifting world, as in the Continuant Probe. Three arms, holding architecture, world, and information identical:

- **Arm A — Assigned (replication baseline).** The Continuant Probe's winning condition: fixed private taste, experimenter-defined. Reproduces the known self-versus-yoked advantage. Anchors the comparison.
- **Arm B — Self-referential outcome (the constitution candidate).** The agent's outcome-criterion is a function of the agent's *own current parameters* — what counts as a good outcome at time *t* is determined by the state the agent has become by time *t*. The taste is not fixed and not experimenter-set; it is whatever the agent's own trajectory has made it. Outcomes feed back to alter parameters, which alter the outcome-criterion, which alters which outcomes register, and so on.
- **Arm C — Yoked self-reference (the critical control).** Identical to B, except the outcome-criterion is driven by a *different* agent's parameter trajectory. Same self-referential *form*, wrong bearer of the self-reference. This is the discriminator: if B beats C, self-reference is doing bearer-constitutive work; if B and C are indistinguishable, the "constitution" is idle and the self-reference is decorative.

The Arm B-vs-C contrast is the whole experiment. Arm A tells you the apparatus still works; B-vs-C tells you whether making the outcome-criterion self-generated rather than assigned buys anything a yoked copy can't have.

---

## The measurement that matters most

**Locate the assigned residue.** Arm B's self-reference cannot be turtles all the way down — somewhere there is a fixed point the designer set (the initial parameters, the update rule, the coupling function). The experiment's central deliverable is not "does B win" but **how deep the assigned residue sits and whether B's advantage over C survives as the residue is pushed deeper.**

Concretely, run B with the assigned fixed point at increasing depths of self-reference:
- *depth 0:* fixed taste (= Arm A).
- *depth 1:* taste is a function of current parameters; the *function* is assigned.
- *depth 2:* the function itself adapts by an assigned meta-rule.
- *depth k:* k levels of self-reference before hitting a designer-set floor.

Plot the B-vs-C advantage against depth. The framework predicts one of two shapes, and both are publishable:
1. **Advantage saturates at a floor > 0 that does not grow with depth.** Constitution-like self-reference buys a fixed amount over yoked, no more — the assigned residue is doing the real work at every depth, and deepening self-reference is cosmetic. This is the *deflationary-consistent* result and would be honest evidence that "constituted" reduces to "assigned with extra steps."
2. **Advantage grows with depth but with diminishing returns toward an asymptote.** Self-reference does progressively more bearer-constitutive work, approaching but never reaching a limit where the assigned residue vanishes. This is the *asymmetry-consistent* result and parallels the construction-order experiment's non-convergence — path dependence again, one level down.

Either shape is the finding. What the experiment must not do is report B beating A and call it constitution — that conflates "self-referential outcome" with "self-constituted stakes," which is the Level-3 version of the internalization overclaim.

---

## Instrumentation (reuse from existing repo)

- **World + drift + agent loop:** fork the Continuant Probe harness directly. The drifting-world and yoked-control machinery already exists and is validated.
- **Self-referential coupling:** new. A function mapping agent parameters → outcome-criterion, with a depth parameter controlling how many self-referential layers precede the assigned floor. This is the only genuinely new code.
- **Residue-depth sweep:** new but small. Loop the existing single-run harness across depth ∈ {0..k} and seed replicates.
- **Metrics:** self-vs-yoked advantage (reuse), plus a new "assigned-residue depth" axis. Log to PROBE_MATRIX with the same schema as the other probes so the result slots into the paper's empirical table.

Estimated new code: the coupling function and the depth sweep. Everything else is fork-and-configure. This is a smaller build than the construction-order experiment was.

---

## Predicted result and why it's worth running anyway

The framework predicts the Constitution Experiment does **not** cross the wall: some assigned residue always remains, because a manufactured system's fixed points are set by its manufacturer. Why run an experiment whose headline result is predicted?

Because the *shape* is not predicted, and the shape is the science. Result 1 (saturating floor) and Result 2 (growing-with-diminishing-returns) are both consistent with the wall standing, but they say very different things about how far constituted-looking stakes can be engineered. And there is a genuine third possibility the framework should be honest about being unable to exclude at toy scale: that B's advantage over C *keeps growing without apparent asymptote* within the tested range. That would not prove the wall was crossed — bounded experiments can't — but it would be the first result the framework couldn't comfortably absorb, and it would set the target for the next scale-up. A framework that can name the result that would trouble it is doing its job.

---

## How it lands in paper #2

Paper #1 (v0.2) ends on the reframed question: *can anything that begins at Level 7 come to be the bearer of its own outcomes rather than their assignee?* The Continuant Probe answered the bearing half. The Constitution Experiment attacks the constituting half. Paper #2's spine:

1. Recap the ladder (modulation → internalization → wall → consequence → subjectivity) as established result, not conjecture.
2. Situate the wall's Level-3 face: bearing (done) vs. constituting (open).
3. The Constitution Experiment: design, the assigned-residue-depth sweep, the B-vs-C discriminator.
4. Whichever shape resulted, with the honest bound.
5. Conclusion: the assigned residue as the measured signature of the wall at Level 3 — the designer-set fixed point that self-reference approaches but (predicted) does not eliminate.

If Result 1 or 2 holds, paper #2 is "the wall characterized at Level 3 with a dose curve." If the troubling third shape appears, paper #2 is "the first result that presses the wall, and the scale-up it demands." Both are strong. Neither requires crossing the wall to be worth publishing — which is the discipline the whole program is built on.

---

## One caution, carried from the last season

The naming lesson holds. Before any term from this spec enters a paper, verify it names actual implemented code in the repo. "Constitution candidate," "assigned residue," "residue-depth" are design language here; they earn paper-status only once they correspond to functions in the harness and columns in PROBE_MATRIX. LayeredFitness died because it skipped that check.
