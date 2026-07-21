# Paper #2 — Full Outline
## *The Assigned Residue: Bearing and Constituting at the Level-3 Face of the Wall*

*Alex Bone · Independent Researcher · outline drafted July 2026*
*Companion to: "Life Before Language" (paper #1, v0.2)*

---

## Positioning: what paper #2 is and is not

Paper #1 is the framework paper. It establishes the category error, the hierarchy, the two gaps, and reports a first empirical season whose headline is the **internalization** rung: formation changes what a system *is*, and still nothing matters *to* it.

Paper #2 is narrower, deeper, and more experimental. It takes one question — the wall's Level-3 face — and prosecutes it with a single flagship experiment plus its controls. It should be readable by someone who has not read paper #1, but it should not re-argue paper #1. Roughly 60% experiment, 40% framing.

**Title candidates**, in order of preference:
1. *The Assigned Residue: Bearing and Constituting in Artificial Continuants*
2. *Whose Outcome Is It? Bearer-Constitution and the Limits of Self-Referential Stakes*
3. *From Bearing to Constituting: Measuring the Designer-Set Fixed Point*

Preference for (1): "assigned residue" is the paper's coined term and its measured object, and the title should name what the paper contributes to the vocabulary.

**Venue:** arXiv cs.AI primary, cs.LG cross-list (the experimental content is legible to an ML audience in a way paper #1's is not), optional phil-sci cross-list. This paper is more likely than paper #1 to be read by people who don't care about Boethius, and that is fine — the design should stand without the metaphysics.

---

## §1 — Introduction: two questions that look like one

**Move:** open on the distinction, not the framework.

When we say an outcome "matters to" a system, we may mean either of two things, and current work in agency, RL, and artificial life routinely runs them together:

- **Bearing:** the outcome lands on this system, alters it, and its alteration is indexed to this system's identity rather than any other's.
- **Constituting:** the *criterion by which* something counts as a good-or-bad outcome for this system is generated and maintained by the system's own ongoing operation, rather than specified from outside it.

An agent can bear outcomes without constituting them: the reward function is the designer's, the agent merely carries the consequences. Whether an artificial system can do the second is a different question from whether it can do the first, and this paper argues the difference is measurable.

**Prior result to recap in one paragraph** (not a full section): a bearer-indexed continuant probe established that bearer-identity becomes load-bearing exactly when outcomes stop being facts about the world and become facts of the bearer. Bearing is engineerable and measured. Constituting is what this paper attacks.

**Contribution statement:**
1. An operational distinction between assigned and constituted outcome-criteria.
2. The **assigned residue** — the designer-set fixed point that any self-referential outcome-criterion must terminate in — introduced as a measurable quantity via a depth sweep.
3. A three-arm experiment (assigned / self-referential / yoked self-reference) isolating whether self-generated criteria buy anything a yoked copy cannot have.
4. The measured shape of the advantage-versus-depth curve, with its interpretation and honest bounds.
5. A pre-registered statement of the result that would trouble the framework's prediction.

**End §1 on the pre-registration promise.** Stating in the introduction what would count as disconfirmation is the paper's strongest credibility move and should be visible before any result.

---

## §2 — Background: bearing, established

Short section. Three subsections, ~1.5 pages total.

**2.1 The continuant setup.** Single agent, drifting world, outcomes update the parameters the agent acts with. The yoked control: same rule, same information, wrong bearer.

**2.2 The bearer-indexing result.** When outcomes are facts about the world, self-driven and other-driven alteration are indistinguishable — bearer-identity is idle. As outcomes are made to depend on the individual (a private per-agent filter on what each experiences), the self-versus-yoked advantage grows monotonically and becomes universal across seeds **when scored on the agent's own (assigned) criterion**. On a bearer-neutral yardstick the same comparison is null (+0.002, 55% of 40 seeds) — the advantage is criterion-internal. Paper #2 must recap it in those terms.

**2.3 What the result does not establish.** The private filter was *assigned*. The criterion of mattering lived in the experimental design, not in the agent. Remove the designer's mapping and nothing about the agent says which outcomes should matter to it. This is the gap paper #2 enters.

**Related-work placement** (keep tight, integrate rather than siloing): intrinsic motivation and curiosity-driven RL (self-generated *objectives*, still designer-specified objective-generators); homeostatic and active-inference agents (self-maintenance as objective, with the viability set assigned — and note the free-energy principle's known dark-room and circularity critiques are the assigned-residue argument in another vocabulary: the generative model that defines surprise is itself the designer's/evolution's fixed point); **Hoffman's interface theory and the fitness-beats-truth results** (evolution selects cheap stake-shaped interfaces over accurate perception — the theoretical frame for why an acting agent's representations cannot be probed independently of its behavior, which this paper hits as a measurement problem); autopoiesis and enactivist self-constitution (the conceptual ancestor of "constituted," largely unmeasured); open-endedness and POET-style co-evolution (environment self-generation, distinct from criterion self-generation). The honest framing: several literatures gesture at self-constitution; none isolate the assigned residue as a measurable depth.

---

## §3 — The distinction, operationalized

The conceptual core. This section must be airtight because everything downstream depends on the definitions being non-question-begging.

**3.1 Assigned outcome-criterion.** Formally: a mapping *(agent, world-state) → outcome-valence* specified by the experimenter and invariant under the agent's own history. The agent bears the outcome; the criterion is exogenous.

**3.2 Constituted outcome-criterion.** The criterion at time *t* is a function of the agent's own parameter trajectory up to *t*, such that what counts as good-or-bad for this agent is not specifiable independently of this agent's history of being itself.

**3.3 The regress, and why it does not dissolve the distinction.** Self-reference cannot be unbounded in a constructed system: somewhere there is an initial condition, an update rule, or a coupling function the designer set. Call this the **assigned residue**. The obvious objection — "then 'constituted' just means 'assigned with extra steps,' and the distinction is empty" — is the objection the paper must meet head-on and does *not* meet by argument. It meets it by measurement: if the distinction were empty, deepening the self-reference would buy nothing, and the depth sweep would be flat. A flat curve would vindicate the objection. A non-flat curve would refute it. **Make this explicit: the experiment is designed so the deflationary objection has a clean way to win.**

**3.4 Why the wall is predicted to stand regardless.** Even a steeply rising curve would show constituted-*looking* stakes scaling with depth — not stakes. The paper's prediction is that a residue always remains, and the value of the experiment is the *shape* of the approach, not a crossing. Signpost §7 here.

---

## §4 — Experimental design

**4.1 Three arms.** Architecture, world, drift schedule, and information held identical.
- **A — Assigned.** Fixed private taste, experimenter-set. Replication baseline; confirms the apparatus reproduces the known bearer-indexing effect.
- **B — Self-referential.** Outcome-criterion is a function of the agent's own current parameters. Outcomes alter parameters, parameters alter the criterion, criterion alters which outcomes register.
- **C — Yoked self-reference.** Identical in form to B; the criterion is driven by a *different* agent's parameter trajectory. Same self-referential structure, wrong bearer of the self-reference.

**B-vs-C is the discriminator.** A-vs-B alone cannot distinguish "self-generated criterion" from "richer assigned criterion." Only the yoked control isolates whether the self-reference is doing bearer-constitutive work or is decorative. **Say this in the design section, not just the analysis section** — reviewers will look for exactly this and should find it stated before the results.

**4.2 The depth sweep — the paper's central instrument.**
- depth 0: fixed taste (= Arm A)
- depth 1: taste is a function of current parameters; the function is assigned
- depth 2: that function adapts by an assigned meta-rule
- depth *k*: *k* self-referential layers before the designer-set floor

Plot B-vs-C advantage against depth. The assigned residue is what sits below the deepest layer at each setting; the sweep measures how much the advantage depends on how deep the residue is pushed.

**4.3 Controls and confounds to pre-empt.**
- **Capacity confound.** Deeper self-reference adds parameters. Include a *parameter-matched assigned* arm at each depth — an equally expressive but exogenous criterion — or the curve measures capacity, not constitution. This is the most likely reviewer objection; handle it in the design.
- **Drift-rate interaction.** Sweep at ≥2 drift rates; a result that holds only at one drift schedule is a result about that schedule.
- **Seed replicates and universality.** Report per-seed, not just aggregate. The bearer-indexing result's strength was that the advantage became *universal across seeds*, not merely large on average; hold this paper to the same standard.
- **Yoking fidelity.** Verify the yoked partner's trajectory is genuinely matched in distribution — a yoked control that is accidentally easier or harder invalidates the discriminator.

**4.4 Pre-registered predictions.** State all three shapes before results:
1. **Flat / saturating floor.** Advantage exists but does not grow with depth — the residue does the work at every depth; "constituted" reduces to "assigned with extra steps." *Deflationary-consistent.*
2. **Rising with diminishing returns toward an asymptote.** Self-reference does progressively more bearer-constitutive work, approaching but not reaching a limit where the residue vanishes. *Asymmetry-consistent; parallels the construction-order non-convergence one level down.*
3. **Rising without apparent asymptote in range.** Does not prove the wall crossed — bounded experiments cannot — but is the shape the framework cannot comfortably absorb, and defines the scale-up. *Troubling.*

---

## §5 — Results

Structure to write once data exists. Reporting order:

**5.1** Arm A replication — apparatus validity. (If A does not reproduce the bearer-indexing effect, nothing downstream is interpretable; report this first and plainly.) **Note:** Arm A reproduces *on the own-criterion metric only* — it is null on the bearer-neutral yardstick — so report both metrics at every arm and depth. A single-metric result is not interpretable here.
**5.2** B-vs-C at each depth, per-seed and aggregate.
**5.3** The advantage-versus-depth curve — the headline figure. This is the paper's one image that must be immediately readable.
**5.4** Parameter-matched assigned control at each depth, overlaid on the same axes. The gap between "B-vs-C" and "B-vs-parameter-matched-assigned" is what separates constitution from capacity.
**5.5** Drift-rate replication.
**5.6** Which pre-registered shape obtained, stated in one sentence before any interpretation.

**Discipline:** report the shape before interpreting it. The internalization result in paper #1 was strong partly because the number (0.41–0.45 occupancy, invariant across variations) was stated flatly before its meaning was argued. Same here.

---

## §6 — Interpretation

Written three ways in advance; keep the branch that obtains.

**If shape 1 (flat/saturating):** The deflationary objection wins on its own terms, and this is a real contribution — "constituted" outcome-criteria, as currently constructible, are assigned criteria with additional indirection. The assigned residue is not merely present but *dominant*. This constrains a lot of ambitious language in the artificial-agency literature and is worth publishing as a negative result with a positive method.

**If shape 2 (asymptotic):** Self-reference buys progressively more, without eliminating the residue. This mirrors the construction-order experiment's path-dependence-with-non-convergence at a different level of the hierarchy, which is a notable structural echo: two independent probes, two levels apart, both finding *approach without arrival*. Worth flagging as a possible general signature of the near side of the wall — carefully, since two instances is a pattern-in-waiting, not a law.

**If shape 3 (unbounded in range):** Report it as troubling for the prediction and insufficient for any positive claim. The honest reading: within tested bounds, deepening self-constitution did not exhibit the predicted saturation; this is exactly the result that motivates scale-up rather than conclusion. **Do not soften this branch if it obtains.** A framework that pre-registers its disconfirmation condition and then hedges when it appears has spent its credibility for nothing.

**Common to all branches:** none of the three shapes shows anything mattering to the agent. The paper should say this in the same place regardless of outcome, so that it reads as structural rather than as consolation for a disappointing result.

---

## §7 — The wall at Level 3, restated

Short, ~1 page. Situate the finding on paper #1's ladder:

> modulation → internalization → ⟨ THE WALL ⟩ → consequence → subjectivity

Bearing and constituting both sit on the near side. Bearing was measured in the prior probe; constituting is measured here; the wall is what neither reaches. The Level-3 face of the wall gets its sharpened statement: **the assigned residue is the measured signature of the wall at Level 3 — the designer-set fixed point that self-reference approaches and (predicted) does not eliminate.**

The parallel to paper #1's internalization result is the section's payoff: there, a system was *changed* by its formation and nothing mattered to it. Here, a system *generates its own criterion of mattering* and still nothing matters to it. Two different routes to the wall, both arriving at the same face.

---

## §8 — Limitations

Written plainly and early-drafted, not appended.

- Toy scale, one research lineage, no independent replication. Say it in these words.
- The depth sweep's *k* is bounded by compute; the curve's behavior beyond tested range is unknown and the paper claims nothing about it.
- The self-referential coupling is one implementation of a large family; a different coupling could show a different curve. One probe is not the class — the same caution paper #1 applied to the holographic memory result.
- The parameter-matched control mitigates but does not eliminate the capacity confound.
- "Constituted" as operationalized here is a formal property of the criterion's dependence structure. Whether it bears any relation to what the word means for organisms is exactly the question the wall marks as open, and the paper does not claim it does.
- **Measurement in acting agents is behavior-coupled.** The capstone attempt found that in a system whose behavior determines its input distribution, representational measurements (occupancy) cannot be cleanly separated from policy: probing on self-generated inputs contaminates the measurement with behavior; probing on fixed inputs discards the signal. This is interface theory's claim in measurement form, and it is why the passive-model results (fixed corpus = fixed probe set by default) were clean. Any future joint measurement should prefer a world where bearer-indexing is *structural* — e.g. position-holding environments where the same world-event has position-dependent valence — rather than assigned via a filter.

---

## §9 — Conclusion

Two paragraphs.

First: what was measured, in the flattest possible language. The distinction between bearing and constituting is operational; the assigned residue is measurable via depth sweep; the obtained shape was X; the interpretation is Y, bounded by Z.

Second: the question that survives. Paper #1 ended by asking whether anything beginning at Level 7 could become the bearer of its own outcomes rather than their assignee. The bearing half is answered. The constituting half now has a measured shape and an unmeasured remainder — the residue. The next question is whether the residue is a fact about constructed systems as such, or an artifact of construction at this scale, and that question is not answerable by any experiment this paper can run.

---

## Figures (target: 4)

1. **The three arms.** Schematic: assigned / self-referential / yoked self-reference, showing where the criterion originates in each.
2. **The depth sweep.** Diagram of depth 0→k with the assigned residue marked as the floor. This figure teaches the paper's central concept and should be legible standing alone.
3. **Advantage vs. depth** (headline result), with the parameter-matched control overlaid and per-seed points visible.
4. **The ladder**, with bearing and constituting both placed on the near side of the wall. Reuse paper #1's ladder graphic for visual continuity across the two papers.

---

## Writing sequence (recommended)

1. §3 (the distinction) — everything depends on these definitions; write them before running anything so the experiment tests what the paper claims.
2. §4 (design) — write in full, including pre-registered predictions, *before* data collection. This is what makes the pre-registration real rather than retrospective.
3. Run the experiment.
4. §5 (results) → §6 (interpretation, keeping the branch that obtains).
5. §1, §2, §7, §8, §9 last — framing is easiest once the result's shape is known, and writing it earlier risks the framing steering the analysis.

---

## Terminology check (carry the LayeredFitness lesson)

Before any of these terms enter the manuscript, verify each names actual implemented code and a PROBE_MATRIX column:
- `assigned residue` — needs a concrete definition in the harness (which parameters constitute the floor at each depth)
- `residue depth` / depth sweep — needs to be a real configuration axis, not a conceptual one
- `constitution candidate` (Arm B) — needs an implemented coupling function
- `yoked self-reference` (Arm C) — needs verified distribution-matching against Arm B

A term that names a design intention rather than running code does not appear in the paper. That is how "LayeredFitness" got into a draft it had no business being in.
