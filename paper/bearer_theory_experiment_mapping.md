# Bearer Theory and Our Experiments: Mapping, Status, and Proposed Updates

*Working note, July 2026. Maps the continuant-probe and CCA experiments onto the
Bearer Theory manuscript (Revised Chapters 1-9) and the PSD working draft, marks
what each experiment confirms, refines, or leaves open, and proposes concrete
additions. No em-dashes, per house style.*

---

## 1. What arrived, and why it matches

Two formal manuscripts, plus study guides:

- **Bearer Theory: A Formal Theory of Developmental Experience and Intelligence**
  (Chapters 1-9, 78 pages). A decomposition of ordinary dynamics into four loci:
  the acting bearer, the outcome constituted relative to that bearer
  `o = V(e, b)`, the update of the bearer `b' = U(b, o)`, and the recurrence of
  the same continuant as actor. Chapter 8 is a predictions-and-tests chapter.
  Chapter 9.9 specifies a minimal testbed.
- **PSD: Possibility, Selection, and Development in Adaptive Artificial Systems**
  (62 pages). The same loop with a front-end: a possibility distribution
  `q = Pi(b, w)`, a selection `a = S(q, xi)`, then the same event, outcome,
  update, and developmental capture. It shares the bearer formalism exactly and
  adds the entropy/selection decomposition and a reserved quantum-selector arm.

The match to our work is not loose. The manuscript already cites our probes as
preliminary evidence (the continuant self-versus-yoked experiment under
Prediction 2, the holographic-retrieval presence-tax result under Prediction 4).
Our CCA forager is a point-for-point instantiation of the Chapter 9.9 testbed.
The theory was written asking for exactly the experiment we built. What follows
lines them up.

---

## 2. Chapter 8 predictions mapped to our experiments

The manuscript's formal outcome mixture is `o(lambda) = (1-lambda) F(w,a) +
lambda V(w,a,b)`, with `lambda` the bearer-indexing dial, `F` a bearer-neutral
component, and `V` the bearer-relative one. This is exactly the lambda dial in
our bearer sweep. The self-versus-yoked gap `G_SY(lambda, h)` is exactly our
self-minus-wrong_bearer difference. The dual scoring (bearer-relative criterion
and bearer-neutral criterion) is exactly our diet_quality versus raw_food split.
The vocabulary is shared because the code was built to it.

| Ch 8 prediction | Our experiment | Status |
|---|---|---|
| **P1 (8.2): matched event, different effective outcome**, `V(e, b_A) != V(e, b_B)` | The wrong_bearer counterfactual: same action, same event, same dice, valued through a different constitution. Measured directly. | **Confirmed and quantified.** corr(V_self, V_wrong) = 0.79; the wrong-body outcome differs from the own-body one by 36% of signal magnitude. The manipulation is real, not trivially weak. |
| **P2 (8.3): self-history and yoked-history diverge as bearer indexing rises**, `dG/dlambda > 0`, with dual scoring | The bearer sweep over lambda in {0, .25, .5, .75, 1}, self vs wrong_bearer, scored on a bearer-neutral yardstick (raw food) and a bearer-indexed one (diet quality). | **Partially disconfirmed in this substrate; see Section 4.** Bearer-neutral null reproduced robustly. The dose relation `dG/dlambda > 0` did NOT appear in the forager, even when the setup was built to force it. This is a refinement, not a contradiction. |
| **P4 (8.5): record exchange vs bearer-state exchange; presence-tax / content-indifference** | The FTTF holographic retrieval probe. | **Confirmed** (already cited). Memory presence taxed competence; corrupting its content did not. Storage and the recurrent actor were separable in that setup. |
| **P5 (8.6): update freezing differs from record removal; under drift a frozen bearer decays** | The ablation gate: freeze the fast weights and bearer state, measure the performance drop, under static versus drifting worlds. | **Confirmed, and elevated to a mandatory gate; see Section 3.** Static world: freezing costs 0.005 (frozen bearer fine, because nothing was incorporated). Drift world: freezing drops the agent from -0.55 to near random baseline (+0.24 cost); blind agent +0.34 cost. Exactly the predicted temporal signature. |
| **P7 (8.8): matched snapshots conceal different plasticity regimes** | The construction-order experiment (tax / crutch / internalized), and the static-vs-drift-vs-blind champions that all plateau near -0.40 while differing entirely in whether the pathway is load-bearing. | **Confirmed.** Grown-in constraint leaves a persistent signature (occupancy 0.41-0.45 of chance) removable at almost no cost; retrofit keeps ~3x the dependence; no convergence. Snapshot performance hides the regime. |

Predictions P3 (delayed counterfactual divergence from a specific incorporated
outcome), P6, P8 (individual vs lineage closure), and P9 (developmental-coupling
response curve) are not yet run. They are the clean next targets and the testbed
already supports them.

---

## 3. The ablation gate is a contribution the manuscript should absorb

Chapter 8.6 predicts that update-freezing and record-removal have different
signatures. Our work turned that prediction into a **mandatory precondition** for
interpreting any bearer result, and showed why it is necessary with a concrete
failure.

The rule: before believing any self-versus-yoked or bearer result, verify that
the system actually depends on the outcome-driven pathway the manipulation acts
on. Freeze that pathway and measure the cost. If freezing costs nothing, the
system solved the task another way, the bearer manipulation has no surface to
act on, and any null is uninterpretable.

Our first sweep failed exactly this in a static world (freeze cost 0.005: the
pathway was idle). The drift world passed it (freeze cost 0.24), and the blind
agent passed it maximally (0.335). Only after the gate passes is the bearer
sweep worth reading. This maps onto the manuscript's own "four relations that
experiments must not collapse" (Table 8.1) and its measurement-declaration rule,
but it deserves to be stated as a gate, not only as a comparison. Proposed home:
a short methods subsection after 8.6, or a paragraph in 8.13 (Reporting
discipline).

---

## 4. The honest refinement to Prediction 2

This is the most important scientific update, and it strengthens the theory by
tightening it.

Prediction 2 says the self-versus-yoked gap grows with bearer indexing,
`dG/dlambda > 0`, in a regime where the update mechanism can exploit the
bearer-indexed information. The continuant probe found this on the bearer-relative
criterion (+0.909 on the assigned criterion, +0.002 on world food). Our CCA
forager, an independently built second substrate, did **not** reproduce the dose
relation, across four escalating experiments:

1. Static world: null, uninterpretable (gate failed).
2. Drift world, homeostasis criterion: null, faint lean.
3. Drift world, clean bearer-indexed diet-quality criterion: faint positive lean
   at every lambda above zero (up to +0.17), the predicted direction, but not
   significant.
4. Blind agent (homeostatic sense removed, so the outcome pathway is the only
   channel): the faint lean did not survive more seeds. Null. Diet quality stayed
   near random even blind.

The mechanism explains it. The forager agent maintains homeostasis by learning
**when and how much** to act, not **which** resources suit its particular body.
Survival needs enough total energy, not the right kind, so bearer-specific
selectivity is never incentivized. Bearer indexing can only drive the gap through
selectivity. With no selectivity pressure, `dG/dlambda` stays flat.

The refinement Prediction 2 needs: `dG/dlambda > 0` requires not only that the
outcome be bearer-indexed, but that the bearer-indexed component drive a
**load-bearing selectivity** in the task. Where the bearer-relative outcome can be
satisfied by a bearer-neutral strategy (timing, quantity, generic regulation),
the gap does not appear even though `V` genuinely differs. This is consistent
with the manuscript's own criterion-internal caveat in 8.3, and with its stated
goal of locating the scale over which each relation closes rather than forcing a
binary. It also predicts where the effect should return: raise the private
constitution strength so that eating the wrong resources is actively harmful,
making selectivity necessary. That is the untested dose curve (Section 6).

The continuant probe's +0.909 is best read, in light of this, as near-tautological
criterion-internal alignment: the agent was scored on the very quantity its
self-condition was trained to maximize. The forager decouples regulation from
selectivity, so no such alignment exists, and the effect vanishes. Two substrates
therefore agree on the robust claim (bearer-neutral null) and disagree on the
fragile one, which is exactly the criterion-internal picture the theory already
commits to.

---

## 5. Our CCA build versus the Chapter 9.9 testbed spec

Section 9.9 lists ten requirements for the minimal Constitutive Cognition
Architecture testbed. The continuant-probe/cca build satisfies most of them.

| 9.9 requirement | In our CCA build |
|---|---|
| 1. Small persistent actor | Yes. Recurrent policy net, d = 64, small enough for direct intervention. |
| 2. Continuing bearer core | Yes. Bearer state, outcome-gated fast weights, plasticity controller. |
| 3. World with controlled drift | Yes. drift_every = 200; static and drifting both run. |
| 4. Private constitution | Yes. Per-body nutrition, hazard vulnerability, action costs, sensory noise. |
| 5. Within-life update | Yes. Outcomes update the state the same agent acts from next. |
| 6. Yoked controls | Yes. wrong_bearer (same event, wrong body) and stream (classic yoked). |
| 7. Independent external logs | Partial. Results and champions are logged; a per-event audit trail is not yet separated as a first-class object. |
| 8. Ablation access | Yes for update and plasticity (the gate). Constitution and memory exchange/corruption not yet wired. |
| 9. Two evaluation criteria | Yes. raw_food (bearer-neutral) and diet_quality (bearer-indexed). |
| 10. Construction-order variants | Yes. The spectral construction-order experiment (tax/crutch/internalized), plus static/drift/blind formation variants here. |

The gaps are small and specific: a first-class per-event audit log (item 7), and
constitution-exchange and record-corruption ablations (item 8). Both are modest
additions to the existing GPU pipeline. With them, the testbed would support
Predictions 3, 4-exchange, and 8 directly.

Section 9.9 also asks for "a family of curves showing where histories cease to be
fungible, how delayed effects survive cue removal, how update freezing differs
from record removal, and whether plasticity itself changes with development." The
GPU batched evaluator is precisely the tool that makes such curve families cheap:
population evaluated as one tensor, so sweeps over lambda, drift rate, and
construction order are minutes, not hours.

---

## 6. What remains absent, per Section 8.14

The manuscript's evidence table has a "what remains absent" column. Our new
results move two rows forward but leave the honest gaps the theory already names.

- **Individual action-outcome-update closure with intrinsic stakes.** Still
  absent, by design. Nothing we measured crosses from functional consequence to
  stakes-bearing consequence. This is the wall, and the theory does not claim to
  cross it.
- **The constitution-strength dose curve.** The one place the forager null could
  turn positive. Raise const_sigma so own-poison resources do real damage, retrain
  per strength, and sweep. This is the direct test of the Section 4 refinement.
- **Independent replication.** All probes are one research lineage. The theory
  states this. Our second substrate reduces the single-design risk for the
  bearer-neutral null but is still in-house.

---

## 7. Proposed concrete updates to the manuscripts

1. **Section 8.3 (Prediction 2).** Add the load-bearing-selectivity necessary
   condition and the second-substrate result. Suggested addition:

   > An independently constructed foraging testbed reproduced the bearer-neutral
   > null across four escalating conditions but did not reproduce the dose
   > relation `dG/dlambda > 0`, because its task allowed the bearer-relative
   > outcome to be satisfied by bearer-neutral regulation (timing and quantity)
   > rather than bearer-specific selectivity. The prediction is therefore refined:
   > the gap requires the bearer-indexed outcome to drive a load-bearing
   > selectivity, not merely to differ. Where a bearer-neutral strategy suffices,
   > the gap does not appear even though `V` genuinely differs. This is consistent
   > with the criterion-internal qualification above.

2. **Section 8.6 or 8.13.** Promote the update-freeze test to a mandatory
   interpretive gate, with the static-versus-drift ablation numbers as the worked
   example of the failure mode it prevents.

3. **Section 8.14 (evidence table).** Add a row for the gated GPU CCA forager:
   relation tested, bearer indexing under the ablation gate with dual criteria;
   narrow result, bearer-neutral null robust across four conditions, dose relation
   absent without selectivity pressure, pathway confirmed load-bearing by the
   gate; what remains absent, the constitution-strength dose curve and intrinsic
   stakes.

4. **Section 9.9.** Note that a reference implementation of the testbed exists
   (continuant-probe/cca), satisfying items 1-6, 9, 10 fully and 7-8 partially,
   with the GPU batched evaluator providing the curve-family tooling the section
   calls for.

---

## 8. One-paragraph summary

The Bearer Theory and PSD manuscripts formalize the exact loop our experiments
test, and their Chapter 8 predictions and Chapter 9.9 testbed are a
specification of the CCA forager we built. Four of the manuscript's predictions
are already met by our probes: matched-event outcome divergence (quantified at
corr 0.79), the presence-tax memory result, the update-freeze signature, and the
construction-order regimes. Our main new contribution is methodological and
honest: an ablation gate that makes bearer nulls interpretable, and a
second-substrate result that reproduces the bearer-neutral null robustly while
failing to reproduce the self-versus-yoked dose relation, because the forager
lets a bearer-neutral strategy satisfy a bearer-relative outcome. That refines
Prediction 2 rather than breaking it, confirms the theory's own criterion-internal
caveat, and points to the one clean next experiment: the constitution-strength
dose curve.
