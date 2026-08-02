# Bearer Theory and Our Experiments: Full Audit, Status, and Proposed Updates

*Working note, July 2026. Full chapter-by-chapter audit of the Bearer Theory
manuscript (Revised Chapters 1-9) and the PSD working draft against the
continuant-probe and CCA experiments. Marks what each experiment confirms,
refines, or leaves open, and proposes concrete additions. No em-dashes, per
house style.*

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

The match is not loose. The manuscript already cites our probes as preliminary
evidence, and its notation is our code's notation. The theory was written asking
for exactly the experiment we built.

---

## 2. Chapter-by-chapter conceptual audit (Chapters 1-7 and PSD)

This section walks the conceptual chapters and marks where our systems and
results instantiate each construct. Chapter 8 (predictions) and 9.9 (testbed)
get their own sections below.

**Chapters 1-2 (the bearer problem, definitions).** The minimal formal package is
`a = pi(b, w)`, `e = E(w, a)`, `o = V(e, b)`, `b' = U(b, o)`. Our CCA agent
implements it line for line: `act(obs)` is `pi`, the environment step is `E`, the
outcome mixture is `V`, and `plastic_update` is `U`. The manuscript's core
warning, that persistence of information is not persistence of the bearer, is the
exact distinction our ablation gate operationalizes: an external record can
survive without the actor being changed.

**Chapter 3 (bearer-relative constitution).** This is the chapter our forager was
built to test. Section 3.4 gives the outcome mixture
`o = (1 - lambda) F(w, a) + lambda V(w, a, b)` and the bearer-sensitivity measure
`I_B = norm(do/db)`, with the explicit note that for discrete systems `I_B`
should be estimated by matched bearer swaps or self-versus-yoked comparisons
rather than a literal derivative. Our wrong_bearer counterfactual is exactly that
matched bearer swap, and corr(V_self, V_wrong) = 0.79 with a 36% signal
difference is a direct discrete estimate of `I_B > 0`. Confirmed. Section 3.6
(constitution is not development) and 3.3 (the Continuant Probe reference) are
already tied to our lineage.

**Chapter 4 (developmental update).** Our `plastic_update` is `U`. Section 4.5
(development without improvement) matters for reading our nulls honestly: a
bearer can be genuinely altered without its coupling improving, and our
bearer-neutral nulls are development-without-bearer-advantage, not absence of
development. The gate shows the update is real and load-bearing; the sweep shows
it does not create a bearer-identity performance gain. Those are two distinct
findings, exactly as Chapter 4 insists.

**Chapter 5 (the bearer loop and closure).** The four loci (acting, constitutive,
update, recurrent) closing over one continuant is bearer closure (Definition 5).
Section 5.3's two architectures are our two probe families: the external-record
architecture is the FTTF holographic retrieval probe (record changes, actor
unchanged), and the changed-actor architecture is the CCA agent (outcome alters
the state the next action proceeds from). Section 5.6 (self-history and
yoked-history as a closure test) is the design of our bearer sweep. The gate is
the operational check that the update and recurrent loci actually coincide with
the acting locus.

**Chapter 6 (history, plasticity, developmental intelligence).** Section 6.4
introduces history-shaped plasticity, `b' = U_theta(b, o)` with `theta = Theta(b)`.
That is precisely our PlasticityController: a network that reads bearer state and
sets the update regime (the eta and delta of the fast-weight rule). Our system is
therefore in the manuscript's richer meta-development class, not the fixed-rule
class. Definition 6 (developmental intelligence) has four requirements: closure
over one bearer, outcomes entering causally rather than as detachable records,
alteration available over a horizon, and improvement against a declared
criterion. Our gated, drift-trained, held-out-selected CCA agent meets the first
three operationally and is scored against the fourth on two declared criteria. On
the bearer-neutral criterion it is developmentally intelligent by this
definition; on the bearer-indexed criterion the advantage is criterion-internal.

**Chapter 7 (relationship to existing fields).** Section 7.7 (large language
models and agent systems) is the formal version of paper 1's central move: an LLM
assembly contains weights, activations, context, retrieval store, and controller
at different timescales, and calling the whole thing an agent does not identify
the bearer. Paper 1's seven-level hierarchy and its wall are one articulation of
that bearer problem. Section 7.11's novelty claim (keeping four causal identities
aligned, plus five distinctions) is the frame our probes were built to test one
distinction at a time.

**PSD (possibility, selection, development).** PSD shares the loop and adds the
front-end `q = Pi(b, w)`, `a = S(q, xi)`. Its classical experiments manipulate
possibility width, selection sequence, update coupling, environmental drift, and
scaffold removal. Our program already covers four of these: update coupling (the
bearer sweep), environmental drift (the drift world), scaffold removal
(construction-order), and self-versus-yoked histories. PSD's central negative
prediction, developmental capture `C_qh ~ 0` (variation occurred but was not
incorporated), is instantiated by our Gabriel's Horn entropy-decay probe:
inference-time regulation changed the output distribution while leaving the actor
unchanged, so the variation was not developmentally captured. The one PSD arm we
have not touched is the reserved quantum or physical selector source at `xi`,
which the draft itself defers.

---

## 3. Chapter 8 predictions mapped to our experiments

The manuscript's outcome mixture, self-versus-yoked gap `G_SY(lambda, h)`, and
dual scoring (bearer-relative and bearer-neutral criteria) are exactly our lambda
dial, our self-minus-wrong_bearer difference, and our diet_quality versus
raw_food split. The vocabulary is shared because the code was built to it.

| Ch 8 prediction | Our experiment | Status |
|---|---|---|
| **P1 (8.2): matched event, different effective outcome**, `V(e, b_A) != V(e, b_B)` | The wrong_bearer counterfactual: same action, same event, same dice, valued through a different constitution. | **Confirmed and quantified.** corr(V_self, V_wrong) = 0.79; wrong-body outcome differs by 36% of signal magnitude. The manipulation is real, not trivially weak. |
| **P2 (8.3): self-history and yoked-history diverge as bearer indexing rises**, `dG/dlambda > 0`, dual scoring | The bearer sweep over lambda in {0, .25, .5, .75, 1}, self vs wrong_bearer, scored on raw food (neutral) and diet quality (indexed). | **Partially disconfirmed in this substrate; see Section 5.** Bearer-neutral null reproduced robustly. The dose relation did not appear in the forager. A refinement, not a contradiction. |
| **P4 (8.5): record exchange vs bearer-state exchange; presence-tax / content-indifference** | The FTTF holographic retrieval probe. | **Confirmed** (already cited). Memory presence taxed competence; corrupting its content did not. |
| **P5 (8.6): update freezing differs from record removal; under drift a frozen bearer decays** | The ablation gate: freeze fast weights and bearer state, measure the drop, static vs drift. | **Confirmed and elevated to a mandatory gate; see Section 4.** Static: freezing costs 0.005. Drift: +0.24. Blind agent: +0.34. Exactly the predicted temporal signature. |
| **P7 (8.8): matched snapshots conceal different plasticity regimes** | Construction-order (tax / crutch / internalized); and static/drift/blind champions all plateau near -0.40 while differing entirely in whether the pathway is load-bearing. | **Confirmed.** Grown-in signature at 0.41-0.45 of chance, removable at almost no cost; retrofit keeps ~3x the dependence; no convergence. |

Predictions P3 (delayed counterfactual divergence), P6, P8 (individual vs lineage
closure), and P9 (developmental-coupling response curve) are not yet run. The
testbed extensions in Deliverable 3 enable P3, the exchange half of P4, and P8.

---

## 4. The ablation gate is a contribution the manuscript should absorb

Chapter 8.6 predicts that update-freezing and record-removal have different
signatures. Our work turned that prediction into a mandatory precondition for
interpreting any bearer result, with a concrete failure showing why.

The rule: before believing any self-versus-yoked or bearer result, verify the
system depends on the outcome-driven pathway the manipulation acts on. Freeze it
and measure the cost. If freezing costs nothing, the system solved the task
another way and any null is uninterpretable.

Our first sweep failed this in a static world (freeze cost 0.005: idle pathway).
The drift world passed (0.24) and the blind agent passed maximally (0.335). Only
after the gate passes is the sweep worth reading. This maps onto Table 8.1 (the
four relations not to collapse) and the measurement-declaration rule, but it
deserves to be stated as a gate. Proposed home: a methods subsection after 8.6,
or a paragraph in 8.13.

---

## 5. The honest refinement to Prediction 2

Prediction 2 says the self-versus-yoked gap grows with bearer indexing,
`dG/dlambda > 0`, where the update can exploit bearer-indexed information. The
continuant probe found this on the bearer-relative criterion (+0.909 assigned,
+0.002 world food). Our CCA forager, an independent second substrate, did not
reproduce the dose relation across four escalating conditions:

1. Static world: null, uninterpretable (gate failed).
2. Drift world, homeostasis criterion: null, faint lean.
3. Drift world, diet-quality criterion: faint positive lean at every lambda above
   zero (up to +0.17), predicted direction, not significant.
4. Blind agent (homeostatic sense removed): the faint lean did not survive more
   seeds. Null. Diet quality stayed near random even blind.

The mechanism: the forager maintains homeostasis by learning when and how much to
act, not which resources suit its body. Survival needs enough total energy, not
the right kind, so bearer-specific selectivity is never incentivized. Bearer
indexing can only drive the gap through selectivity, so the gap stays flat.

The refinement: `dG/dlambda > 0` requires the bearer-indexed component to drive a
load-bearing selectivity, not merely to differ. Where the bearer-relative outcome
can be satisfied by a bearer-neutral strategy (timing, quantity, generic
regulation), the gap does not appear even though `V` genuinely differs. This is
consistent with the criterion-internal caveat in 8.3 and with the chapter's goal
of locating the scale at which each relation closes. It predicts where the effect
returns: raise the private constitution strength so eating the wrong resources is
harmful, making selectivity necessary (the untested dose curve, Section 7).

The continuant probe's +0.909 is best read as near-tautological criterion-internal
alignment: the agent was scored on the very quantity its self-condition was
trained to maximize. The forager decouples regulation from selectivity, so no
such alignment exists and the effect vanishes. Two substrates agree on the robust
claim (bearer-neutral null) and disagree on the fragile one, exactly the
criterion-internal picture the theory already commits to.

---

## 6. Our CCA build versus the Chapter 9.9 testbed spec

Section 9.9 lists ten requirements. The continuant-probe/cca build satisfies most.

| 9.9 requirement | In our CCA build |
|---|---|
| 1. Small persistent actor | Yes. Recurrent policy net, d = 64. |
| 2. Continuing bearer core | Yes. Bearer state, outcome-gated fast weights, plasticity controller. |
| 3. World with controlled drift | Yes. drift_every = 200; static and drifting both run. |
| 4. Private constitution | Yes. Per-body nutrition, hazard vulnerability, action costs, sensory noise. |
| 5. Within-life update | Yes. Outcomes update the state the same agent acts from next. |
| 6. Yoked controls | Yes. wrong_bearer and stream. |
| 7. Independent external logs | Now yes, via Deliverable 3 (per-event audit log). Previously partial. |
| 8. Ablation access | Update and plasticity via the gate; constitution and record exchange/corruption now added in Deliverable 3. |
| 9. Two evaluation criteria | Yes. raw_food (neutral) and diet_quality (indexed). |
| 10. Construction-order variants | Yes. Spectral construction-order plus static/drift/blind formation variants. |

Section 9.9 also asks for a family of curves showing where histories cease to be
fungible, how delayed effects survive cue removal, how update freezing differs
from record removal, and whether plasticity itself changes with development. The
GPU batched evaluator is the tool that makes such curve families cheap.

---

## 7. What remains absent, per Section 8.14

- **Individual closure with intrinsic stakes.** Absent by design. Nothing crosses
  from functional consequence to stakes-bearing consequence. This is the wall.
- **The constitution-strength dose curve.** The one place the forager null could
  turn positive. Raise const_sigma, retrain per strength, sweep. Direct test of
  the Section 5 refinement.
- **Independent replication.** All probes are one research lineage. The second
  substrate reduces single-design risk for the bearer-neutral null but is
  in-house.

---

## 8. Proposed concrete updates

See the companion file `bearer_theory_proposed_insertions.md` for ready-to-paste
manuscript text in the Bearer Theory voice: a Prediction 2 refinement paragraph
(8.3), an update-freeze gate paragraph (8.6/8.13), an evidence-table row and a
present-evidence paragraph (8.14), and a reference-implementation note (9.9).

---

## 9. One-paragraph summary

The Bearer Theory and PSD manuscripts formalize the exact loop our experiments
test, and their Chapter 8 predictions and Chapter 9.9 testbed are a specification
of the CCA forager. Our systems instantiate the conceptual chapters directly:
`V` and `I_B` (Chapter 3), `U_theta` history-shaped plasticity (Chapter 6.4), the
external-record versus changed-actor split (Chapter 5), and PSD's
developmental-capture null (Gabriel's Horn). Four Chapter 8 predictions are met.
The main new contribution is methodological and honest: an ablation gate that
makes bearer nulls interpretable, and a second-substrate result that reproduces
the bearer-neutral null while failing to reproduce the self-versus-yoked dose
relation, because the forager lets a bearer-neutral strategy satisfy a
bearer-relative outcome. That refines Prediction 2, confirms the criterion-internal
caveat, and points to the constitution-strength dose curve as the next
experiment.
