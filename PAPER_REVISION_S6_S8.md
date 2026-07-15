# Paper Revision: §6 and §8 against the measured results

**Date:** 2026-07-14
**For:** *Life Before Language* (arXiv draft v0.1)
**Status:** Drop-in replacement sections. Prose matches the draft's voice.
Every number traces to `PROBE_MATRIX.md`; every claim is stated at the
strength a toy-scale, design-diverse-within-one-lineage experiment supports.

Summary of what changed and why:
- §6 keeps its conceptual argument and gains a forward pointer to a measured
  result; the answerable/modulated distinction is no longer only "a Workstream 2
  question" but one with a first empirical answer.
- §8 changes from *three illustrative fragments* to *four measured probes plus
  a flagship experiment*. Oracle Prime → **Continuant Probe** (renamed; the
  cited "LayeredFitness" implementation did not resolve and is dropped).
  **FTTF holographic memory** is added as the Level-5 probe. A new **§8.5**
  reports the construction-order experiment. A new **§8.6** introduces the
  four-stage refinement of the wall that the data forced.
- Framing discipline: the evidence is "five design-diverse systems built within
  one research lineage repeatedly reproduce the same bearer/change separation,"
  never "five independent systems." No result is asked to carry a metaphysical
  claim; the wall is located, never crossed.

---

## §6 — The Retrofit Asymmetry (revised)

*(First three paragraphs unchanged from the draft. Replace the final sentence
of the section — the "arguably the sharpest one the program generates" clause —
with the following, and append the new paragraph.)*

… Whether this directional difference is merely historical (a fact about
construction order with no ongoing significance) or constitutive (a permanent
structural difference in how the levels relate) is itself a Workstream 2
question — arguably the sharpest one the program generates, since it is
empirically approachable: retrofitted systems can be probed for whether their
symbolic layer ever becomes genuinely *answerable* to the added lower layers,
or merely modulated by them. **§8.5 reports a first measurement.**

**A note on what "answerable" can be made to mean.** The distinction between
*answerable to* and *modulated by* is not merely rhetorical; it can be given an
operational form and measured. A lower layer added to a symbolic system is
merely *modulated* if the system's competence survives the layer's removal
unchanged, and if corrupting the layer's content leaves the system's reasoning
untouched — the layer is consulted, not constitutive. It approaches
*answerable* to the degree that competence depends on the layer, that
perturbing the layer's state propagates into reasoning without the system being
told, and — the decisive variable — that the layer was present during the
system's formation rather than attached after it. §8.5 shows that the last of
these, construction order, has a measurable and repeatable effect, and §8.6
draws the consequence for how the wall itself should be described.

---

## §8 — Empirical Probes: Measured Fragments of the Interface (revised)

The program is grounded in implemented systems, each realizing one fragment of
the developmental interface. Their value is dual — as Workstream 1 engineering,
and as Workstream 2 evidence — and it is now, in several cases, quantitative:
each probe was subjected to a common battery of ablation and perturbation tests
whose logic is the *answerable/modulated* distinction of §6. A methodological
caution frames all of it. These systems were built within a single research
program; they are *design-diverse* — a decoding constraint, a memory
architecture, a hidden-state geometry, an evolutionary runtime — but they are
not independent samples. The claim the evidence supports is therefore precise:
**five architecturally unrelated systems, built within one lineage, repeatedly
reproduce the same separation between the bearer of a function and the thing the
function changes.** Every measurement below is at small scale, and the section
is explicit about what each result does and does not establish.

**8.1 Internalized temperature regulation (Gabriel's Horn), measured at
Level 7.** The temperature ladder of the original argument stands: (i) *fixed*
external temperature (standard decoding); (ii) *scheduled* temperature
(Gabriel's Horn: a 1/x entropy-decay schedule applied at inference); (iii)
*state-dependent* temperature (a dynamical thermostat maintaining its own
effective temperature); (iv) *consequence-trained* temperature, achieved by no
system. Rung (ii) was tested directly. Applying the entropy schedule to a
small instruction-tuned model and measuring competence on tasks unrelated to
the schedule's content, we find the schedule shapes output without measurable
effect on competence (ablation effect ≈ 0, confidence interval including zero),
and — because an inference-time decoder never alters the model's weights —
removing it restores the bit-identical model. For this class of regulation the
modulation verdict is therefore not merely observed but *architectural*: a
constraint applied at decoding shapes what the system does and cannot, in
principle, change what it is. Perturbing the schedule's internal state without
notifying the model redistributes the model's behavior across tasks while
leaving total competence unchanged — a clean illustration of a constraint that
shapes output without becoming a condition of operation.

**8.2 Structural perception (LGI / lattice-grid inference), at Level 4.** The
LGI architecture registers deformation of a metric structure over its own state
space and treats forbidden zones as internally represented constraints,
implementing the formal shape of Level-4 orientation: the system "perceives"
its own structural situation. The precise shortfall is unchanged and is the
hinge of the whole program — the constraint is not a consequence. Violation
costs the system nothing it bears; the metric is answerable to the designer's
objective, not to the system's continuation. What §8.5 adds is that this same
geometric mechanism, when moved from an inference-time overlay into the model's
*training*, produces a measurably different relationship between the layer and
the reasoning — which is exactly the retrofit-asymmetry question made
experimental.

**8.3 Retrofit memory (holographic RAG), measured at Level 5.** A holographic
reduced-representation memory compresses a corpus into superposed
high-dimensional vectors and serves retrieved context to a language model. It
is the paper's Level-5 probe, and it matters because its formal structure is
closer to *sedimentation* than ordinary storage: each stored trace is superposed
into the same vector, so accumulation modifies the accumulating object rather
than filing a new record beside the old ones. Yet the measured behavior is
starkly that of a service consulted, not a ground grown from. Two tests,
run on a small model over tasks unrelated to the corpus, give a dissociation
that states the Level-5 shortfall more sharply than argument could. First, the
*presence* of the memory layer significantly *degrades* competence on unrelated
tasks (the retrieved context taxes the reasoner's attention). Second, corrupting
the memory's *content* — returning systematically wrong retrievals, without
telling the model — has no measurable effect at all. The system is sensitive to
having the service and indifferent to whether the service is right. This
"presence-tax / content-indifference" pattern is the precise inverse of
constitutive memory, where presence is transparent and corruption is
devastating; and it is what "connected to memory" rather than "constituted
through memory" looks like when made measurable. (One honest qualification: the
sedimentation is real only for the routing index, a compressed layer; the
underlying store remains discrete, and the superposition has a measured capacity
limit that biological memory need not share. The probe implements the formal
structure and, characteristically, its residue.)

**8.4 Consequence-shaped variation and the continuant (the Continuant Probe),
at Level 3.** An evolutionary runtime with per-individual fitness and mutation
implements the nearest available analogue of consequence: variation disciplined
by differential outcome. This is real selection pressure. Its shortfall is
temporal locus — the discipline operates on the *lineage*, not within a *life*;
no individual accumulates a developmental history, and consequence shapes the
distribution of systems rather than the maturation of one system. This is the
control case. To ask what the lineage regime omits, we built its complement: a
single agent in a slowly drifting world whose own outcomes update the very
parameters with which it acts, alongside a yoked control whose parameters are
updated by a *different* agent's outcome stream — same update rule, same
information volume, wrong bearer. The comparison isolates B2's requirement, the
identity of actor and bearer, from mere information transfer.

The result is a dose relationship that locates consequence precisely. When
outcomes are facts about the world, self-driven and other-driven alteration are
indistinguishable: any agent profits equally from any history, and the identity
of the bearer is idle. As outcomes are made to depend on the individual — in the
model, a fixed private "taste" that filters what each agent experiences — the
self-versus-yoked gap grows monotonically and becomes universal across seeds.
The finding is therefore not that "learning from one's own experience helps"
(that is information transfer) but its precondition: **bearer-identity becomes
load-bearing exactly when outcomes stop being facts about the world and become
facts of the bearer.** This is the Level-3 claim — that appetite is the
organism's own condition, not a signal about a state — in operational dress.
The same experiment reproduces, inside a single life, the two signatures §8.5
finds across training runs: removing an external record of outcomes collapses
competence instantly (the actor was never changed), whereas freezing the
within-life updates leaves competence in place, decaying only as the world drifts
away from what was learned (the alteration lives in the actor and outlasts the
loop that produced it).

**Joint reading.** The probes triangulate the wall: regulation without stakes
(8.1), perception without consequence (8.2), sedimentation without a changed
reasoner (8.3), and consequence to a lineage rather than a life, with a first
operational handle on what the missing within-life version would require (8.4).
Each fragment is buildable; their unification in a single continuing center whose
own continuation is at stake is what has not been built, and §4 argues it is an
open question whether building it is a matter of architecture at all.

---

## §8.5 — The Construction-Order Experiment (new)

The retrofit asymmetry of §6 turns on a single variable — *when* a lower layer
enters a system's formation — that is normally impossible to isolate, because a
layer's presence and its construction history are confounded. A small
spectrally-regulated language model lets us separate them. A geometric
regularizer that projects a designated "forbidden" subspace out of the model's
hidden states can be introduced at three different points in the model's
history, holding the architecture, the geometry, and the data fixed: attached to
a finished model at inference only (C2a); attached to a finished model and then
fine-tuned into it (C2b); or present from the first step of training, so the
model's entire formation occurs through it (C3). Ablation is uniform across
conditions — the regularizer is switched off — and we measure the change in
clean-text competence, paired per validation sample.

The three conditions fall into three regimes, and the regimes are stable. The
inference-only retrofit is a *tax*: switching the regularizer off *improves*
competence, because the model never adapted to it and the projection is pure
overhead. The fine-tuned retrofit is a *crutch*: the model has come to lean on
the regularizer, and removing it now costs competence. The grown-in model is
*internalized*: removing the regularizer costs almost nothing, because the model
developed within the allowed geometry and no longer needs the external
projection to stay inside it. The naive expectation — that the most deeply
integrated layer would show the *largest* dependence — is therefore inverted in
the most informative way. Deep formation under a constraint does not make the
external constraint more necessary; it makes it nearly removable, because the
constraint's effect has passed into the model's own form. That the effect has
so passed is directly measurable: with the regularizer entirely removed, the
grown-in model's hidden states occupy the forbidden subspace at a fraction of
the chance rate, while an unregulated model sits at chance. The regulator's
discipline is legible in the weights of a model that no longer carries the
regulator.

Three properties of the result bear on the paper's claims, and each is stated at
the strength the experiment supports.

*The ordering is scale- and tokenization-robust.* The tax → crutch →
internalized ordering holds across a sixfold change in model size and across
both character-level and word-level (byte-pair) tokenization — four independent
training regimes — and, where replicated across random seeds, the internalized
and crutch regimes separate with non-overlapping seed ranges. Effect magnitudes
vary (they track the fraction of the state space the constraint occupies, not
model capacity); the *ordering* is the robust claim.

*The internalization signature has a candidate invariant.* Across every
configuration tested — two model sizes, two constraint fractions, two
tokenizations, replicated across seeds — the grown-in model's occupancy of the
forbidden subspace lands between 0.41 and 0.45 of the chance rate, while the
unregulated baseline holds at chance. We flag this as a candidate quantitative
regularity of internalization rather than a law; it is the kind of stable number
that a later, formal treatment would seek to explain, and it is reported here so
that it can be confirmed or overturned.

*On permanence versus convergence, the data favor neither pole but a third
option.* The original question — whether deeper retrofit adaptation eventually
becomes indistinguishable from grown-in construction (convergence) or remains
permanently distinct (permanence) — was tested by fine-tuning the retrofit for
increasing durations. The dynamics are regime-dependent: at small scale the
retrofit's dependence on the layer decays toward a floor with more adaptation;
at word level with adequate data it does not decay at all. But the *gap* is
constant where the dynamics are not: in every regime tested, the adapted
retrofit retains a multiple — roughly threefold — of the grown-in model's
scaffold dependence, and no amount of adaptation within the tested range closes
it. The defensible statement is thus neither "construction order leaves a
permanent residue" (a finite experiment cannot establish permanence) nor "it
washes out" (it demonstrably did not), but: *within the tested regimes, increased
retrofit adaptation approached but did not reach the grown-in signature, and the
residual gap is consistent with a path dependence that adaptation does not
close.* This is stronger for the retrofit-asymmetry thesis than either pole,
because it is what the thesis predicts and what the deflationary reading does
not.

A recurring side-observation deserves note because it is on-message and was not
sought. The same pipeline includes a "cure" phase that attempts to remove
poisoned behavior from an already-trained model by the same geometric means.
That post-hoc removal succeeds at small scale and *fails* — marginally but
repeatably — at larger scale and at word level. Retrofit correction gets harder
as the system grows, in the very runs where grown-in internalization gets
cleaner. Construction after the fact and construction from the start scale in
opposite directions.

---

## §8.6 — What the Wall Is, at Higher Resolution (new)

The construction-order result complicates, productively, the paper's central
distinction. §4 draws a line between *constraint* (which shapes what a system
does) and *consequence* (which changes what a system is). The grown-in model
shows that a constraint present during formation genuinely changes what the
system is — its competence comes to depend on, and its very state-space geometry
comes to embody, a discipline it was formed under. By the paper's own criterion
this is more than modulation. And yet nothing in the experiment brings the
system one step closer to *stakes*: the grown-in model was shaped by the
geometry, but no outcome was ever borne by it, no continuation of its was ever at
issue. It was changed; nothing mattered to it.

The clean way to hold this is to recognize that the near side of the wall has
internal structure the original binary compressed. Four stages can be
distinguished, and the paper's probes now populate the first three:

- **Modulation** — external structure affects operation; remove it and the
  system is unchanged. (Gabriel's Horn; retrofit memory; the inference-only
  regularizer; a within-life external record.)
- **Internalization** — formation under a structure changes the system, so that
  the structure's discipline persists in the system after the structure is
  removed. (The grown-in model; the within-life continuant.) This is where the
  measured construction-order effect lives.
- **— the wall —**
- **Consequence** — the acting continuant bears its own outcome and proceeds
  from its altered condition; the outcome is a fact of the bearer, not a fact
  about the world assigned to it. (Approached operationally by the Continuant
  Probe's bearer-indexing result; instantiated by no system.)
- **Subjectivity** — there is someone for whom the altered unfolding is had.
  (The framework makes no claim here; this is Level 2.)

The refinement changes nothing about where the wall sits — it still divides a
system that is shaped by its situation from a system for whose continuation the
situation is at stake — but it dissolves an apparent paradox. Internalization
can be deep, measurable, and real without being consequence, because
*constitution and consequence-bearing are separable*, and the grown-in model is
the first concrete case of the former without the latter. The wall did not move
when we found internalization above it; the frontier of what engineering can
reach moved, and the wall was still there. That is the most the framework asks
its probes to show, and now, at small scale and with the appropriate
caution, they show it.

---

## Integration notes (not part of the paper)

- **Abstract / §1:** update "three implemented systems" → "a family of
  implemented systems"; the specific probes are Gabriel's Horn (L7), LGI /
  spectral geometry (L4), holographic RAG (L5), and the Continuant Probe (L3).
- **References §145:** replace the Oracle Prime bullet; add repository links —
  horn-experiment, the spectral construction-order code (in NanoGPT), FTTF
  holographic RAG, continuant-probe. Flip horn-experiment and continuant-probe
  to public before submission.
- **Draper's Sphere** is cited nowhere in the draft and exists in no repository;
  no action needed unless a later draft introduces it.
- **§10.1** ("innate orientation is just pretraining") can now cite §8.4: the
  bearer-indexing dose curve is the concrete functional expression the objection
  was told, in the original, only to expect.
- Keep the "design-diverse within one lineage" wording anywhere the probes'
  agreement is invoked; never "independent."
