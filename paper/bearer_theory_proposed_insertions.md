# Proposed Manuscript Insertions for Bearer Theory (Chapters 8-9)

*Ready-to-paste text in the manuscript's register, drawn from the gated CCA
forager results. Each block names its target section and whether it adds,
replaces, or extends existing text. Symbols follow the manuscript
(`o(lambda) = (1 - lambda) F + lambda V`, `G_SY`, `Q_h`). No em-dashes.*

---

## Insertion A. Section 8.3, refinement of Prediction 2

*Placement: append to 8.3, after the paragraph reporting the continuant probe
result. This adds the second-substrate finding and tightens the prediction.*

> An independently constructed foraging testbed was used to test the dose
> relation directly. It implemented a private per-body constitution, a within-life
> outcome-driven update, matched self and yoked provenance, controlled drift, and
> two declared criteria: a bearer-neutral resource measure and a bearer-relative
> diet-quality measure equal to the mean own-body value of the resources the agent
> consumed. Across four conditions of increasing stringency, the bearer-neutral
> null was reproduced: self-coupled and yoked agents were interchangeable on the
> resource measure at every value of the bearer-indexing parameter. The dose
> relation `dG_SY / dlambda > 0` did not appear, even in a condition in which the
> agent's homeostatic sense was removed so that the outcome pathway was its only
> route to regulation and the update pathway was verified to be load-bearing.
>
> The reason locates the scale of the effect rather than removing it. In this
> testbed the bearer-relative outcome could be satisfied by a bearer-neutral
> strategy: the agent maintained viability by regulating when and how much it
> acted, not by selecting resources suited to its particular constitution.
> Survival required sufficient total intake rather than constitution-appropriate
> intake, so bearer-specific selectivity carried no advantage, and bearer identity
> could not become load-bearing through it.
>
> A constitution-strength dose curve tested whether making the wrong resources
> actively harmful would restore the effect. The strength of the private
> constitution was raised across a fourfold range, so that some resources became
> strong own-poison. It did not restore the effect. At every strength the agent's
> realized diet remained at or below the value a random diet would give under its
> own constitution, and the self-versus-yoked gap did not grow. The agent absorbed
> the stronger poison by consuming more of the beneficial resources rather than by
> avoiding the harmful ones, a bearer-neutral response.
>
> Prediction 2 is therefore refined in a strong form. The self-versus-yoked gap
> grows with bearer indexing only where the task admits no bearer-neutral strategy
> that keeps the bearer viable, so that the agent is forced to act on its own
> private constitution. Where a quantity or timing policy suffices, the gap can
> remain flat at every constitution strength even though the constitution function
> `V` differs sharply across bearers. This is consistent with the criterion-internal
> qualification above. An advantage that appears only under the agent's own
> developed lights, and only when the task forecloses every bearer-neutral route to
> viability, is a demonstration that bearer identity can become causally
> load-bearing, not a general performance law.

---

## Insertion B. Section 8.6, the update-freeze gate as a precondition

*Placement: append to 8.6 (or as a short 8.13 methods paragraph). Elevates the
freeze test from a comparison to a precondition, with the worked failure.*

> The update-freeze test also serves a prior methodological function. Before any
> self-versus-yoked or bearer-indexing result is interpreted, the analyst should
> establish that the proposed bearer actually depends on the outcome-driven update
> pathway that the manipulation acts upon. The freeze intervention supplies this
> check directly: hold the bearer state and its plasticity fixed after time T and
> measure the change in coupling quality `Q_h`. If freezing the pathway leaves
> coupling essentially unchanged, the system has solved the task through some
> other route, the bearer manipulation has no surface on which to act, and a null
> self-versus-yoked result carries no information about bearer identity.
>
> This failure mode is not hypothetical. In an early run of the foraging testbed,
> conducted in a static environment, a trained agent reached full competence while
> freezing its entire outcome-driven pathway changed coupling by a negligible
> amount. The static task was solvable by a fixed policy, so no outcome was
> incorporated, and the subsequent bearer sweep on that agent was uninterpretable.
> Introducing controlled drift restored the dependence: freezing the pathway then
> reduced coupling toward the untrained baseline. Only agents that pass this gate,
> whose coupling depends measurably on the update pathway, yield interpretable
> bearer results. The gate should be reported alongside every bearer-indexing
> measurement.

---

## Insertion C. Section 8.14, present-evidence table row and paragraph

*Placement: add a row to the evidence table and a short paragraph. Updates the
continuant entry with the gated second substrate.*

New table row:

| Probe | Relation tested | Narrow result | What remains absent |
|---|---|---|---|
| Gated constitutive foraging testbed (second substrate) | Bearer indexing under an update-freeze gate, with dual criteria | Update pathway confirmed load-bearing under drift; bearer-neutral self-versus-yoked null reproduced across four conditions; the dose relation did not appear without selectivity pressure | The constitution-strength regime in which selectivity is required; delayed-counterfactual and record-exchange tests; intrinsic stakes |

Accompanying paragraph:

> A second foraging substrate, constructed independently of the continuant probe
> and evaluated on a batched population implementation, extends this evidence in
> two directions. First, it makes the update-freeze gate a reported precondition,
> which distinguishes an interpretable null from an uninterpretable one. Second,
> it reproduces the bearer-neutral self-versus-yoked null robustly while failing
> to reproduce the dose relation, and it identifies the responsible structural
> condition: the task allowed a bearer-relative outcome to be satisfied by
> bearer-neutral regulation. The two substrates therefore agree on the robust
> claim and clarify the conditions under which the fragile, criterion-internal
> effect should and should not appear. Neither substrate provides independent
> replication in the strong sense, since both belong to one research program, and
> neither approaches intrinsic stakes.

---

## Insertion D. Section 9.9, reference implementation note

*Placement: append to 9.9, after the ten-item specification.*

> A reference implementation of this testbed exists. It provides a small recurrent
> actor with an explicit bearer core (policy disposition, an outcome-gated
> fast-weight store, and a plasticity controller of the form `theta = Theta(b)`),
> a world with controlled drift, a private constitution that alters the outcome of
> a matched event, within-life update, matched yoked provenance, an independent
> per-event audit log held separately from the bearer, ablation access to update,
> plasticity, constitution, and external record, two evaluation criteria, and
> construction-order variants. It is evaluated as a batched population on a single
> device, so that the families of curves called for above, over the bearer-indexing
> parameter, the drift rate, and construction order, are inexpensive to produce.
> The present manuscript's empirical claims about bearer indexing and the
> update-freeze gate were obtained from this implementation and from the earlier
> probes it consolidates.

---

## Notes for the author

- Insertion A is the substantive scientific change. It should be checked against
  Chapters 3 and 5 so the phrase "load-bearing selectivity" does not collide with
  any earlier definition. It does not, in the read performed here, but the author
  knows the manuscript best.
- Insertion C's table row uses the same four-column shape as the existing 8.14
  table. Adjust wording to match the surrounding rows' brevity.
- Insertion D presumes Deliverable 3 is merged (per-event audit log and
  constitution/record exchange ablations). Until then, soften "an independent
  per-event audit log" and "ablation access to ... constitution, and external
  record" to match what is implemented.
- All four blocks avoid any claim of stakes, subjectivity, or consciousness, per
  the manuscript's scope discipline.
