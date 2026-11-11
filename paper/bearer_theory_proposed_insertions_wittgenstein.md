# Proposed Manuscript Insertions, Wittgenstein Batch (E through G)

*Drawn from the 2026-11-10 shared ChatGPT conversation on the Tractatus
Logico-Philosophicus and its intersections with Bearer Theory and Life Before
Language. Each block states whether it is adopted as-is, adopted with a
correction, or rejected with reason. The batch obeys the program's own
three-part rule: what Wittgenstein claims, what his argument establishes, and
where our reading extends beyond the text. Symbols follow the manuscript
(`o_t = f(w_t, b_t)`, `b_t+1 = U(b_t, o_t)`, `Q_h`). No em-dashes.*

---

## Insertion E. Formal core, decomposition of the encounter chain

*Placement: at the introduction of the formal dynamics (the `o_t = f(w_t, b_t)`
equations), before the update operator is discussed. Adopted with one
caution noted below.*

> The outcome function decomposes into three separable operations. The bearer
> acts, `a_t = pi(b_t, w_t)`. The world responds, `e_t = E(w_t, a_t)`. The
> bearer appraises the response under its own constitution,
> `o_t = V(e_t, b_t)`. The decomposition is not decorative. Each stage names a
> distinct ablation surface: a policy intervention acts on `pi`, an environment
> intervention acts on `E`, a constitution intervention acts on `V`, and the
> update-freeze gate acts on `U`. A result reported only at the level of
> `f(w_t, b_t)` cannot say which stage carried the effect.
>
> The chain also fixes the status of action in the theory. Action is not only
> an expression of the bearer's present organization; through its consequences
> it can become a mechanism of the bearer's future organization. This is the
> dynamical content of the history-bearing claim: the bearer at `t+1` is not
> merely in a new state, it occupies a different accessibility structure, and
> which transitions are available to it next depends on which realizations
> have already occurred.
>
> Two restrictions hold throughout. Not every action produces a lasting
> update, and not every update is beneficial. A bearer can learn, habituate,
> become injured, or lose capabilities. Development in this theory is change
> in what the bearer can subsequently do, not improvement, and the empirical
> question for any update claim is whether the action left a persistent,
> causally demonstrable change in the bearer's accessible range.

*Author note: the chat also offers, two messages later, a policy written as
`a_t = pi(b_t, r_t)` over a representation variable. That is a different
model from `a_t = pi(b_t, w_t)` and the two must not be pasted adjacently
without the identifiability discipline in Insertion G. Do not adopt the
r-layer here.*

---

## Insertion F. The Tractatus as the limiting case (U = identity)

*Placement: philosophical positioning or related-work section, wherever the
Wall signature is stated. Adopted; this is the original contribution of the
batch.*

> The early Wittgenstein is the cleanest existing formalization of the
> opposite of Bearer Theory. In the Tractatus the possibilities of
> combination that belong to an object are intrinsic to its logical nature
> (2.012), and elementary states of affairs are logically independent of one
> another (1.21, 2.061). Occurrences select which possibilities are
> realized; they do nothing to the form of the object, and no fact inherits
> its place in logical space from the history of any other fact. In the
> vocabulary of this theory, the Tractatus is the case `U = identity`: the
> bearer enters the encounter and is never the thing that is changed, while
> the space of possibility is fixed prior to and independent of any
> bearer's history.
>
> Read this way the Tractatus is not a precursor of Bearer Theory, it is its
> null model, stated in the strongest available terms. Bearer Theory's
> load-bearing claim is precisely the denial of that picture: the
> accessibility structure is bearer-dependent and construction-order
> dependent, which is what the L5 construction-order result measures, grown
> in flow geometries differing from retrofitted ones at the same endpoint
> state. The philosophical lineage is therefore adversarial rather than
> convergent, and the paper is stronger for naming the null model it breaks
> than for collecting similarities.

---

## Insertion G. Prelinguistic representation: the licensed bridge and the identifiability discipline

*Placement: Life Before Language scope section, and the Bearer Theory
treatment of internal representation if one exists. Adopted with corrections;
the raw chat formulation fails review as stated.*

> **(a) What Wittgenstein establishes and what the reading extends.** The
> Tractatus establishes that combination possibilities are intrinsic to
> objects and prior to their propositional description: logical possibility
> does not depend on some system already possessing a proposition that
> describes it. It does not establish that a system can represent a state of
> affairs without linguistic or logical form. On the contrary, proposition 3
> identifies thought itself with the logical picture of facts, so the early
> Wittgenstein is the strongest available opponent of the prelinguistic
> cognition claim, not its witness. The stronger claim, that structured
> discrimination and development occur before any symbolic registration,
> belongs to the later Wittgenstein's discussions of rule following and form
> of life, and any citation supporting it must come from there. If the
> 2.012 bridge appears in the manuscript it must be labeled as our
> extension, with the text's own anti-representationalist reading of thought
> stated alongside it.
>
> **(b) The representation variable requires an identifiability criterion.**
> The extended chain `r_t = R(w_t, b_t)`, `a_t = pi(b_t, r_t)` is a real
> extension of the model only if `R` is measurable independently of the
> behavior it explains. Without such a measure, `pi(b_t, R(w_t, b_t))`
> collapses into a reparameterized policy `pi'(b_t, w_t)`, absorbs every
> anomaly by postulating a changed representation, and the theory loses
> falsifiability at exactly the seam where it becomes interesting. The
> program's own evidence makes the cost concrete: at the finest granularity
> yet probed, block-level representational difference in trained bearer
> substrates was bearer blind (uniform weight relative distribution across
> all layers, bearer-indexed event rates indistinguishable, Fisher p = 1).
> The question "did the representation change or only the policy" is
> therefore to be carried as an open experimental question with its null
> attached, not as an available explanatory resource. Until `R` has a
> non-behavioral measure, the policy-only model is preferred by parsimony,
> and representation language in the manuscript should be kept
> methodological rather than ontological.

---

## Rejected from this batch

1. The chat's formulation that "the existence of structured possibilities
   does not depend upon a system possessing linguistic symbols for those
   possibilities," offered as a defensible starting point for Life Before
   Language on the authority of TLP 2.012-2.0121. Rejected as an
   over-licensing of the text; see Insertion G(a) for the licensed version.
2. The unqualified `r_t` extension. Rejected under the identifiability
   discipline; see Insertion G(b).

## Cross-reference check for the author

- Insertion E should be checked against the existing definition of `V` in
  Chapter 5 so that `V(e_t, b_t)` here and the constitution function `V` in
  the dose relation are the same symbol used consistently, not two
  functions wearing one name.
- Insertion F's "U = identity" framing should be checked against the Wall
  section so the two statements of "the bearer is not the thing changed,"
  the empirical one and the Wittgenstein null model, cite each other rather
  than repeat each other.
- Insertion G(b) must carry the null result citation from the power run;
  do not cite the representation/policy distinction without it.
