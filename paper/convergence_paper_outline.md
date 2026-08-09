# Outline: When identity is load-bearing — a substrate-independent design law for incorporated developmental state

*Paper outline, 2026-08-08. The cross-substrate paper: CCA forager, Constitutive GPT,
and FourD, one design law. Written after the FourD self-versus-wrong null sharpened the
thesis. No em-dashes.*

---

## Thesis (one sentence)

An incorporated developmental state, a bearer that both biases and is biased by its own
outcomes, becomes load-bearing and non-transferable if and only if the task forecloses
every non-incorporated route to viability, and this holds across substrates that share
nothing but the bearer loop.

## Why a cross-substrate paper

A single demonstration invites the objection that the effect is an artifact of one
architecture. The claim earns the word law only if it recurs in formalisms that share no
implementation. This paper reports the same structure in three: a reinforcement-learning
forager with fast weights, a sequence model with attention and a fast-weight memory, and
a competitive-fatigue dynamical system on a manifold. What transfers is not code but a
loop and a law.

## The three substrates

| | substrate | bearer realization | task |
|---|---|---|---|
| CCA | evolved RL forager | outcome-gated fast weights + bearer state | foraging under drift, then obligate metabolism |
| Constitutive GPT | sequence model | fast-weight associative memory + attention | obligate bandit identification, escape foreclosed by construction |
| FourD | dynamical system on S3/S2/R4 | persistent bearer vector biasing perception | competitive fatigue, order/disorder transition |

All three instantiate the four loci: act conditioned on the bearer, outcome constituted
relative to the bearer, update of the bearer from its own outcomes, one continuant.

## The three results, and what is substrate-independent

1. **The bearer loop and incorporation (Prediction 3) are substrate-independent.**
   Delayed incorporation is measured in all three: the CCA testbed (a specific outcome
   leaves a residue that survives the cue, ||db|| = 5.76), the GPT (the memory integrates
   over a horizon a bounded context cannot hold), and FourD (a one-shot perturbation
   persists about 120 steps, a memory horizon that is zero without the bearer).

2. **The Generic-Policy Escape Principle is substrate-independent.** The bearer stays
   idle whenever the task leaves a non-incorporated route to viability. The forager showed
   this across nine designs plus two capacity levers, and the levers sharpened it into a
   meta-level result: a more capable optimizer exploits the escape better, not the bearer.
   FourD showed the same by a clean null: transplanting a different body's bearer changes
   competency within noise (self-maintenance 2.714 versus 2.716, basin divergence 0.003
   versus 0.003), because the body's fixed preferences are a non-incorporated route. Two
   unrelated formalisms, one escape.

3. **Foreclosing the escape yields load-bearing non-transferability (Prediction 2 and the
   sharp Prediction 4).** The Constitutive GPT forecloses the context escape by
   construction (an information bound: no bounded context can solve the task). There the
   incorporated state becomes load-bearing (freezing it collapses accuracy to chance) and
   bearer-specific (self beats wrong-bearer, gap 0.509 plus or minus 0.049 across six
   seeds, wrong-bearer at exact chance, falsifier isolating the effect). The record versus
   bearer distinction is shown inside one model by a double dissociation, and the whole
   result reproduces at 133,637 parameters with a real transformer block.

## The design law (the unifying claim)

    An incorporated bearer is load-bearing and non-transferable exactly on the tasks
    that admit no non-incorporated route to viability at the operating threshold. It is
    idle on all others, and increasing capacity increases the escape, not the
    incorporation.

This is a biconditional with a mechanism, not an intuition. It predicts idleness (the
forager and FourD) and load-bearing specificity (the GPT) from a single structural
property of the task, and it does so across substrates.

## Falsifiable predictions for new substrates

- Any substrate with a surviving non-incorporated route will show an idle bearer
  (null self-versus-wrong), regardless of capacity.
- Foreclosing that route, and only that, will make the bearer non-transferable. The
  stated concrete test is the FourD foreclosure: weaken or remove the fixed preferences
  so behavior must route through the bearer, then repeat the self-versus-wrong test. The
  prediction is that non-transferability then appears.

## Relation to the manuscripts

This is the empirical core of Bearer Theory's Prediction 2 and its criterion-internal
qualification, generalized across substrates, and it grounds Life Before Language's claim
that a developmental substrate is what a mind is built on rather than a record it
carries. FourD's manifold bifurcation, sharp and substrate-independent, additionally
offers a concrete dynamical model for the wall.

## Honest scope

The claim is structural, about when an incorporated state does causal work and cannot be
transferred. It is agnostic on the phenomenal question by design, as each substrate's own
write-up is. The convergence strengthens the near-side result and says nothing about the
far side, which is the correct division of labor for the program.

## Suggested venue

An interdisciplinary complexity or cognitive-systems venue, or arXiv cs.NE with a
cross-post to nlin.AO for the dynamical-systems substrate. The three code repositories
(constitutive-gpt public, gabriels-horn public, FourD) supply the reproducibility base.

## Status of the pieces

- CCA forager: complete, consolidated (paper/bearer_theory_empirical_consolidation.md).
- Constitutive GPT: complete and hardened (paper/constitutive_gpt_addendum.md).
- FourD: bearer loop and Prediction 3 confirmed, escape principle confirmed by the null
  (paper/fourd_bearer_bridge.md); foreclosure experiment optional.
- This paper draws the three together. It can be written now on the current results; the
  FourD foreclosure would add a fourth positive cell but is not required for the law.
