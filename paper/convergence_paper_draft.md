# When Identity Is Load-Bearing: A Substrate-Independent Design Law for Incorporated Developmental State

**Alex Bone**
Independent Researcher, Long Beach, CA
*Draft, August 2026*

---

## Abstract

An incorporated developmental state, a bearer that both biases behavior and is updated
by the behavior's own outcomes, is central to Bearer Theory's account of when a system
has an identity that is its own. We ask when that state actually does causal work rather
than sitting idle, and we answer with a single design law demonstrated across three
substrates that share nothing but the loop. An incorporated bearer becomes load-bearing
and non-transferable exactly on tasks that admit no non-incorporated route to viability
at the operating threshold; on all other tasks it is idle, and increasing capacity
increases the escape rather than the incorporation. We show three things are
substrate-independent. First, the bearer loop and delayed incorporation (Prediction 3)
appear in a reinforcement-learning forager, a transformer with a fast-weight memory, and
a competitive-fatigue dynamical system. Second, the Generic-Policy Escape Principle, that
the bearer stays idle while any non-incorporated route survives, is confirmed in the
forager and independently in the dynamical system. Third, in the one substrate that
forecloses the escape by construction, a transformer on an obligate task, the
incorporated state becomes load-bearing (freezing it collapses performance to chance) and
bearer-specific (updating it on another individual's outcomes fails on one's own
criterion, a self versus wrong-bearer gap of 0.51 with tight error bars, replicated,
dissociated from attention, and holding at scale). We also find the difficulty valley,
the failure that appears when an escape is foreclosed on a substrate whose bearer is not
adaptive, in two substrates. The claim is structural, about when an incorporated state
does work and cannot be transferred, and it is agnostic on the phenomenal question by
design.

---

## 1. Introduction

Bearer Theory decomposes an agent into four loci over one continuant: an actor whose
action is conditioned on a bearer state, an outcome constituted relative to that bearer,
an update of the bearer from its own outcomes, and the recurrence of all three over a
single persisting thing. The theory's Prediction 2 is that self-coupling, updating the
bearer on one's own bearer-relative outcomes, yields an advantage over a yoked control
that grows with bearer-indexing, and that the advantage is criterion-internal. The
sharpened question is not whether a bearer can be built, but when it does causal work.

A single demonstration of a bearer effect invites the objection that it is an artifact of
one architecture. The claim earns the word law only if it recurs in formalisms that share
no implementation and were not built to agree. This paper reports the same structure in
three: an evolved reinforcement-learning forager with outcome-gated fast weights, a small
transformer whose incorporated state is a fast-weight associative memory read alongside
attention, and a dynamical system in which tangent-force subsystems compete for control of
a state on a manifold under fatigue. The forager and the transformer were built for this
program. The dynamical system, FourD, was built independently for a different purpose and
arrived at the bearer loop on its own. What transfers between them is not code. It is a
loop and a law.

## 2. The bearer loop and the four loci

Write the loop as

    a_t = pi(b_t, w_t)        act, conditioned on the bearer
    o_t = V(e_t, b_t)          outcome, constituted relative to the bearer
    b_{t+1} = U(b_t, o_t)      update the bearer from its own outcome

over one continuant b. Each substrate instantiates all four loci.

- **Forager.** b is a bearer vector plus outcome-gated fast weights; pi is the recurrent
  policy conditioned on b; V is the bearer-relative homeostatic outcome; U is the plastic
  update.
- **Transformer (Constitutive GPT).** b is a fast-weight associative memory W, a matrix
  reset each episode; pi reads W bilinearly for a per-arm value and samples; V is the
  reward gated by the chosen arm; U is a reward-gated Hebbian write into W.
- **Dynamical system (FourD).** b is a persistent vector b_t; pi is perception biased by
  b_t then resolved into subsystem activities; V is the outcome the system registers,
  filtered through b_t; U is b_{t+1} = (1 - decay) b_t + gain activities_t.

The correspondence is exact, not analogical. FourD's own description of the property,
that b_t is a closed loop and not a passive log, is Bearer Theory's Prediction 4, the
incorporated bearer versus the retrievable record.

## 3. Result one: the bearer loop and incorporation are substrate-independent

Delayed incorporation, Prediction 3, is that a specific outcome leaves a residue in the
continuant that persists after its cue. It is measured in all three substrates.

- **Forager.** A specific outcome leaves a residue that survives after the cue, a delayed
  incorporation magnitude of 5.76 in the continuant probe.
- **Transformer.** The incorporated memory integrates its own outcomes over a horizon that
  a bounded context cannot hold; freezing the write removes exactly that integration.
- **Dynamical system.** A one-shot perturbation injected into b_t persists for about 120
  steps before decaying below 1/e of its peak, a memory horizon that is 0 without the
  bearer.

The loop and its incorporation are therefore not features of one architecture.

## 4. Result two: the escape principle is substrate-independent

The bearer stays idle whenever the task leaves a non-incorporated route to viability.

- **Forager.** Across nine task designs and two capacity levers, self-coupling produced no
  bearer-neutral performance advantage, because each task retained a bearer-independent
  policy that preserved viability. The two levers sharpened this into a meta-level result:
  softening the viability threshold re-admitted a quantity route, and giving the learner a
  richer update rule made it exploit that route more efficiently, not the bearer. A more
  capable optimizer does not discover its bearer identity; it finds the escape better.
- **Dynamical system.** The same appears as a clean null. Transplanting a different body's
  incorporated bearer trajectory into a body A, matched in volume, changes A's competency
  within noise (self-maintenance 2.714 versus 2.716, basin divergence 0.003 versus 0.003
  against a same-body falsifier), because A's fixed subsystem preferences are a
  non-incorporated route to its behavior and the bearer is a weak add-on the dynamics wash
  out. The Generic-Policy Escape in a dynamical system.

Two unrelated formalisms, one escape.

## 5. Result three: foreclosure yields load-bearing non-transferability

The transformer is the one substrate that forecloses the escape by construction. Its task
is stochastic bandit identification: a hidden per-instance code, weakly informative
outcomes, so identifying the code requires integrating outcomes over a horizon far longer
than any bounded context. Gate zero is an information bound, computed with no training:
a context-only learner restricted to the last L outcomes stays near chance (0.170 at the
locked calibration) while the full horizon is solvable (0.874). The context window, the
sequence model's most powerful escape, is foreclosed.

On that task the incorporated memory becomes load-bearing and bearer-specific.

- **Load-bearing.** The trained model reaches near the information ceiling, and freezing
  the write collapses accuracy to exact chance, a drop of 0.607. A dual-channel model with
  a real attention block shows a double dissociation: removing the attention record drops
  accuracy by 0.003 (the record is idle) while freezing the memory collapses it to the
  context bound (the bearer carries the task).
- **Bearer-specific.** Updating the memory on a different individual's outcome stream fails
  on one's own criterion. Self beats wrong-bearer with a gap of 0.509 plus or minus 0.049
  across six seeds, wrong-bearer landing on exact chance, while a falsifier (the same
  individual's outcomes, independent noise) holds and isolates the effect. The result is
  not a one-bit lookup: a multi-dimensional private code (a vector identity) shows the same
  non-transferability. And it holds at scale, reproducing at 133,637 parameters with a
  genuine transformer block, the gap if anything larger.

## 6. The design law

The three results combine into a biconditional with a mechanism.

    An incorporated bearer is load-bearing and non-transferable exactly on the tasks that
    admit no non-incorporated route to viability at the operating threshold. It is idle on
    all others, and increasing capacity increases the escape, not the incorporation.

This predicts idleness (the forager and the dynamical system) and load-bearing specificity
(the transformer) from a single structural property of the task, and it does so across
substrates that share only the loop. It is the positive form of the program's earlier
finding, which could only report the negative half.

## 7. The difficulty valley is also substrate-independent

Foreclosing the escape is necessary but not sufficient. On a substrate whose bearer is not
adaptive, foreclosing the escape can leave the identity unsustainable. In the forager, an
obligate task with the escape closed became unlearnable at the smallest scales, the
difficulty valley. In the dynamical system, a foreclosure attempt that made the body's
identity live only in the bearer produced the effect at short horizon (a basin divergence
of 0.103 under the wrong bearer against 0.008 for the falsifier) but lost it at the full
horizon (0.004 versus 0.004): a seeded identity is transient, and without an ongoing
objective the loop converges to a generic attractor and the identity washes out. The
lesson is the same in both: a bearer must be adaptive and task-driven, integrating a
body-specific parameter from its own outcomes that behavior then requires, for the
incorporated identity to be sustained. The transformer had exactly this, an adaptive
learned memory and an obligate task, which is why it alone reaches the positive result.

## 8. Falsifiable predictions for new substrates

- Any substrate with a surviving non-incorporated route will show an idle bearer, a null
  self versus wrong-bearer, regardless of capacity.
- Foreclosing that route, and only that, together with an adaptive bearer, will make the
  bearer non-transferable.
- The stated concrete test is the FourD adaptive foreclosure: give the dynamical system's
  bearer a task-driven objective that requires an integrated body parameter, then repeat
  the self versus wrong-bearer test. The prediction is that non-transferability then
  appears, as it did once the transformer closed the escape.

## 9. Relation to the manuscripts

This is the empirical core of Bearer Theory's Prediction 2 and its criterion-internal
qualification, generalized across substrates, and it grounds Life Before Language's claim
that a developmental substrate is what a mind is built on rather than a record it carries.
FourD's order/disorder transition, a saddle-node bifurcation that is sharp and independent
of the embedding manifold, additionally offers a concrete dynamical model for the wall,
the determinate threshold the framework posits between a graded engineering approach and
what lies past it.

## 10. Honest scope

The claim is structural, about when an incorporated state does causal work and cannot be
transferred. It is agnostic on the phenomenal question by design, as each substrate's own
write-up is. The convergence strengthens the near-side result, that a developmental bearer
can be built, foreclosed, and measured, and it says nothing about the far side, which is
the correct division of labor for the program. Randomness quality, including quantum
random number generation tested in the dynamical system, was found not to drive the
incorporation effect: perturbation shape does, not entropy source. Incorporation is a
structural property of the loop, not of the noise.

## 11. Methods and reproducibility

The transformer code and results are public under MIT at
github.com/vikingdude81/constitutive-gpt, with the milestone map and the trained policies.
The forager foundation is in the same repository. The entropy-decay probe is public at
github.com/vikingdude81/gabriels-horn. The dynamical system is the FourD repository, with
the bearer loop in bearer_state_competency.py and the self versus wrong-bearer and
foreclosure tests in fourd_self_vs_wrong_bearer.py and fourd_foreclosure.py. Training in
the forager and transformer is evolutionary search; the dynamical system is a fixed
integrator; the gate-zero bound is a Monte Carlo of the optimal decoder and needs no
training. Per-experiment numbers are in the RESULTS files of each repository.
