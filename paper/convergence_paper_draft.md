# When Identity Is Load-Bearing: A Proposed Design Law for Incorporated Developmental State

**Alex Bone**
Independent Researcher, Long Beach, CA
*Draft, August 2026*

---

## Abstract

An incorporated developmental state, a bearer that both biases behavior and is updated by
that behavior's own outcomes, is central to Bearer Theory's account of when a system has
an identity that is its own. We ask when that state actually does causal work rather than
sitting idle, and we propose a design law, supported by cross-substrate evidence, that
predicts three regimes. When the task admits a non-incorporated route to viability, the
bearer is idle, and increasing capacity increases the escape rather than the
incorporation. When the route is foreclosed but the bearer cannot adapt to carry the
required state, the system fails, the difficulty valley. When the route is foreclosed and
the bearer is adaptive, the incorporated state becomes load-bearing and bearer-specific. We
give operational definitions of these terms and show the mechanism across three substrates
that share nothing but the loop: a reinforcement-learning forager, a transformer with a
fast-weight memory, and a competitive-fatigue dynamical system. The loop and delayed
incorporation appear in all three; the escape principle in two; the difficulty valley in
two; and load-bearing non-transferability in the one substrate that forecloses the escape
with an adaptive bearer, where a self versus wrong-bearer gap of 0.51, replicated with
tight error bars, dissociated from attention, and holding at scale, establishes it. A
counterintuitive corollary follows for scaling: a more capable optimizer does not discover
its bearer identity, it exploits the escape better. The claim is structural and agnostic on
the phenomenal question by design.

**Keywords:** developmental state, fast weights, meta-learning, personal identity,
substrate independence, incorporated memory, non-transferability, machine consciousness

---

## 1. Introduction

Bearer Theory decomposes an agent into four loci over one continuant: an actor whose
action is conditioned on a bearer state, an outcome constituted relative to that bearer, an
update of the bearer from its own outcomes, and the recurrence of all three over a single
persisting thing. Its Prediction 2 is that self-coupling, updating the bearer on one's own
bearer-relative outcomes, yields an advantage over a yoked control. The sharpened question
is not whether a bearer can be built, but when it does causal work.

A single demonstration invites the objection that a bearer effect is an artifact of one
architecture. The claim earns the word law only if it recurs in formalisms that share no
implementation and were not built to agree. This paper reports the same structure in three:
an evolved reinforcement-learning forager with outcome-gated fast weights, a small
transformer whose incorporated state is a fast-weight associative memory read alongside
attention, and a dynamical system in which tangent-force subsystems compete for control of
a state on a manifold under fatigue. The forager and the transformer were built for this
program. The dynamical system, FourD, was built independently for a different purpose and
arrived at the bearer loop on its own. What transfers is not code. It is a loop and a law.

The core result, informally: identity does not become load-bearing because a system has
memory. It becomes load-bearing when the system can no longer succeed by treating its
history as somebody else's history, or as no history at all.

**Contributions.**

1. A proposed design law with three regimes (escape open, escape closed with a static
   bearer, escape closed with an adaptive bearer), stated formally with operational
   definitions.
2. Cross-substrate evidence for both sides of its mechanism: the loop and incorporation in
   all three substrates, the escape principle in two, the difficulty valley in two, and
   load-bearing non-transferability in the one that forecloses the escape adaptively.
3. A scaling corollary: capacity does not imply incorporation when an escape exists; it
   implies better exploitation of the escape.
4. Falsifiable predictions for new substrates and a concrete experiment that would convert
   the dynamical system from an escape demonstration into a positive one.

## 2. Related work

The bearer loop is a fast-weight architecture in the sense of Hinton and Plaut and of
Schmidhuber's fast-weight programmers: a slow network sets, and a fast state carries,
information that changes within a single episode. Our incorporated memory is a learned-write
fast weight, close in mechanism to modern fast-weight and linear-attention memories, but the
question we ask is different. That literature asks whether such a memory improves
performance. We ask when it becomes the only viable carrier of a behavior and therefore
non-transferable, which is a question about task structure, not memory capacity.

The self-versus-yoked comparison is the meta-learning setup of learning to reinforcement
learn (RL-squared and relatives), in which a recurrent state adapts within an episode. Our
result explains a failure mode of that setup: when the environment leaves a bearer-neutral
route, the adapting state is idle, and a more capable meta-learner exploits the route more.
This is a caution for reading emergent in-context adaptation as a sign of a persistent self.

The record-versus-bearer distinction is the episodic-memory-versus-weights distinction made
sharp. Retrieval-augmented and long-context models treat history as a record that is read;
our foreclosure result shows that when the needed information cannot be held in a bounded
context, an incorporated state carries it and a record cannot, framed as a structural
necessity rather than an efficiency.

Philosophically the target is the developmental and constitutive component of personal
identity, the Lockean intuition that a self is constituted by an appropriated history,
rendered as a non-transferability property rather than a memory-continuity criterion. We
part from psychological-continuity accounts by locating identity in an incorporated state
that cannot be copied into another actor without failing, and we take no position on the
phenomenal question.

## 3. The bearer loop and the four loci

Write the loop as

    a_t = pi(b_t, w_t)        act, conditioned on the bearer
    o_t = V(e_t, b_t)          outcome, constituted relative to the bearer
    b_{t+1} = U(b_t, o_t)      update the bearer from its own outcome

over one continuant b. Each substrate instantiates all four loci: the forager (bearer plus
outcome-gated fast weights, a recurrent policy, a bearer-relative homeostatic outcome, a
plastic update), the transformer (a fast-weight associative memory W, a bilinear read, a
reward gated by the chosen arm, a reward-gated Hebbian write), and the dynamical system
(a persistent vector b_t biasing perception, subsystem activities, an outcome filtered
through b_t, and b_{t+1} = (1 - decay) b_t + gain activities_t). The correspondence is exact,
not analogical: FourD's own words, a closed loop and not a passive log, are Prediction 4.

## 4. Definitions and the proposed design law

Let J be a task-viability score, tau a viable threshold, and B a bearer under a fixed actor
and update rule.

- **Load-bearing.** B is load-bearing on task T if freezing its update costs viability:

        J(B_self) - J(B_frozen) > epsilon.

- **Bearer-specific (non-transferable).** B is bearer-specific if updating it on another
  individual's outcomes, matched in volume, fails on one's own criterion:

        J(B_self) - J(B_wrong) > epsilon,   with J(B_self) - J(B_falsifier) ~ 0,

  where the falsifier is the same individual's outcomes with independent noise.

- **Escape route.** An escape exists for T if some policy pi_g that does not require
  incorporated bearer history reaches viability:

        E(T, tau):   there exists pi_g with J(pi_g) >= tau.

- **Foreclosure.** T forecloses the escape if no such policy exists:

        not E(T, tau):   there is no pi_g with J(pi_g) >= tau.

- **Adaptivity.** A(B) holds if the bearer can acquire, from its own outcome history, the
  task-specific state that viability requires.

**Proposed design law.**

    LB(B, T)  and bearer-specific(B, T)   <=>   ( not E(T, tau) )  and  A(B).

In words: a developmental bearer becomes load-bearing and bearer-specific when task
viability requires information that can only be acquired through that bearer's own
incorporated outcome history, and the bearer is capable of adapting to carry it. If a
non-incorporated viable policy remains, optimization preferentially exploits that route
instead.

The law separates into three regimes, which the substrates instantiate:

| Task structure | Bearer | Result |
|---|---|---|
| escape open (E) | any | bearer idle |
| escape closed (not E) | inadequate or static | difficulty valley (system fails) |
| escape closed (not E) | adaptive (A) | incorporated state becomes load-bearing |

We do not claim to have proven the universal biconditional. We claim cross-substrate
evidence for both directions of the mechanism, and we state below the experiment that would
occupy the presently thin cell (escape closed, adaptive) in a second substrate.

## 5. The loop and incorporation are substrate-independent

Delayed incorporation, a specific outcome leaving a residue that persists after its cue, is
measured in all three substrates: the forager (a delayed incorporation magnitude of 5.76 in
the continuant probe), the transformer (the memory integrates its own outcomes over a
horizon a bounded context cannot hold; freezing the write removes exactly that), and the
dynamical system (a one-shot perturbation persists about 120 steps, a memory horizon that is
0 without the bearer). The loop and its incorporation are not features of one architecture.

## 6. The escape principle is substrate-independent, and a scaling corollary

The bearer stays idle whenever the task leaves a non-incorporated route to viability.

- **Forager.** Across nine task designs and two capacity levers, self-coupling produced no
  bearer-neutral advantage, because each task retained a bearer-independent viable policy.
- **Dynamical system.** A clean null: transplanting a different body's incorporated bearer
  into body A, matched in volume, changes A's competency within noise (self-maintenance
  2.714 versus 2.716, basin divergence 0.003 versus 0.003 against a same-body falsifier),
  because A's fixed subsystem preferences are a non-incorporated route and the bearer is a
  weak add-on the dynamics wash out.

**The scaling corollary.** The two forager levers make the sharp point. Softening the
viability threshold re-admitted a quantity route, and giving the learner a richer update
rule made it exploit that route more efficiently, not the bearer.

    capacity up, escape present   =>   escape exploitation up,   not   incorporation up.

A more capable optimizer does not discover its bearer identity; it finds the escape better.
This is a design caution: increasing model capability, context length, retrieval quality, or
agent-loop sophistication is not evidence that a persistent developmental identity will
emerge. A sufficiently capable system may become better at avoiding the very channel one
wanted to make constitutive.

## 7. Foreclosure with an adaptive bearer yields load-bearing non-transferability

The transformer is the one substrate that forecloses the escape by construction and has an
adaptive bearer. Its task is stochastic bandit identification: a hidden per-instance code,
weakly informative outcomes, so identifying the code requires integrating outcomes over a
horizon far longer than any bounded context. Gate zero is an information bound, computed with
no training: a context-only learner restricted to the last L outcomes stays near chance
(0.170 at the locked calibration) while the full horizon is solvable (0.874). The context
window, the sequence model's most powerful escape, is foreclosed.

On that task the incorporated memory is load-bearing and bearer-specific by the definitions
of section 4.

- **Load-bearing.** J(B_self) - J(B_frozen) = 0.607: freezing the write collapses accuracy to
  exact chance. A dual-channel model with a real attention block gives the double
  dissociation, removing the attention record drops accuracy by 0.003 (record idle) while
  freezing the memory collapses it to the context bound (bearer carries the task).
- **Bearer-specific.** J(B_self) - J(B_wrong) = 0.509 plus or minus 0.049 across six seeds,
  wrong-bearer at exact chance, the falsifier holding and isolating the effect. It is not a
  one-bit lookup: a multi-dimensional vector identity shows the same non-transferability. And
  it holds at scale, reproducing at 133,637 parameters with a genuine transformer block, the
  gap if anything larger.

## 8. The difficulty valley

Foreclosure is necessary but not sufficient, which is the middle row of the regime table. On
a substrate whose bearer is not adaptive, foreclosing the escape leaves the identity
unsustainable. In the forager, an obligate task with the escape closed became unlearnable at
the smallest scales. In the dynamical system, a foreclosure attempt that made identity live
only in the bearer produced the effect at short horizon (basin divergence 0.103 under the
wrong bearer against 0.008 for the falsifier) but lost it at the full horizon (0.004 versus
0.004): a seeded identity is transient, and without an ongoing objective the loop converges
to a generic attractor and the identity washes out. The lesson is the same in both, and it
is why the law contains adaptivity: a bearer must integrate a body-specific parameter from
its own outcomes that behavior then requires. The transformer alone had both, which is why it
alone reaches the positive result. The valley explains why prior attempts to force persistent
identity yielded either nothing (escape open) or a broken learner (escape closed, static):
the middle variable, adaptivity, was missing.

## 9. Cross-substrate summary

| | forager (RL) | transformer (GPT) | dynamical system (FourD) |
|---|---|---|---|
| bearer loop (four loci) | yes | yes | yes |
| incorporation (Prediction 3) | ||db|| = 5.76 | horizon > context | memory horizon 0 -> 120 |
| escape principle (idle) | yes, 9 designs + 2 levers | n/a (foreclosed) | yes, self ~= wrong null |
| difficulty valley | yes (obligate unlearnable) | avoided | yes (seeded identity transient) |
| foreclosure + adaptive -> load-bearing | not reached | yes, freeze -> chance | not reached |
| bearer-specific (non-transferable) | not reached | yes, gap 0.509 +/- 0.049 | not reached |

## 10. Falsifiable predictions

- Any substrate with a surviving non-incorporated route shows an idle bearer, a null self
  versus wrong-bearer, regardless of capacity.
- Foreclosing that route, and only that, together with an adaptive bearer, makes the bearer
  non-transferable.
- The concrete stated test is the FourD adaptive foreclosure: give the dynamical system's
  bearer a task-driven objective that requires an integrated body parameter, then repeat the
  self versus wrong-bearer test. The prediction is that non-transferability then appears.

## 11. Discussion, limitations, and the three-paper program

The tasks are deliberately minimal so that the causal structure is legible; that is a
strength for a law and a limit for claims about rich behavior. The positive result's
substrate is a real attention model at 133,637 parameters, not a language model; the natural
next step is a gradient-trained language model with the same incorporated memory on an
obligate language task, where the escape to foreclose is again the context window. The
dynamical system remains an escape demonstration until its bearer is made adaptive.
Evolutionary search was noisy at the larger sizes; the self-versus-wrong result is reported
with across-seed error bars for that reason. The law is about causal load and
transferability, not experience.

This paper sits in the middle of a three-layer program, and the layers should be read as
distinct: Bearer Theory supplies the formal object and the developmental loop; this paper
supplies the empirical design law governing when that object becomes causally necessary; and
Life Before Language supplies the philosophical interpretation of how far such engineering
reaches toward consequence and subjectivity. Keeping the empirical layer free of the
phenomenal question is what makes it hard to dismiss as metaphysics dressed as machine
learning: here is a measurable property, bearer-specific developmental dependence, and its
governing conditions; what it means is a separate question, asked elsewhere.

## 12. Honest scope

Randomness quality was tested in the dynamical system, a pseudo-random source against the
operating system's cryptographic source under a two-one-sided-tests equivalence check, and
was found not to drive the incorporation effect: perturbation shape does, not entropy source.
A hardware quantum source was deliberately not claimed, since none was run, and an early
apparent quantum-randomness effect was traced to a shared-RNG-stream artifact and corrected.
Incorporation is a structural property of the loop, not of the noise.

## 13. Methods and reproducibility

The transformer code and results are public under MIT at
github.com/vikingdude81/constitutive-gpt, with the milestone map and trained policies, and
the forager foundation in the same repository. The entropy-decay probe is public at
github.com/vikingdude81/gabriels-horn. The dynamical system is the FourD repository, with the
bearer loop in bearer_state_competency.py and the self versus wrong-bearer and foreclosure
tests in fourd_self_vs_wrong_bearer.py and fourd_foreclosure.py. Training in the forager and
transformer is evolutionary search; the dynamical system is a fixed integrator; the gate-zero
bound is a Monte Carlo of the optimal decoder and needs no training. Per-experiment numbers
are in the RESULTS files of each repository.
