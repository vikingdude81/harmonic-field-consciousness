# The Constitutive GPT: an incorporated sequence model where bearer identity is load-bearing

*Addendum to Bearer Theory, 2026-08-06. A self-contained report of the Constitutive
GPT experiment (milestones M1 to M3), the sequence-model successor to the Chapter 9.9
testbed. It confirms the sharpened Prediction 2 on a substrate that reaches the
observability window. Written to stand alone as an appendix or companion note. No
em-dashes.*

---

## Abstract

The forager program established that bearer identity can be causally real, through
matched-event constitution (Prediction 1) and delayed incorporation (Prediction 3),
yet found no bearer-identity performance advantage across nine task designs and two
capacity levers. It reduced that null to a design law: a bearer-identity effect is
observable only when the task admits no bearer-neutral route to viability at the
operating threshold, and any residual route is exploited more, not less, as the
learner improves. This addendum builds and tests the one construction the law
permits. A small sequence model carries an incorporated fast-weight state whose only
long-horizon competitor, the context window, is foreclosed by construction. On that
substrate the incorporated state becomes load-bearing (freezing it collapses
performance to chance) and bearer-specific: updating it on the instance's own outcomes
succeeds, updating it on a different instance's outcomes fails on the instance's own
criterion, with a self versus wrong-bearer gap of 0.515 and a falsifier that isolates
the effect. This is the sharpened Prediction 2, confirmed.

## 1. Motivation: the design law

The forager result, consolidated separately, is that the Generic-Policy Escape
Principle operates at the meta level. Softening the viability threshold re-admits a
bearer-neutral quantity route, and strengthening the inner-loop learner exploits that
route more efficiently rather than discovering the bearer. The barrier is structural.
It follows that to observe a bearer effect one must foreclose every bearer-neutral
route by construction, on a substrate whose incorporated state can carry information
that observation and retrieval cannot supply. For a sequence model the primary route
to foreclose is the context window itself, since in-context attention over past
inputs and outcomes is a retrievable record, exactly the record that Prediction 4
distinguishes from an incorporated bearer.

## 2. The obligate task and gate zero (M1)

The task is stochastic bandit identification, the sequence-model analogue of the
private-metabolism forager. Each instance draws a hidden correct arm theta from K
possibilities, uniform and unobservable. Each step the agent pulls an arm and receives
a noisy reward: the correct arm pays Bernoulli(0.5 + eps), every other arm pays
Bernoulli(0.5). The gap eps is small, so one pull is weakly informative and
identifying theta requires integrating outcomes over many pulls. No universal policy
beats chance, because each instance has a different theta, so the task is obligate:
the only way above chance is to infer theta from the agent's own outcome stream.

Gate zero is the decisive precondition. A context-only learner, one that can use only
the last L outcomes, must stay near chance, while a full-horizon integrator succeeds.
If a short window sufficed, attention would do the inference and the incorporated
state would sit idle. The bound is computed against the Bayes-optimal decoder given n
uniformly-explored pulls, which upper-bounds any context-only model including a
transformer attending over its context, so no model is trained to establish it.

Calibration. With K equal to four the chance floor is high and even sixteen pulls
leave a detectable edge, so the task leaks. Raising to K equal to eight lowers the
chance floor and widens the statistical transition, which spans about two orders of
magnitude in pulls. The locked calibration is K equal to eight, eps equal to 0.06, L
equal to thirty two, horizon N equal to four thousand. The accuracy of the optimal
decoder as a function of pulls:

    pulls       8    16    32   128   256   512  1000  4000
    accuracy 0.144 0.158 0.170 0.220 0.280 0.378 0.500 0.874

Context-only (L equal to thirty two) sits at 0.170 against a chance of 0.125, while
the full horizon reaches 0.874, a five-fold separation. The context escape is
foreclosed by construction.

## 3. The incorporated model (M2)

The incorporated state is a fast-weight associative memory W, a matrix reset at the
start of each episode, the sequence-model form of the forager's bearer fast weights.

- Read. For each arm k, v_k equals u dot (W e_k), a bilinear read against a learned
  arm embedding e_k. With near-orthonormal embeddings, v_k accumulates the gated
  reward written for arm k, so v is a per-arm value estimate.
- Policy. The action is sampled from softmax(beta v). The agent explores when W is
  empty and v is flat, and exploits as W fills.
- Write, the incorporated update. On the agent's own outcome, W is updated by
  W plus c_write times o times outer(w_val, e_a) minus c_decay times W, keyed by the
  pulled arm a and gated by the centered reward o. This is an outcome-gated Hebbian
  write into the memory.

Nothing is hand-indexed. Evolutionary search optimizes the arm embeddings, the read
and write vectors, the write and decay gains, the reward centering, and the policy
temperature, about three hundred and twenty parameters at width thirty two. The
question is whether the search finds a setting where the memory carries the
integration, not whether one can be hand-built. It does. The trained model reaches
0.732 held-out at horizon two thousand and 0.858 at the full horizon four thousand,
against the 0.874 ceiling.

The ablation gate is the mediator check for this substrate. Freezing the write, by
setting c_write to zero, collapses accuracy to exact chance, 0.125, a drop of 0.607.
The incorporated state does essentially all the work; it is load-bearing, not an idle
channel. This is the first time the program occupies the observability window: viable,
on the incorporated pathway, and learnable, at once.

## 4. Self versus wrong-bearer (M3)

The sharpened Prediction 2, in the same lambda-sweep form as the forager bearer sweep.
The agent acts from its incorporated memory as in M2. The outcome that writes the
memory is a lambda-mix of two reward streams for the same action:

    o equals (1 minus lambda) times (r_own minus shift) plus lambda times (r_alt minus shift)

where r_own is the reward under the instance's own arm theta_own and r_alt is the
reward under a different instance's arm theta_alt. Accuracy is always scored on the
own code. At lambda zero the memory is written by own outcomes and identifies
theta_own; at lambda one it is written by the alt outcomes and identifies theta_alt,
which matches theta_own only by chance.

Results, scored on two hundred and fifty six held-out instances:

    lambda    self..wrong    falsifier (alt equals own)
    0.00        0.633              0.646
    0.25        0.438              0.684
    0.50        0.288              0.729
    0.75        0.191              0.692
    1.00        0.118              0.605

Self, at lambda zero, reaches 0.633 and falls monotonically to 0.118, chance, at
lambda one, a gap of 0.515. The falsifier, where theta_alt equals theta_own so the
two streams carry the same code, does not collapse: it stays at 0.605 at lambda one.
The bearer effect isolated at lambda one, the falsifier minus the wrong-bearer, is
0.487. Writing the incorporated state with a different bearer's outcomes is nonviable
for this instance; writing it with the same bearer's outcomes is fine.

The falsifier rises slightly at intermediate lambda, from 0.646 to 0.729 and back to
0.605. This is benign and non-monotone: at intermediate lambda the write averages two
independent same-code reward streams, which reduces write noise and slightly raises
accuracy. It is variance reduction, the opposite shape from the wrong-bearer collapse,
and the correct test is the divergence at lambda one rather than the flatness of the
whole line.

## 5. Why this is not a scoring artifact

The forager program's hardest lesson was that a self versus wrong gap can be
criterion-internal, an artifact of scoring an agent on the very quantity its
self-condition optimizes. Three facts rule that out here.

- The M2 ablation gate shows the incorporated memory is load-bearing: freezing the
  write collapses accuracy to chance. The mediator is established, not assumed.
- The falsifier shows the gap vanishes when the alt code equals the own code. The gap
  is driven by the bearer difference, not by any generic property of updating.
- Information volume is matched. Both streams deliver the same number of Bernoulli
  rewards with the same marginal statistics. Only whose code the outcomes reflect
  differs.

The gap is therefore a real consequence of bearer-relative incorporation.

## 6. Relation to Bearer Theory

The Constitutive GPT realizes the four loci directly. The read of W into the policy
is the actor conditioned on the incorporated state; the reward gated by the arm choice
is the outcome constituted relative to the bearer, o equals V of the event and the
bearer; the Hebbian write is the update, b prime equals U of the bearer and the
outcome; and the persistence of W across the episode is the continuant. The double
entry of the bearer state, into both the interpretation and the policy, matches
Chapter 3. The write to W rather than to an external buffer is the incorporated
update of Chapter 4. The foreclosure of the context window is Prediction 4 made
sharp: the record is retrievable and is built to be insufficient, so only the
incorporated state suffices. The self versus wrong-bearer comparison is the sharpened
Prediction 2, and it is confirmed.

A dual-channel model demonstrates the record-versus-bearer distinction inside a single
sequence model rather than from the gate-zero bound alone. A real single-head
attention over the last L action and reward tokens, a bounded transformer context, is
added alongside the incorporated memory, with learned per-channel weights, and the
search is free to lean on either. The result is a clean double dissociation. Ablating
the context, which removes the record, drops accuracy only from 0.732 to 0.703, so the
record is idle. Freezing the incorporated memory, which removes the bearer, drops
accuracy to 0.160, a fall of 0.572, and 0.160 is the gate-zero context-only bound of
0.170, so the context on its own reaches exactly its information ceiling and no more.
The learned context weight, 0.44, is small next to the memory weight, 1.72. The model
has attention, could use the record, and does not; the incorporated state carries the
task. With M3, where that state is bearer-specific and non-transferable, both halves
of Prediction 4 now hold: a record cannot substitute for the bearer, and the bearer
cannot be swapped for another.

## 7. Methods and reproducibility

All code is in the continuant-probe repository, cca directory. The task and gate zero
are in constitutive_gpt_m1.py, the incorporated model in constitutive_gpt_m2.py, and
the self versus wrong-bearer sweep in constitutive_gpt_m3.py, with the trained policy
saved as champ_m3.npz. The dual-channel model is in constitutive_gpt_dual.py, with its
policy saved as champ_dual.npz. Training is evolutionary search: population ninety six,
mirror sampling, centered-rank utilities, at width thirty two and training horizon two
thousand, evaluated on held-out instances. The gate-zero bound is a Monte Carlo of the
optimal decoder and requires no training. Results are recorded in RESULTS_M1.md,
RESULTS_M2.md, RESULTS_M3.md, and RESULTS_DUAL.md.

## 8. Limitations and next steps

The task is a single hidden scalar per instance, the simplest obligate code, and the
incorporated memory is a single fast-weight matrix. A richer, structured private code
is the natural next probe of whether the effect is genuine identity rather than a
one-dimensional lookup. The attention channel, shown idle here on a bounded context,
would face a stronger test under a longer context or a task where the record carries
partial but tempting information. Evolutionary search was noisy late in training and
the returned policy is the best held-out checkpoint; seed replicates with error bars
on the self versus wrong-bearer gap, a smaller step size, or a larger population would
harden the numbers. M4, scale, is the natural
successor, and it is warranted precisely because the causal relations are now
established, the precondition Chapter 9.9 sets before scaling. The result to carry
forward is that the incorporated state is load-bearing and bearer-specific once the
task forecloses every bearer-neutral route, which is the positive form of the
program's design law.
