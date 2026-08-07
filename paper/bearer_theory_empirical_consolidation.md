# The bearer-identity experiments: a consolidated result

*Manuscript-ready synthesis of the CCA empirical program, 2026-08-05. Folds the
nine task designs, the curriculum, and the two calibration levers into one coherent
finding for the Bearer Theory manuscript (sections 8.3, 8.6, 9.9). No em-dashes.*

---

## 1. What the program set out to test

Bearer Theory predicts (Prediction 2) that when an agent updates on its own
bearer-relative outcomes, self-coupling should yield an advantage over a yoked
control that grows with bearer-indexing, and that this advantage is
criterion-internal: visible on a bearer-indexed criterion, muted on a
bearer-neutral one. The CCA continuant-forager was built to test this with a clean
self-versus-yoked comparison, matched in information volume, update rule, and
environment, differing only in whose consequences perform the update.

## 2. What is established

Two components of the theory are directly confirmed and are not in question.

- **Bearer-relative constitution is real.** The same environmental event constitutes
  a different effective outcome for different bodies: corr(V_self, V_wrong) = 0.79, a
  36 percent difference of signal (matched-event counterfactual, Prediction 1).
- **Historical incorporation is real.** A specific outcome leaves a residue in the
  continuant that persists after its cue: delayed incorporation ||db|| = 5.76
  (testbed extensions, Prediction 3).

The outcome-driven plastic pathway is also load-bearing under drift: the ablation
gate passes on a trained agent (plastic on -0.55 versus off -0.80).

## 3. The central negative and its structure

Across nine task designs, a curriculum, and two calibration levers, self-coupling
produced no bearer-neutral performance advantage. This is not a scattering of null
results. It has a single structure.

**The Generic-Policy Escape Principle.** When a task admits any bearer-independent
policy that preserves viability, a flexible agent satisfies the task without making
its private constitution behaviorally load-bearing. Every early design contained
such an escape: timing, quantity, reduced exposure, or uniform consumption. Each
null is that escape being taken.

**The obligate construction and the difficulty valley.** Foreclosing the escape (the
private-metabolism task, where uniform eating starves and no universal diet exists)
removes the bearer-neutral policy, but then the only viable policy requires inferring
a private demand vector blind, which the substrate could not learn. Escape available
means the agent trains via the escape and shows no bearer effect; escape foreclosed
means the agent cannot train at all. The two are one phenomenon seen from both sides.

**The Bearer Observability Window.** An observable bearer effect requires three
conditions at once: no escape (E = 0), the bearer behavior on the incorporated
pathway rather than an observable shortcut (the ablation gate passes), and the
policy learnable (L = 1). Observability enables learning but provides a bypass that
fails the gate; blindness forces the pathway but blocks learning. The window is
squeezed from both sides.

## 4. The new result: the escape is meta-level

The two calibration levers were run to reach the window by tuning rather than by
task design. Both failed, and they failed in a way that upgrades the finding from a
limitation into a principle.

- **Lever A, soften the viability threshold.** A gain multiplier on the net-energy
  function, pinned so uniform still starves. Strict (gain 2): selective but
  sub-viable, gate dark. Soft (gain 5): viable and the gate opens, but selectivity
  collapses to near-uniform (mediator 0.542, a quantity route). Softening enough to
  be viable re-admits a near-uniform, high-volume quantity escape.
- **Lever B, strengthen the learner.** Replace the fixed gated-Hebbian rule with a
  4-term learned update rule, ES-optimized. At the same operating point the richer
  rule raised viability to the highest of any run (advantage +0.039) and lowered
  selectivity further (mediator 0.507, uniform). More expressive plasticity did not
  buy bearer-selectivity. It bought a better escape.

Together these show that the Generic-Policy Escape Principle operates at the meta
level. A more capable optimizer does not discover its bearer identity. It exploits
the bearer-neutral route more efficiently. Softening viability re-creates an escape;
strengthening the learner exploits it better. The barrier is not a weak learner or a
steep threshold. It is structural: whenever any bearer-neutral path to viability
survives at the operating threshold, increasing capacity increases escape, not
incorporation.

## 5. What this contributes to the theory

This sharpens Prediction 2 and its criterion-internal qualification into a design
law with a mechanism.

    Causal bearer identity does not become adaptive leverage merely because it
    exists. The environment must make it necessary, and necessary in a strong sense:
    the task must admit no bearer-neutral route to viability at the operating
    threshold. Any residual route will be found, and found better as the learner
    improves.

The unifying statement is that incorporation becomes load-bearing exactly when the
information the agent needs is not available from an observable or retrievable
source. This subsumes Prediction 4 (record versus incorporated bearer): a record is
a retrievable source, and any observable or retrievable substitute is an escape that
capacity will exploit.

## 6. The consequence and the next substrate

The window cannot be reached in the ES-plus-fast-weight forager by tuning, because
both the threshold and the update rule are absorbed by the escape. The requirement
is a task that forecloses every bearer-neutral route at viability by construction,
on a substrate whose incorporated state can carry information that observation and
retrieval cannot supply.

That is the Constitutive GPT: an incorporated private state that gates value and
policy, on a task whose required information cannot be held in context or read from
observation, so that no in-context, retrieval, or quantity route substitutes for the
bearer. The lesson from Levers A and B travels with it. Raw capacity alone will find
any escape that exists, so the task, not the learner, must close the door. The
forager program is complete; it has produced a design law and named the one
construction that could satisfy it.

## 7. The Constitutive GPT: the window reached and Prediction 2 confirmed

The design law was then built and tested on a sequence model, in three milestones.
This is the positive half of the program: the forager showed when a bearer effect
cannot be observed, the Constitutive GPT shows it when the escape is foreclosed by
construction.

**M1, the obligate task and gate zero.** A stochastic bandit-identification task is
the sequence-model analogue of the private-metabolism forager. Each instance has a
hidden correct arm, the correct arm pays slightly above one half and the rest exactly
one half, so identifying it requires integrating noisy outcomes over a horizon far
longer than any short context. Gate zero, computed as an information bound rather than
a trained model, is the decisive check: a context-only learner restricted to the last
L outcomes must stay near chance while the full horizon is solvable. At the locked
calibration (eight arms, L equal to thirty two, horizon four thousand) a context-only
learner reaches 0.170 against a chance of 0.125, while the full horizon reaches 0.874,
a five-fold separation. The context window, which is the sequence model's most
powerful escape, is foreclosed by construction. This is Prediction 4 with the
sharpest edge: a record, here the context, is a retrievable source, and the task is
built so no retrievable source suffices.

**M2, the incorporated model.** The incorporated state is a fast-weight associative
memory, the sequence-model form of the bearer fast weights: learned arm embeddings, a
reward-gated Hebbian write on the agent's own outcome, and a bilinear read that
returns a per-arm value estimate. Nothing is hand-indexed; evolutionary search learns
the embeddings, the read and write vectors, the gains, and the policy. It reaches
0.732 held-out at horizon two thousand and 0.858 at the full horizon, against the
0.874 ceiling. The ablation gate is decisive: freezing the write collapses accuracy to
exact chance, 0.125, a drop of 0.607. The incorporated state does essentially all the
work. This is the first time the program occupies the observability window, viable and
on the incorporated pathway and learnable at once, which the forager never reached.

**M3, self versus wrong-bearer.** The sharpened Prediction 2, in the same lambda-sweep
form as the forager. The write is a lambda-mix of the instance's own outcome stream
and a different instance's, and accuracy is scored on the instance's own private code.
Self, at lambda zero, reaches 0.633 and falls monotonically to 0.118, chance, at
lambda one, a gap of 0.515. The falsifier, where the alt stream carries the same code,
does not collapse: it stays at 0.605 at lambda one, isolating the bearer effect by
0.487. Writing the incorporated state with a different bearer's outcomes is nonviable
for this instance; writing it with the same bearer's outcomes is fine. With the M2
ablation gate showing the memory is load-bearing, the flat falsifier showing the gap
is the bearer difference and not generic updating, and matched information volume
across conditions, the criterion-internal reading is ruled out. The gap is a real
consequence of bearer-relative incorporation.

## 8. Status line for the manuscript

The CCA program confirms bearer-relative constitution (Prediction 1) and historical
incorporation (Prediction 3), and it establishes, through nine designs and two
capacity levers, that a bearer-identity performance advantage is observable only under
an obligate task with no bearer-neutral escape at threshold, a positive structural
result about when identity is load-bearing. The Constitutive GPT then satisfies that
law and confirms the sharpened Prediction 2 directly: on a task whose escape is
foreclosed by construction, the incorporated state is load-bearing and bearer-specific,
with a self versus wrong-bearer gap of 0.515 and a falsifier that isolates it. The
program thus has both halves, the boundary condition and the positive effect, on two
independent substrates.
