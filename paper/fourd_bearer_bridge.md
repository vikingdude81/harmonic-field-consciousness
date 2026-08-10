# FourD as a third bearer substrate: the bearer loop in a dynamical system

*Bridging note, 2026-08-08. Maps the FourD project's bearer state onto Bearer
Theory's four loci and Predictions 3 and 4, reports what its current results do and do
not show, and states the one experiment that would complete the correspondence. Keeps
the program's discipline: structural claim only, metaphysics bracketed. No em-dashes.*

---

## 1. What FourD is

FourD (github.com/vikingdude81/FourD) is a dynamical-systems model of cognition. N
subsystems act as tangent-force fields competing for control of a state on a manifold
(S3, S2, or flat R4), modulated by activity-dependent fatigue. Its documented results
are a manifold-independent saddle-node bifurcation in the order/disorder transition, a
persistent bearer state that produces transient path-dependence a reduced skeleton
does not, and partial recovery of manifold geometry from basin-transition graph
combinatorics. The FourD paper draft deliberately brackets the consciousness framing,
which is exactly the posture Bearer Theory takes toward the metaphysical gap.

## 2. FourD independently built the bearer loop

The `BearerEngine` update rule is Bearer Theory's decomposition in a different
notation:

    influences_t = base_influences_t + bearer_weight * b_t     perception biased by b
    activities_t = normalize(influences_t * fatigue + noise)    decision
    b_{t+1}      = (1 - bearer_decay) * b_t + bearer_gain * activities_t + xi_t   incorporation

with bearer_weight 0.6, bearer_decay 0.04, bearer_gain 0.35. The mapping onto the four
loci is exact:

| Bearer Theory locus | FourD |
|---|---|
| act, a = pi(b, w) | influences biased by b_t, then normalized to activities |
| constitute, o = V(e, b) | the outcome the system registers is filtered through b_t at perception |
| update, b' = U(b, o) | b_{t+1} = (1 - decay) b_t + gain * activities_t |
| recur, one continuant | b_t is a single persistent vector carried across all steps |

FourD's own words for the property, "a closed loop, not a passive log ... constitutive
rather than merely recorded," are Prediction 4 verbatim: the incorporated bearer versus
the retrievable record.

## 3. What the current FourD results show, honestly

The `bearer_state_competency.py` study (N=256, 1800 steps, 10 seeds, 3 manifolds x 2
topologies x bearer on/off) measures four competencies with an immediate-versus-
cumulative-deficit decomposition, which is a more careful ablation instrument than the
single ablation gate used in the CCA and GPT work. Two findings matter here.

**Prediction 3 is clearly present.** The bearer state creates a memory horizon where
there was none. With the bearer off the memory horizon is 0 by construction; with it on
it is about 120 steps across manifolds (s3/cyclic 120, flat4 124, s2 120). A one-shot
perturbation injected into b_t persists for roughly 120 steps before decaying below 1/e
of peak. This is the developmental-capture signature, the FourD analogue of the CCA
testbed's delayed incorporation (||db|| persisting after the cue).

**The on/off ablation is confounded, and instructively so.** The large, slow-recovering
competency deficits (lesion recovery about 867 to 899 steps, adaptation about 718 steps)
appear only in the s3/cyclic geometry, and they are roughly the same with the bearer on
or off (s3/cyclic lesion recovery 867 off versus 899 on). So the headline competency
effect is driven by the manifold geometry, not by turning the bearer on or off. This is
the same confound the CCA forager hit: an on/off ablation gate can be dominated by
another variable (there the observable-state shortcut, here the manifold), so it does
not isolate whether the incorporated state is load-bearing. The program already knows
the fix.

## 4. The self versus wrong-bearer test, run: the escape principle is substrate-independent

At the structural level the correspondence is strong and now threefold. The bearer loop
is implemented in an RL forager (CCA), a sequence model (Constitutive GPT), and a
competitive-fatigue dynamical system (FourD), and the developmental-capture or
persistence signature (Prediction 3) is measured in all three. That is genuine
substrate independence for the loop and for incorporation.

The decisive test was then run in FourD (`fourd_self_vs_wrong_bearer.py`). A body A is
run under three bearer streams: its own (self), a different body B's incorporated bearer
trajectory transplanted step by step (wrong), and a second same-body instance's bearer
with independent noise (falsifier). Competency is self-maintenance (clarity stability)
and basin divergence from the self run. Result, six bodies, s3/cyclic, N=96, 1500 steps:

    self-maintenance    self 2.714    wrong 2.716    falsifier 2.716   (all +/-0.006)
    basin divergence     wrong 0.003            falsifier 0.003

A clean null. Transplanting a completely different body's bearer changes A's competency
by about 0.002, within noise, and no more than transplanting a same-body bearer. The
incorporated bearer is not body-specific here.

The reason is structural and familiar. In FourD the body is its fixed subsystem
preferences and macro-centers, and the bearer enters only as a weak additive bias
(bearer_weight 0.6) that fatigue and normalization wash out. The body's own fixed
machinery is a non-incorporated route to its behavior, so the incorporated bearer is not
load-bearing. This is the Generic-Policy Escape Principle from the forager, now seen in a
dynamical system: the bearer is idle exactly because the task leaves a bearer-neutral
route. The escape principle is therefore itself substrate-independent, confirmed in two
unrelated formalisms.

This reframes the convergence. Three things are now substrate-independent: the bearer
loop and incorporation (Prediction 3), demonstrated in all three substrates; the escape
principle (the bearer is not load-bearing while a non-incorporated route survives),
demonstrated in the forager and FourD; and, in the one substrate that forecloses the
escape by construction, the Constitutive GPT, the positive result that the incorporated
state becomes load-bearing and non-transferable (Prediction 2, sharp Prediction 4). The
design law holds across substrates in both directions.

The foreclosure was then attempted, once, timeboxed (`fourd_foreclosure.py`). All
bodies were made physically identical (shared seed, so identical preferences), the
preference channel was weakened (prefs scaled to 0.15) and the bearer channel
strengthened (bearer_weight 2.5), and each body's identity was seeded into b_t and left
to be sustained only by the incorporation loop. At short horizon the effect appeared, a
clear basin divergence under the wrong bearer (0.103) against a held falsifier (0.008),
but at the full horizon measured on the tail it vanished (wrong 0.004, falsifier 0.004,
six bodies). The seeded identity is a transient. Without an ongoing objective anchoring
it, the loop converges to a generic attractor and the body-specific identity washes out
before the measurement window.

This is the difficulty valley again, and it is itself substrate-independent. It shows
precisely why the Constitutive GPT reached the positive result and FourD does not yet: a
seeded identity is not enough, the bearer must be adaptive and task-driven, integrating a
body-specific parameter from its own outcomes that behavior then requires, which is the
forager's obligate arc ported into FourD. That is a genuine build, not a quick run, and
it is left as stated future work. The honest status stands: FourD confirms the bearer
loop, incorporation, and the escape principle, and a sustained, load-bearing,
non-transferable bearer in FourD awaits an adaptive mechanism.

## 5. Two more connections

**The bifurcation and the wall.** FourD's transition is a saddle-node bifurcation, sharp
and manifold-independent, not a continuous critical point. That is a concrete
dynamical-systems instance of a sharp, substrate-independent threshold, which is what
Life Before Language means by the wall and what Schwitzgebel's determinacy feature
requires. It offers the graded-ladder-with-a-sharp-wall picture a real bifurcation
rather than a metaphor.

**A richer ablation instrument.** FourD's immediate-versus-cumulative-deficit
decomposition (d_immediate, cumulative shortfall, and t_recovery reported as NaN rather
than 0 when nothing was disrupted) solves the exact ambiguity the CCA program flagged,
where a recovery time of 0 conflates no disruption with instant recovery. This
instrument should be imported into the GPT and CCA ablation gates.

## 6. What this contributes

A third independent substrate for the bearer loop and for incorporation (Prediction 3),
which strengthens the substrate-independence claim from two formalisms to three. A
second independent confirmation of the escape principle, now in a dynamical system,
which strengthens the design law. A concrete bifurcation model for the wall. And a
better ablation instrument to bring back into the core experiments. The honest current
status: the bearer loop, incorporation, and the escape principle are substrate-
independent; the positive load-bearing non-transferability result holds in the one
substrate that forecloses the escape (the GPT); and the FourD foreclosure is the clear
next experiment if that direction is pursued.
