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

## 4. The convergence, and the one experiment that completes it

At the structural level the correspondence is strong and now threefold. The bearer loop
is implemented in an RL forager (CCA), a sequence model (Constitutive GPT), and a
competitive-fatigue dynamical system (FourD), and the developmental-capture or
persistence signature (Prediction 3) is measured in all three. That is genuine
substrate independence for the loop and for incorporation.

The load-bearing and non-transferability result (Prediction 2, and the sharp form of
Prediction 4) is confirmed in the GPT and is not yet isolated in FourD, because FourD
has run only the on/off ablation, which its own geometry confounds. The decisive
experiment is the self versus wrong-bearer test lifted directly from M3: run a
BearerEngine whose b_t is driven, in matched volume, by a different engine's activity
stream, and measure the competency vector on its own task. If competencies degrade under
the wrong bearer while the falsifier (its own bearer, independent noise) holds, FourD
joins the GPT in showing the incorporated state is not just persistent but specifically
this bearer's and non-transferable. FourD already has the infrastructure for it: a
pluggable perturbation source and the competency vector. It is a cheap run.

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
concrete bifurcation model for the wall. A better ablation instrument to bring back into
the core experiments. And a single, cheap, well-specified next experiment, self versus
wrong-bearer in FourD, that would extend the non-transferability result (Prediction 2
and the sharp Prediction 4) to the dynamical-systems substrate and make the convergence
paper airtight. The honest current status is structural convergence confirmed,
load-bearing specificity in FourD still to be shown.
