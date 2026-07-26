# A Second Substrate for the Bearer-Indexing Result: What the CCA Forager Tested and What It Found

*Companion note to "Life Before Language." Written July 2026. Maps the
continuant-probe/cca experiments onto the paper and proposes where they belong.*

---

## 1. What this is

The paper's sharpest empirical claim lives in Section 8.4, the Continuant Probe.
It says bearer-identity becomes load-bearing exactly when outcomes stop being
facts about the world and become facts of the bearer, and it reports a specific,
carefully qualified result: an agent altered by its own outcomes does better by
its own assigned criterion (+0.909, 100% of seeds), and no better on a
bearer-neutral yardstick (world food, +0.002, 55% of seeds). The paper calls
this the criterion-internal qualification.

This note reports an independent second substrate built to test that same claim.
It is a different architecture, a different world, and a different measurement
pipeline. The purpose was to see whether the bearer-indexing result reproduces,
and if so, in what form. The short answer: the null side reproduces robustly,
the +0.909 side does not, and understanding why sharpens the paper rather than
weakening it.

---

## 2. The new substrate: the CCA forager

The original Continuant Probe was a lineage runtime plus a tabular drifting-world
agent. The CCA forager (Constituted-Consequence Agent, Milestone 1) is
deliberately unlike it:

- A 32x32 world with resources and hazards, egocentric vision, and homeostasis
  (energy and integrity that must stay in band).
- Each agent has a private constitution: a fixed, never-observable body that
  sets how nutritious each resource is for it, how vulnerable it is to each
  hazard, and what each action costs. This is the Level-3 innate orientation of
  the paper's hierarchy, rendered as a private body.
- The agent is a recurrent network with an outcome-gated plasticity pathway:
  fast weights and a bearer state that are written by the outcomes the agent
  experiences, not by an external trainer.

The bearer manipulation is one line. The agent lives its own life, and every
outcome is valued either through its own constitution (the *self* condition) or,
counterfactually, through a different constitution (the *wrong_bearer*
condition): same action, same event, same dice, different body. A dial, lambda,
mixes a world-fact component and a bearer-relative component of the outcome, so
lambda = 0 is a pure world fact and lambda = 1 is fully bearer-relative. At
lambda = 0 the two conditions are identical by construction, which gives a
built-in falsifier: the self-minus-wrong difference must be zero there, or the
plumbing is broken.

Why a second substrate matters: a result that appears in one hand-built world
could be an artifact of that world. If bearer-indexing is real, it should show
up in a system designed independently. If it does not, that tells us the effect
is fragile or criterion-internal, which is itself a finding.

---

## 3. What was built

The pipeline is worth naming because it is reusable and because two of its
pieces are the reason the results can be trusted.

- A GPU batched-population evaluator. It runs an entire evolutionary population
  as one tensor in a single process, so the throughput rises with population
  size instead of falling. It was parity-checked against the plain CPU
  implementation: matched world initialization, matched fitness distributions
  within noise. The numbers below come from the GPU path but mean the same thing
  the CPU path would.
- Held-out training. Agents are meta-trained by evolution strategies, and the
  champion is selected by performance on worlds it never trained on. This is the
  guard against the failure that ruined the first attempt, where a champion
  looked good only on the specific worlds it was selected on.
- The ablation gate. Before any bearer sweep is believed, the pipeline checks
  whether the outcome-driven pathway is load-bearing at all: it freezes the fast
  weights and bearer state and measures how much performance drops. If freezing
  the pathway costs nothing, the agent solved the task some other way, the bearer
  manipulation has no surface to act on, and any null result is meaningless. The
  gate is the methodological contribution of this work and it is discussed in
  Section 6.

---

## 4. The experiments: a ladder of necessary conditions

The investigation was not one experiment. It was four, each removing a reason the
previous null could not be interpreted. This structure is the real content.

**Sweep 1: static world.** Train a champion in a fixed world, then run the
bearer sweep. Result: null, and uninterpretable. The ablation gate failed:
freezing the plastic pathway changed performance by 0.005. A fixed world is
solvable by a fixed policy, so the agent never needed to adapt from its own
outcomes, so the pathway the bearer manipulation acts on sat idle. Lesson:
bearer-indexing can only have purchase where online adaptation is necessary.

**Sweep 2: drifting world.** Retrain in a world whose resource values and hazards
move over time, so a fixed policy decays and the agent must keep adapting.
Result: the gate now passes decisively. Freezing the pathway drops the agent from
-0.55 nearly to the random baseline, a cost of +0.24 (versus 0.005 in the static
case). The bearer sweep, scored on homeostasis, is null, with a faint lean in the
predicted direction. This is the first interpretable measurement, and it
reproduces the paper's null on a bearer-neutral yardstick. A check confirmed the
manipulation was real, not trivially weak: the self and wrong-bearer outcome
signals correlate 0.79 and differ by 36% of signal magnitude. The wrong body
genuinely changes the learning signal. Performance simply does not care.

**Sweep 3: bearer-indexed criteria.** Score the same drift champion on the
distinction the paper turns on. Add a bearer-neutral yardstick (raw food
collected, the same for any body) and a clean bearer-indexed one (diet quality,
the average own-body value of what the agent chose to eat). Result: no
significant effect, but a faint signature. On diet quality the self-minus-wrong
difference leaned positive at every lambda above zero (up to +0.17), the
predicted direction, while the bearer-neutral raw-food metric scattered in sign.
That contrast is the criterion-internal pattern the paper describes, present but
underpowered.

**Sweep 4: blind agent.** Remove the homeostatic channels from the agent's
observation, so it cannot sense hunger and must learn its own body through
consequence. The outcome signal still feeds the plastic pathway, so the bearer
manipulation keeps its channel; only the reactive shortcut is cut. Prediction:
this uncaps taste selectivity and makes the effect significant. The prediction
was falsified. The blind agent trained just as well (+0.34) and the gate passed
larger than ever (+0.335), so the setup was ideal, yet diet quality stayed near
random (~0.8) and self versus wrong-bearer was null on all four metrics. The
faint lean from Sweep 3 did not survive more seeds. It was noise.

---

## 5. The headline finding

Across four sweeps that escalate the necessary conditions, bearer-identity of
consequence produces no measurable bearer-neutral performance effect in this
forager. This holds even when consequence-driven adaptation is necessary, even
when the pathway is load-bearing, and even when that pathway is the only channel
the agent has. The negative was reached by exhausting the obvious confounds, not
by a weak test.

The reason is mechanical and clear. The blind agent regulates by learned timing
and rate: it eats when the pathway signals that its unobservable energy is low,
and it eats whatever is nearby. It does not regulate by which resources it eats.
Diet quality stays near random because homeostasis needs enough total energy, not
the right resources, so taste selectivity is never incentivized. Bearer-identity
could only matter through selectivity. With no selectivity pressure, whose body
values the outcome does not change what the agent does.

---

## 6. How this relates to the paper, section by section

**Section 2 (the hierarchy) and Section 5 (Workstream 2).** The CCA forager is a
Level-3 probe: a private, unobservable constitution is exactly the innate
orientation the hierarchy places at Level 3, and studying it to locate the wall
is Workstream 2's remit. The forager gives that workstream a second, independent
instrument.

**Section 8.4 (the Continuant Probe) is the main point of contact.** The forager
tests the same claim in a new substrate. It reproduces the null on bearer-neutral
yardsticks robustly (raw food, homeostasis, across four experiments). It does not
reproduce the +0.909 on the assigned criterion. The likely reason is structural,
and it is worth stating plainly: in the original probe the assigned criterion was
scored on the very taste-filtered quantity the self-condition was trained to
maximize, so the self agent was almost tautologically aligned with the score. The
forager decouples regulation (timing) from taste, so no such alignment exists,
and the effect vanishes. This does not contradict the paper. It confirms the
paper's own qualification: the bearer advantage is criterion-internal, and it
does not appear as a bearer-neutral performance gain. A second substrate now says
the same thing, and refuses to inflate it.

**Section 8.6 (bounds) and the companion-paper pointer.** Section 8.6 promises a
follow-up on the separability of constitution and consequence-bearing and on the
coupling family. The CCA work is concrete material for that companion paper, and
it adds a result Section 8.6 does not yet claim: a required precondition for the
whole probe methodology, described next.

**Section 9 (the wall).** The forager is another case of constitution without
consequence-bearing. Its agents are changed by their formation and their
outcomes, and nothing matters to them. It sits on the near side of the wall with
everything else the paper measures. It adds no crossing, only more resolution on
the near side.

**Paper 2 outline (the assigned-residue argument).** This is where the CCA
findings most naturally live. The assigned-residue argument says self-referential
outcome-coupling gives a system a criterion it can trivially satisfy, and that
the apparent advantage is residue of the assignment rather than a real gain. The
forager is direct evidence for that argument: the bearer advantage is
criterion-internal and fragile, and disappears in a substrate where the criterion
is not handed to the agent by construction.

---

## 7. What it fills, and proposed additions

Three concrete holes this work can fill.

1. **Replication for Section 8.4.** The paper reports the criterion-internal
   result at one substrate. A single sentence and a footnote can now say it was
   tested in a second, independently built substrate, where the bearer-neutral
   null reproduced and the assigned-criterion effect did not, consistent with the
   effect being criterion-internal. Proposed wording:

   > A second substrate (an independently constructed foraging agent with an
   > outcome-driven plasticity pathway) reproduces the bearer-neutral null across
   > four escalating conditions. It does not reproduce the assigned-criterion
   > advantage, which there is traceable to the score being defined on the same
   > quantity the self-condition optimizes. The bearer advantage is therefore
   > criterion-internal in both substrates, not a bearer-neutral performance
   > gain.

2. **A methodological principle for the whole probe program.** No bearer or
   consequence result should be believed unless the system demonstrably depends
   on the pathway the manipulation acts on. This is the ablation gate. It
   deserves a short methods paragraph, because it is the difference between an
   interpretable null and a meaningless one, and Sweep 1 shows the failure mode
   concretely.

3. **A necessary-condition ladder.** The paper argues bearer-indexing has
   purchase only where outcomes are facts of the bearer. The four sweeps give
   that argument operational teeth: the effect needs a world that requires online
   adaptation, a pathway that is load-bearing, and a criterion the agent does not
   trivially satisfy. State those as the conditions under which the question is
   even askable.

---

## 8. Bounds and the one remaining lever

Everything here is small-scale and single-lineage, with the same honesty the
paper applies to itself. The forager is a toy. The champions plateau, and 2x
parameters bought nothing, so capacity is not the limit.

One lever is genuinely untested. Selectivity is never incentivized because eating
a resource that is bad for your body is not harmful enough at the current
constitution strength. Raising that strength, so that own-poison resources do
real damage, would make taste selectivity load-bearing and force the agent to
avoid what is bad for it specifically. Whether a bearer effect appears there, and
how it scales with constitution strength, is the paper's dose curve rebuilt in
this substrate. It needs retraining at each strength to keep the agent in
distribution. It is the natural next experiment and it is the one place the
current negative could still turn positive.

---

## 9. One-paragraph version

We built a second, independent test of the paper's bearer-indexing result: a
foraging agent with a private body and an outcome-driven learning pathway, run
through a GPU pipeline with a built-in check that the pathway is actually used.
Across four experiments that each removed a reason a null could be dismissed, the
bearer-neutral result reproduced every time: whose body a consequence is valued
through does not change how well the agent does. The one place the original probe
found a large effect, the agent's own assigned criterion, did not transfer,
because that effect came from scoring the agent on the very thing it was trained
to maximize. This is exactly what the paper already says in its criterion-internal
qualification. A second substrate now confirms it and declines to overstate it.
The bearer advantage is real but criterion-internal, not a bearer-neutral gain,
and the wall stands where the paper put it.
