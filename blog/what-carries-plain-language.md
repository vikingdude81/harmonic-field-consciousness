# What Carries Across, and What Doesn't: The Story So Far

*A plain-language tour of the Bearer Theory experiments — what each machine shows, what we got wrong, and where the road points. Not a paper. Numbers live at the bottom.*

---

## The question, said plainly

Somewhere between a thermostat and a person, something changes. A thermostat responds to the room but nothing is at stake for the thermostat. A person who touches a hot stove is *changed by the outcome* and goes on living as the changed person — and something about that loop is what "having stakes" seems to mean.

This research program asks one deliberately small version of a very large question:

> **In a system we build, when does the thing that gets changed turn out to be the bearer itself — and when does the bearer finally *care* that it changed?**

It makes no consciousness claims. It never says a model is "aware" or "sentient." It says: here is a ladder of precisely-different capabilities, we can build rungs of it in code, measure which ones a given system has, and watch what breaks. That's it. And it turns out that's already surprising enough.

## The ladder

Four rungs, each one testable:

1. **Modulation** — some structure shapes what the system does *right now*, without touching what it *is*. (A dimmer switch on a lamp.)
2. **Internalization** — the process of being formed *under* that structure changes what the system is, permanently, even after the structure is gone. (A child who learned about fire doesn't need the stove present anymore.)
3. **Consequence** — the results of the system's own actions flow back into the *same continuing system* that acted, and it proceeds as the altered entity. (Not a record that updates — the actor itself updates.)
4. **Subjectivity** — there is someone *for whom* the whole unfolding is had.

Rung 4 is where the big claims live, and we keep it deliberately empty. Everything below is about rungs 1–3, measured.

## Five machines, one wall

The first phase built five deliberately different architectures — five ways of bolting something like memory, regulation, or selection onto a language model — and asked each the same question: *when this layer is removed, is the system that remains different from the system that would have been there without the layer?*

- **The dimmer on reasoning** (Gabriel's Horn). We varied the "temperature" of a model's reasoning step by step. Output changed, sometimes a lot. The model itself: bit-for-bit identical, always. The dimmer shapes what it *says*, never what it *is*.
- **The gatekeeper** (LGI executive). A regulator watches each reasoning state and allows, dampens, or blocks. Remove the gate and the original model runs untouched. Sensing without consequence: violating the rules costs the system nothing it bears.
- **The holographic notebook** (FTTF). A memory store where every trace is superposed into one high-dimensional object. Two measurements, and they're the pair that made people sit up: *having* the notebook attached measurably **degrades** unrelated tasks (the model pays a tax to carry it), while scribbling false entries into it changes **nothing**. Compare a human: having memory is free, corrupting memory is devastating. The machine is the exact mirror image of the biological case.
- **The lesson-logger** (self_learner). The system distills lessons after each session and pastes them into tomorrow's prompt. Delete the lessons table: same model, same competence. Accumulation that never touches the accumulator.
- **The evolution loop.** Here consequences are real — bad individuals genuinely don't propagate. But they land on the *lineage*, never on the individual who bore them. Differential consequence, wrong bearer.

Five designs, as unrelated as we could make them, one verdict each time:

> **The bearer is not the thing that gets changed.**

That sentence became the program's signature — and then we went to work on breaking it.

## The turn: grow it in, don't bolt it on

Here's the trick that made the whole program pay off. A rule can enter a system three ways:

- **A sticker on the wall.** The rule is present at runtime, the system was formed without it. It's pure overhead — a *tax*.
- **A cane.** The system was formed, *then* given the rule and retrained around it. Now it leans on the cane. Remove it and the system stumbles — a *crutch*.
- **A shape the body grew into.** The rule was there from the very first moment of training. The system grew into the allowed space and never knew another. Remove the rule and — nothing happens. Because the rule isn't in the regulator anymore. **It's in the shape of the weights.** *Internalized.*

And this is where the wall developed its first crack, which we measured two independent ways:

1. **The scaffold is gone, the scar remains.** Grown-in models show essentially zero runtime dependence on the regulator that shaped them — while bolted-on models retain a large one. And retraining bolted-on models *deeper* approaches but never reaches the grown-in signature: an irreducible gap of roughly 3× survived every dose, scale, and tokenization we tried. History leaves a residue you cannot retrain away — though, humblingly, we found the *shape* of the approach-to-the-floor is itself environment-dependent. The gap is universal; the smooth convergence story was one lab's local weather.
2. **You can see the sediment.** With the regulator fully off, grown-in models keep their internal activity *away* from the forbidden region of their own state space — at a remarkably constant depth (a bit under half of what chance would give) across five very different configurations. The training history is written into the geometry of the finished thing. Bolted-on models, retrained enough, can *copy that geometry* — an important humility check that killed one of our favorite claims (more on that below).

Then the same asymmetry showed up **with a different kind of layer**: not a geometric constraint but a *memory record*. A model that grew with its memory genuinely depends on it now; a model handed the same memory afterward barely uses it. And it showed up again **inside a single lifetime**, not across training: an agent whose record is wiped mid-life snaps instantly back to its baseline (modulation), while an agent whose *growth* is frozen keeps everything it learned and only decays as the world moves on (internalization). Timescale changed; the structure carried.

So one column of the wall fell: **the bearer can be the thing that's changed.** Rung 2 is buildable. Rung 1 is everywhere. That's a real result — and then we tested rung 3, and it got strange.

## Being changed is not the same as having stakes

The foraging probe was an artificial animal whose outcomes could flow back into its own parameters. First result: yes, an animal altered by its own outcomes does better — **by its own assigned lights**. Scored on any bearer-neutral yardstick (food actually collected), the advantage vanished. That was a punchline we took seriously: we hadn't built stakes; we'd discovered how strongly the designer's assignment leaks into "success."

Then we tried making the criterion the animal's *own*, generated from its own condition. It got worse — literally. An animal that defines value from where it has been collapses into itself: it values its own footsteps, returns to them, and starves in a changing world, while the control animal with an externally-sourced purpose survives. Self-constitution without a world is not stakes. It's a closed loop eating itself.

Then came the design where stakes were finally *structural* rather than assigned: **markets**. In a market, the same world event is a gain or a loss depending on what *you* hold — bearer-indexing is built into the world, not pasted on by the designer. We ran matched pairs: an agent whose outcomes fed its own weights, and a yoked twin that received the identical outcome stream from outside.

The two things came cleanly apart, inside one acting system:

- **How the agent was built** (grown-in vs bolted-on) showed up in its internal geometry — and ignored whether it bore outcomes.
- **Whether it bore its own outcomes** showed up in its capital (10× difference, measured in dollars-of-the-world) — and ignored how it was built.

Being shaped and bearing consequences are *different factors*, and you can have one without the other in a single living system. That's the cleanest statement of the wall we've ever measured: we can build rung 3's *formal structure* — the outcome alters the same continuing actor, and the actor proceeds as the altered entity — and still nobody cares. The Bearer Is Not The Thing Changed broke in one column and holds in the next.

## We audited ourselves, and the audit bit

In November, we re-audited our own numbers and found the memory-metric result — "having the grown-in record *costs* the model" — had been computed against the wrong tokens (we were scoring padding). Redone correctly, the headline **flipped sign**: grown-in records are genuinely load-bearing, bolted-on records sit idle. Two other claims died in the same pass (the occupancy level as a *bearer-specific* signature; a claimed threefold ratio that survives only as an absolute gap). We retrained the missing checkpoints from scratch to replace ones we'd accidentally deleted, and we kept the withdrawn claims visible and struck-through in every internal map. A program whose central virtue is saying what it didn't show has to survive that virtue applied to itself. It did — the ladder, the gap, the ladder's universality, and the market separability all survived real-token scoring unchanged.

## The trajectory

Five machines, same wall. One wall-column broken by construction order — twice, across two layer types and two timescales. A ladder of internalization with a residue no amount of retraining closes. A formal structure of bearing-outcomes that is fully buildable and still empty of stakes. And one failed self-constitution loop that may be the most interesting negative result in the corpus.

The next question is no longer philosophical. It's an engineering spec:

> **Can outcomes be constituted *by* the actor — rather than assigned *to* it — without collapsing into the closed loop we measured?**

Everything before this was building the vocabulary to ask it in code.

Rung 4 stays empty on purpose. You can't measure "someone for whom it is had" with any instrument we have, and pretending otherwise is the exact failure mode this entire program studies.

---

## The receipts

For anyone who wants the numbers behind each claim — every figure above traces to a specific artifact:

- Dimmer result: ΔC +0.010, CI (+0.000, +0.025), n=200/condition — `horn-experiment`, PROBE_MATRIX.md
- Notebook tax/content pair: ΔC −0.165 (CI −0.225, −0.105) / +0.005 (CI incl. 0) — FTTF answerability battery
- Ladder ratios: grown-in/crutch ≈ 0.09 at matched constraint fraction, both scales; floor ≈ 2.6–3.2× — spectral C-series, 2 scales × 2 tokenizations × seed replicates
- Occupancy constant: 0.41–0.45× chance in five configurations (passive models) — same series; acted agents sit 0.82–0.91× — market capstone
- Record load-bearing (post-audit): ΔNLL +0.05 median, 15/15 seeds CI>0; withdrawn claim shown struck — `motion-lgi` audit branch, Nov 2026
- Wipe-vs-freeze: collapse to 0.254 vs persistence at 0.69 — continuant probe v0.2
- Criterion leak: own-lights +0.91–0.95 (100% seeds) vs bearer-neutral +0.002 (55%) — continuant probe
- Self-constitution lock-in: B−C = 0 in static world, −65..−137 days under drift — constitution v0.5
- Market separability: capital ×17.5–18.1 (self-borne) vs ×1.3–1.6 (yoked), construction effect constant — capstone 2×2, 8 seeds

Master index: `PROBE_MATRIX.md` + `AUDIT_HANDOFF_MAP.md` in the hub repo; the full substrate×rung grid is at `what_carries.html` in the step player workspace.
