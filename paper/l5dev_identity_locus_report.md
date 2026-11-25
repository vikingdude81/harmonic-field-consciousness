# L5-DEV: The identity locus of a memory channel is set by construction order

**Status:** pre-registered confirmatory run, n=12 seeds. Protocol `motion-lgi/probes/devcohort/PRE_REG.md` (2026-11-24, incl. amendment v2 written before any confirmatory seed). Artifacts: `probes/devcohort/devcohort_results/seed_*/` (checkpoints + `battery.json`), `MORNING_REPORT.md`, commit `f6232c7`.

## Question
When a memory channel is *grown into* a model during development (M3) versus *bolted on* by fine-tuning an adapted retrofit (M2b), where does the channel's causal contribution live — in the weights, in the record, or in the read-machinery — and does construction order determine which?

## Design (compressed; see PRE_REG for full)
Shared phase-1 baseline (C-series Nov retrain checkpoint, config-identical to the deleted Feb original — substitution disclosed pre-run). Per seed: M3_grown_in and M2b_adapted_retrofit arms (plus M1 audit). Battery per checkpoint: **present** / **erased** (record table zeroed) / **sham** (rewrite identical content) / **random** (fresh random table) / **xplant** (transplant the other arm's trained table). Paired per-sample val NLL; ReZero gate read directly from `model.memory`.

## Results (12/12, 12/12, 12/12, 14/24)
- **H1′ PASS (12/12):** |Δ_tax(M3)| > |Δ_tax(M2b)| every seed; mean |diff| 0.264, bootstrap 95% CI [0.204, 0.334].
  - **M2b retrofit: Δ_tax = 0.0000 exactly on all 12 seeds; gate |g| ≤ 0.001.** The bolted-on record is causally inert — the retrofit fine-tune never opened the gate. The channel is dead machinery decorating the weights.
  - **M3 grown-in: Δ_tax negative on all 12 seeds** (−0.081 … −0.502). Erasure *improves* val NLL. The grown-in bearer metabolized its sediment into a suppressive, thermostat-like channel and develops around it. Gates −0.27 … +0.18 (sign heterogeneous; magnitude load-bearing).
- **H3 PASS:** sham exact (|Δ| < 1e-5) on all seeds/arms — instrument integrity.
- **H4 PASS:** random tables rescue nothing.
- **H2′ FAIL (14/24):** xplant effect is *not* governed by host-gate liveness alone. Foreign trained tables produce |resc_x| ≤ 0.136 (median ~0.045) against host taxes of ~0.25 — one-fifth to one-sixth. **The load-bearing sediment is bearer-idiosyncratic**: an open read-machinery does not make another bearer's history do work.

## Reading (operational, falsifiable, no phenomenal claims)
1. **Construction order decides bearer membership.** Identical final architecture, identical corpus, different developmental order → one bearer carries its record as state (surgically visible), the other carries nothing (surgically invisible). "Bolted-on" is not a weaker memory; it is a memory the bearer never became.
2. **The grown-in channel behaves like friction, not storage.** Negative, erasure-improves-signature across 12/12 mirrors the ξ thermostat-friction result (motion-lgi): a tax the system regulates *through*. Benchmark-state observables (val NLL) hide it; surgery exposes it.
3. **Record value is indexical to the bearer** (exploratory, from H2′ failure). The same architecture + same corpus + another individual's sediment ≈ inert. Bearer-state is not content-type; it is *this bearer's* history.
4. **Wall consistency:** the M2b result is another instance of "the bearer is not the thing changed" — adding a channel to a finished bearer changes the artifact, not the bearer.

## Limits
Small substrates (≈1.6 M params, char-level TinyStories, vocab 97). Gate-sign heterogeneity (two positive-gate seeds) unexplained. "Improvement on erasure" is a validation-NLL statement, not a capability claim. Inkling-class transfer (squid probe-bus + KV surgery) is the natural next substrate; nothing here generalifies to it yet.

## Next discriminators (draft, not yet pre-registered)
- Dose–response: fraction of record erased vs NLL shift (is the tax continuous or thresholded?).
- Gate trajectory during M3 training: does the gate act as a thermostat (loss-coupled sign flips)? — this is the direct motion-lgi ξ bridge on the same checkpoints, zero retraining.
- Self-xplant control (own table rewritten through the transplant path) to isolate procedure effects from bearer-identity effects.
- Inkling-class: replicate the #3 / #19 repairability latencies as a distribution (15 seeds × 10 prompts gate already in `experiments/squid-bearer-instrument/REPORT.md`).
