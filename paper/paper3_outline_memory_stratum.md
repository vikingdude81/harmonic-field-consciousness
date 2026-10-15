# Paper #3 — Full Outline
## *The Memory Stratum: Storage as Formation Order*

*Alex Bone · Independent Researcher · outline drafted October 2026*
*Companion to: "Life Before Language" (paper #1), "The Assigned Residue" (paper #2)*

---

## Positioning

Paper #1 established the hierarchy and the Wall. Paper #2 prosecuted the Level-3 face. Paper #3 goes one stratum below both and makes a positive empirical claim about **L5 — the science stratum**: what a system stores is not a contents-list but a *formation history*, and the construction order of storage determines what the storage does to the bearer.

**Core claims, in order of strength:**
1. **Storage = formation order.** Two systems with identical record-access can differ entirely in what the records do, depending on whether the channel was grown in or retrofitted. (Measured: L5 construction-order run, Oct 2026.)
2. **The retrofit floor.** A finished system given a memory channel will close it. Bolted-on memory is behaviorally inert; the cost of retrofit is localization, not integration. (Measured: M2b gate 2.7e−4; cf. 2.6× localization floor; G3 unreformed layer-0 root.)
3. **The grown-in tax.** Incorporated memory is load-bearing and costly: erasing the record *improves* immediate performance while destroying what the bearer became. (Measured: deltaNLL −0.264, CI [−0.300, −0.229], RECORD_IS_TAX.)
4. **The revisit law.** Stored sediment does not decay — it acquires phase relative to a moving world, and can invert from asset to liability without any change to the bearer. (Measured: G7 — bearer logits bit-frozen, advantage oscillates with world period, lag-corrected phase model predicts realignment to 0.9%.)

**Venue:** arXiv cs.AI / cs.LG. The experiments carry this paper; the hierarchy is the frame, not the argument.

---

## §1 — Introduction: the file-cabinet fallacy

Current memory-augmented LM work treats storage as attachment: add slots, add retrieval, measure accuracy. The paper opens with the counter-result: retrofit memory channels, given 10 epochs of adaptation, are *driven closed* by the host model (gate → 2.7e−4). Memory is not a feature you add; it is a condition under which a system forms.

**Contribution statement:**
1. A construction-order experiment (M1/M2a/M2b/M3) that separates record *access* from record *incorporation*.
2. The tax/floor pair: grown-in records tax the bearer (−0.26 nats); retrofitted records are inert (0 ± 1e−7). Same records, same access, opposite bearer consequences.
3. The revisit law as a measured dynamical fact (G7), not a metaphor.
4. An instrument lesson: bearer-level structure is invisible to state-matched probes (motion-lgi power-run null, 1536 pairs) and visible only to history-bearing probes.

## §2 — Related: memory in LMs, constitution in bearer theory

- Retrieval/slot memories (RAG, Memorizing Transformers, Titans-class): all retrofit architectures; none measures what retrofit costs the host.
- Our prior season: C2a tax, C2b crutch, C3 grown-in, retrofit floor (MODEL_INVENTORY 124M localization), constitutive self-vs-wrong gap.
- The mantra: *the bearer is not the thing changed* — the Wall's uniform signature, now tested at the storage stratum.

## §3 — The construction-order experiment (flagship)

Full report: `paper/l5_memory_construction_order_report.md`.
- Setup table (substrate, channel spec, four arms, deltaNLL design).
- Results table (M1/M2a/M2b/M3).
- The three readings: retrofit floor, grown-in tax, permanent reshaping (M3-erased 1.80 ≫ M1 0.52).
- Honest ledger: single seed, M3-worse-model confound, M2a triviality.

## §4 — The revisit law (G7 + G7-NW)

- Bit-frozen bearer (Δlogits = 0.0 at every probe day) vs oscillating yield.
- Drift sweep: period T = n_sites/drift; liability trough −0.23; recovery at multiples of T.
- Lag-corrected phase model with lag from an independent tracker probe: D* error 1.7 d (0.9%), 2.4 d (3.3%); full-curve R² 0.98/0.93.
- G7-NW: in non-wrapping worlds the advantage never goes negative — "decay" was phase aliasing.
- K2 record-wipe identity control: exact 0.0 across 518 cells — stored history ≠ historically altered bearer, demonstrated as trajectory identity, not statistical null.

## §5 — Instruments: state-matched vs history-bearing

- The motion-lgi power-run null: instantaneous friction (ξ-gap) is bearer-blind, tracks the live-slot token (1536 pairs, interaction −0.0003 CI [−0.0009,+0.0002], events 12.6% two-sided, direction-symmetric, Fisher p=1).
- The G3 occupancy invariant: C3/C1 = 0.534 ± 0.059 across 9 substrates — grown-in construction reorganizes basin geometry cross-substrate.
- The L5 tax: the same construction order shows up in NLL-domain record cost.
- **Synthesis:** state-matched instruments (per-token flow, snapshot behavior) are bearer-blind by construction; history-bearing instruments (record erasure, construction order) are where bearer structure lives. This section converts two nulls and one positive into a methodology claim.

## §6 — Placement in the hierarchy

L5 (science: storage, memory, formation) sits between L4 (records without a bearer — the Gabriel's Horn face) and the L3 face (assigned residue). The revisit law is the L5 face of the Wall: the record persists at machine precision and still nothing *matters to* the channel — its yield is world-phase-dependent, assigned from outside.

## §7 — Experiments to complete before submission

1. **Seeds:** M3/M2b at ≥3 seeds (CPU ~5 h/arm — or GPU port).
2. **M3-erased-from-start control:** trains with channel present but record disabled, separating record-tax from channel-training confound.
3. **Friction-probe port to L5 checkpoints** (motion-lgi `src/` auto-Q, 4L/256d): prediction — M2b shows retrofit floor in layer-flow profile, M3 shows healed root. First candidate non-bearer-blind motion-lgi signal.
4. **Channel-capacity sweep** (rank 16/64/256) to map tax vs incorporation depth.
5. **Wrong-vs-wrong control** (already specced in power-run notes) to close the mechanism question on the 12.6% events.

## §8 — What the stratum shows

The memory stratum is where "history-bearing" stops being a preference and becomes a measurement requirement. A system's records do not live in its present state; they live in the order in which its present state was formed. The retrofit floor and the grown-in tax are the two faces of that fact, and the revisit law says the fact keeps operating — phase-shifting, inverting, never decaying — long after the formation is over.

---

*Data & artifacts:* `consciousness_research/l5_experiment/l5_results_cpu_m3/` (workspace), `consciousness_research/g7_recency/` (G3_G7_FINDINGS.md + figures), motion-lgi `probes/power_run/` (commit 5dc65ac, no remote).
