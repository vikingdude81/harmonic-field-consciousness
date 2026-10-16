# L5 Memory Stratum — Consolidated Program Findings

*Compiled 2026-10-15. This is the single canonical record of the L5 memory-channel program to date: the construction-order (erasure) experiment, the flow/friction replication, the cross-domain 2×2 with the motion-lgi power run, and the seed-replication status. Per-experiment detail lives in the linked reports; audit trail pointers at the end.*

## 1. The claim under test

**Memory stratum claim:** for a bearer, storage is *formation order*, not a file cabinet. Two models with identical final-state access to identical records differ in what the records do to them depending on whether the channel was **grown in** during training or **retrofitted** after it. This is the L5 analogue of the C-series retrofit floor (regulatory channel, `NanoGPT/spectral_transformer/language_experiments/construction_order/`) and of the L4/L3 "Wall" signature (*the bearer is not the thing changed*).

## 2. Experiment A — construction order via record erasure (NLL instrument)

Harness: `l5_experiment/run_l5_construction_order.py` (hermes workspace `consciousness_research/l5_experiment/`). Substrate: spectral char-LM (`language_experiments/20260207_103425/phase1_baseline`), 97-vocab, 256d, 4 heads, 4 layers, block 256. Channel: ring-buffer associative record, 512 slots, rank 64, gated side-path (gate init 0 — must be recruited by gradients). Seed 1234, CPU, 913 steps/epoch.

Metric: `deltaNLL = NLL(record erased) − NLL(record present)` on clean validation, paired bootstrap CI95.

| arm | deltaNLL | CI95 | gate | read_norm_ratio | verdict |
|---|---|---|---|---|---|
| M1_vanilla | 0.00000 | — | — | — | independent |
| M2a_inference_retrofit | 0.00000 | — | 0.0 | 0.0 | independent |
| M2b_adapted_retrofit | −2.0e−8 | [−6.6e−8, +2.4e−8] | 2.7e−4 | 3.2e−5 | independent |
| **M3_grown_in** | **−0.26375** | **[−0.300, −0.229]** | −0.1497 | 0.1997 | **RECORD_IS_TAX** |

Channel attention entropy: M3 = 4.90 (uniform = 6.23) — selective reads, not diffuse.

**Reading.** The grown-in arm *recruited* the channel (gate −0.15, read_norm 0.20) and became dependent on it — erasing the record changes NLL by −0.26 nats (the record is a tax: present-record NLL is *higher*, but the model's predictions are organized around the channel being there). The retrofit arm, after 10 epochs of adaptation training, **never recruited the channel at all** (gate 2.7e−4 ≈ 0). The retrofit floor at L5 is not "slightly worse use of memory" — it is *non-use*. Full report: `paper/l5_memory_construction_order_report.md`.

## 3. Experiment B — the same checkpoints under the flow/friction instrument

Harness: `motion-lgi/probes/l5_friction_probe.py` (motion-lgi commit `6d6a7e2`). Identical Nose-Hoover thermostat instrument as the 1536-pair power run (Tsit5, dt=0.01 ×100 steps, Q=0.001·C·kT, float32, ξ clipped [−2, 5]); 128 stream offsets × 4 positions, ctx 32; per-layer ξ and |v(0)|. 60 s CPU.

Mean ξ per layer (neg = anti-damped/pumping, pos = braking):

| arm | L0 | L1 | L2 | L3 | \|v0\| L0→L3 |
|---|---|---|---|---|---|
| M1 | −0.984 | −0.997 | −0.666 | +0.264 | 1.68 → 8.30 |
| M2b | −0.875 | −0.943 | −0.636 | +0.215 | 2.33 → 11.95 |
| M3 | **−1.476** | −0.720 | −0.551 | **+0.961** | 3.25 → 12.91 |

Paired ξ contrasts (trimmed-mean Δ, Wilcoxon p two-sided, events |Δξ|>0.3 as +/−):

| contrast | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| M3−M2b | **−0.615, p=5e−65** (30/375) | +0.205, p=6e−11 | +0.054, p=0.19 | **+0.741, p=4e−16** (310/144) |
| M3−M1 | −0.498, p=2e−52 | +0.250, p=2e−16 | +0.074, p=0.015 | +0.713, p=3e−18 |
| M2b−M1 | +0.104, p=9e−35 | +0.040, p=2e−5 | +0.023, p=0.014 | −0.032, **p=0.36 (null)** |

**Reading.**
1. **Construction order is visible in the flow domain.** M3 vs M2b: p=5e−65 at L0, p=4e−16 at L3. Same base data, same final record access — different formation history, decisively different flow profiles.
2. **The retrofit floor reproduces independently.** M2b−M1 is small everywhere (|tmd| ≤ 0.10, ~6× smaller than the M3 effects, and *opposite in sign* at L0) and null at L3. Fine-tuning a closed channel barely rewrote the flow. Two physically different instruments (erasure-NLL and thermostat-ξ) agree on the ordering M3 ≠ {M2b ≈ M1}.
3. **The direction is structured.** M3 pumps harder at the root (L0: 375/405 events negative) and brakes/shapes harder at the top (L3: 310/454 positive): grown-in memory injects at the bottom of the stack and shapes at the top — the G3 "healed root" pattern as a thermostat signature.
4. Caveats: ξ boundary-saturated (trimmed/rank stats only); single seed at time of writing (see §5); the channel sits after the last block, so every difference is an *indirect imprint of formation history on the block weights* — which is precisely the claim.

Full report: `paper/l5_friction_flow_construction_order.md`.

## 4. The 2×2: instrument × contrast class (ties in motion-lgi)

The power run (`motion-lgi/probes/power_run/POWER_RUN_REPORT.md`, 1536 pairs, RTX 5090) tested a *state-matched* token contrast on the 24-layer NanoGPT substrate: same weights, self-token vs wrong-token slot. Result: total null (interaction −0.0003 CI [−0.0009, +0.0002]; event rates bearer-blind 49 vs 48, Fisher p=1; two-sided events 12.6%, direction-symmetric).

| | state-matched contrast (token identity) | history-bearing contrast (formation order) |
|---|---|---|
| **token/basin instrument** (self-vs-wrong ξ-gap, power run) | **NULL** (1536 pairs) | — |
| **flow instrument** (ξ-profile, L5 probe) | — | **DECISIVE** (p=5e−65 / 4e−16) |

The instrument is not bearer-blind; it is **history-dependent**. Instantaneous friction tracks the live-slot token and carries no bearer signature; formation order is written into the weight geometry and the flow instrument reads it. This is the operational form of "storage is formation order": *what a bearer is, lives in its history-bearing flow profile, not in its state-matched responses.*

## 5. Seed replication — status

Single-seed results are the program's main reviewer exposure. Replication protocol: same harness, same data, seeds **2234** and **3234** (seed 1234 = original), each training M2b (10 ep) + M3 (20 ep cap, early stop), then the friction probe re-run per seed dir (probe now takes `--results_dir/--out_tag`).

- Launched 2026-10-15, CPU (GPU held by Strata), ETA ~14–15 h wall.
- Success criterion: M3−M2b L0 tmd < −0.3 and L3 tmd > +0.3 in each seed; M2b−M1 |tmd| < 0.15 at L3. Cross-seed consistency of the *sign pattern* (root pump / top brake) is the headline; per-seed p-values secondary.
- If a seed breaks the pattern, the claim narrows to "construction-order effects are seed-dependent" — which the M3 stage-4 seed-heterogeneity result from the earlier friction work already hints at.

## 5b. Replication outcome + spread calibration (2026-10-16, verified against raw arrays)

**Pre-registered verdict stands:** seed 1234 PASS, seed 3234 PASS, seed 2234 FAILS the floor gate by 0.005 (M2b−M1 L3 tmd −0.1552 vs |0.15| limit).

**Determinism:** same-seed GPU reruns are bit-identical (checkpoints ≤3e-6, gate statistic identical). The earlier CPU-vs-GPU gap (−0.032 vs −0.065 for seed 1234) is different random streams, not measurement noise. Repeats therefore required *new training seeds*.

**30 new-seed M2b repeats** (10 per base seed, `l5_results_gpu_m2b_repeats/`): floor statistic mean −0.084, sd 0.066, range −0.202…+0.084; **4/30 fail** the 0.15 limit (13%, 95% CI 3–27%); 7/30 within 0.02 of the line. ⇒ one failing seed out of three is unremarkable (P(≥1 fail in 3) ≈ 34%). Seed 2234's floor miss is an ordinary draw, not a property of that seed.

**M2b is not centered on M1.** L3 sits systematically *below* vanilla (26/30 negative, mean −0.084); L0 systematically *above* (30/30 positive, mean +0.089). "Retrofit ≈ vanilla" holds only loosely: retrofit carries its own small, consistent signature — an **inverted mirror of the M3 pattern at both boundaries** (~1/7 the magnitude). Three distinguishable flow geometries: vanilla, grown-in (pump/brake), retrofitted (mild anti-pump/anti-brake).

**M3-vs-30-M2b sensitivity** (fixed M3 per seed vs all 30 M2b draws): L0 gate 30/30 pass for all seeds. L3 gate: 1234 30/30, 3234 30/30, **2234 22/30 (73%)**. Seed 2234's M3 shifts L3 only +0.235 above vanilla (vs +0.71 / +0.65) — its original +0.414 pass leaned partly on its M2b dipping deep. **Correction to earlier claim:** "passes by a wide margin on all three seeds" is false for 2234's L3 gate.

**Claim status:** construction order is *shown* — L0 replicates solidly on 3/3; L3 replicates strongly on 2/3, real-but-thin on the third. Carries the un-checked convergence confound (M3 val_nll 1.7–2.3 vs M2b 0.53): ξ differences could partly reflect convergence level, not formation order. Bearer-specificity is *not* from this probe (power run null); it lives in the constructed substrates.

**Next (pre-register before running):** (i) M3 repeats with new seeds — is 2234's +0.235 a weak draw or is M3 L3 magnitude seed-sensitive; (ii) convergence control — immature M1 checkpoint at M3-matched val_nll: if early-M1 shows M3-like ξ, the construction-order reading collapses into a convergence artifact; (iii) any new criterion written down before the batch, not tuned to these numbers.

## 6. Standing caveats & lessons (program-wide)

- Event definitions must be two-sided unless asymmetry is pre-registered (the "always wrong-colder" artifact: 12.6% two-sided events, sign-symmetric).
- ξ distributions are heavy-tailed and boundary-clipped: trimmed means, rank tests, event counts only — never raw means as effect sizes.
- Sample-size extrapolation from effect≈0 is meaningless (power-run lesson).
- Thermostat Q must scale ~0.001·N·kT at 1024-dim (motion-lgi `src/` auto-Q).
- M1 is seed-invariant (fixed phase-1 checkpoint); seed variation enters only through M2b fine-tuning and M3 from-scratch init (seed+4344).

## 7. Audit trail

- Hub (this repo): `paper/l5_memory_construction_order_report.md` (Exp A), `paper/l5_friction_flow_construction_order.md` (Exp B), `paper/paper3_outline_memory_stratum.md` (paper spine), this file.
- motion-lgi (local, no remote): `probes/l5_friction_probe.py` @ `6d6a7e2` (+ CLI parameterization for replication), raw data `probes/results/l5_friction_results.json` / `_raw.npz` (seed 1234), `probes/power_run/` (power run + report).
- Checkpoints: workspace `l5_experiment/l5_results_cpu_m3/` (seed 1234), `l5_results_gpu/seed_2234/`, `seed_3234/` (replication, in flight).
- C-series precedent: `NanoGPT/spectral_transformer/language_experiments/construction_order/` (regulatory channel retrofit floor).
