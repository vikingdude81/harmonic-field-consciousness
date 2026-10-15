# L5 Flow Report — the friction instrument sees construction order

*Companion to `l5_memory_construction_order_report.md`; closes the 2×2 instrument table of §5 of the Paper #3 outline. motion-lgi commit `6d6a7e2` (`probes/l5_friction_probe.py`, `probes/results/l5_friction_results.json`, `l5_friction_raw.npz`), run 2026-10-15, 60 s CPU.*

## Question

The 1536-pair power run showed the ξ-friction instrument is **bearer-blind to token-level contrasts** (self-vs-wrong: interaction −0.0003, events direction-symmetric, Fisher p=1). That null localized the bearer to *formation history*, but left a prediction untested: if the instrument is state-matched, it should still see **construction-order contrasts** — same architecture, same data, different formation history. The L5 arms are exactly that contrast:

- **M1** vanilla (phase-1 baseline, no channel ever)
- **M2b** retrofit (baseline weights + channel, fine-tuned; gate ≈ 2.7e-4, inert)
- **M3** grown-in (fresh init + channel from step 0; gate −0.15, alive; RECORD_IS_TAX ΔNLL −0.264)

## Instrument

Identical math to the power-run GPU port (`probes/power_run/xi_sim.py`), at native dim 256: velocity field v(h) = Block(h) − h at captured validation-stream states, frozen context K/V (RoPE at true positions, CTX=32, 4 moving positions), Nose–Hoover thermostat (σ=0, Q = 0.001·C·kT, ξ ∈ [−2,5] boundary-zeroed), Tsit5 tableau dt=0.01 × 100 steps, float32 strict. 128 offsets × 4 positions × 4 layers × 3 arms = 6,144 ξ values, paired by (offset, position). Heavy-tail discipline: trimmed means (10%), Wilcoxon signed-rank on paired deltas, two-sided event counts (|Δξ| > 0.3) with sign split. No one-tailed definitions.

## Result

**Mean ξ per layer** (negative = anti-damped / energy pumped into flow; positive = braking):

| arm | L0 | L1 | L2 | L3 | mean \|v\| L0→L3 |
|---|---|---|---|---|---|
| M1 | −0.98 | −1.00 | −0.67 | +0.26 | 1.68 → 8.30 |
| M2b | −0.88 | −0.94 | −0.64 | +0.22 | 2.33 → 11.95 |
| M3 | **−1.48** | −0.72 | −0.55 | **+0.96** | 3.25 → 12.91 |

**Paired ξ contrasts** (trimmed-mean Δ, Wilcoxon p, events +/−):

| contrast | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| M3 − M2b | **−0.615**, p=5e-65, 30/375 | +0.205, p=6e-11, 219/90 | +0.054, p=0.19 | **+0.741**, p=4e-16, 310/144 |
| M3 − M1 | −0.498, p=2e-52 | +0.250, p=1e-16 | +0.074, p=0.015 | +0.713, p=3e-18 |
| M2b − M1 | +0.104, p=9e-35 | +0.040, p=2e-5 | +0.023, p=0.014 | −0.032, p=0.36 |

Three findings:

1. **The flow instrument is not bearer-blind to history.** M3 vs M2b — identical architecture, identical data, identical present behavior at the token level — produces large, directionally consistent, layer-structured ξ differences. The power-run null was about *what* the probe contrasts (state-matched tokens), not about the instrument being dead.
2. **The retrofit floor appears in the flow domain, independently of the NLL metric.** M2b ≈ M1 everywhere (Δ ≤ 0.10, and null at L3). Fine-tuning a closed channel barely rewrote the flow profile. M3's profile is a different shape: deeper root anti-damping (L0 −1.48) and much stronger terminal braking (L3 +0.96). This is the G3 "unreformed root vs healed root" pattern, now measured by a second, independent instrument.
3. **Direction is structured, not noise.** M3 − M2b events at L0 are 375/405 negative (M3 pumps harder at the root); at L3, 310/454 positive (M3 brakes/reshapes harder near the output). A grown-in memory channel is associated with more injection at the bottom of the stack and more shaping at the top.

## Interpretation

The 2×2 completes:

| | token identity (state-matched) | construction order (history-bearing) |
|---|---|---|
| flow instrument (ξ, this run) | **null** (power run, 1536 pairs) | **seen** (p=5e-65 / 4e-16) |
| erasure metric (ΔNLL) | n/a | seen (−0.264 CI [−0.300,−0.229]) |

The bearer passes through the flow and can be detected there — but only by a probe whose contrast pair carries history. Two independent bearer instruments (record-erasure NLL, thermostat ξ-profile) now agree on the same arm ordering: M3 ≠ {M2b ≈ M1}. That agreement is the strongest evidence yet that construction order is a physical property of the weights, not an artifact of one metric.

## Caveats

- Single seed per arm (same limitation as the NLL result); seed replication is the obvious next step.
- ξ saturates at the [−2,5] boundaries in many pairs (heavy tails) — hence trimmed/rank statistics; the effect survives trimming decisively.
- M2b − M1 is statistically significant at L0–L2 but ~10× smaller than M3 − M2b; retrofit moved the profile slightly at the root, not at the top.
- The channel itself sits after the last block, so it never enters the block velocity fields directly; every difference reported here is an *indirect* imprint of formation history on the block weights.

## Reproduce

```
cd motion-lgi/probes && python l5_friction_probe.py
```
Checkpoints: hub `NanoGPT/spectral_transformer/language_experiments/20260207_103425/phase1_baseline/best_model.pt` (M1); workspace `consciousness_research/l5_experiment/l5_results_cpu_m3/{M2b_adapted_retrofit,M3_grown_in}/best_model.pt`.
