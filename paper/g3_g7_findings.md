# G3 + G7 Findings — Cross-Substrate Occupancy & Recency Sweep

Run date: 2026-09-06. All runs pure-CPU, read-only on user repos. Harnesses:
`run_g7_recency.py` (delay grid, drift-parameterized, multiprocessing),
`run_g7_addendum.py` (tracker-state probe + dense interval grid),
`analyze_g7.py` (phase model, figure). Results: `g7_d{400,1200}_drift0.25.json`,
`g7_d400_drift{0.05,0.60}.json`, `g7_tracker_state.json`, `g7_dense_grid.json`,
figure `g7_phase_result.png` (241,005 B).

## G3 — the occupancy invariant, corrected

My earlier "C3 ≈ 0.41–0.45× chance" claim was substrate-specific (spectral only).
Normalized across every substrate that reports occupancy (occ/chance):

| substrate | C1 | C2b | C3 | C3/C1 |
|---|---|---|---|---|
| spectral-256d | 1.064 | 0.718/0.688/0.657/0.604 (10/20/40/80ep) | 0.574 | 0.539 |
| spectral-512d | 0.985 | 0.926 | 0.454 | 0.461 |
| spectral-bpe | 1.044 | 0.914 | 0.449 | 0.430 |
| agent-regulator r16 | 1.835 | 1.430 | 0.990 | 0.540 |
| agent-regulator r25 | 1.679 | 1.387 | 0.934 | 0.556 |
| agent-regulator r40 | 1.631 | 1.281 | 1.061 | 0.650 |
| agent-formation 400d | 1.615 | 1.430 | 0.918 | 0.568 |
| agent-formation 1200d | 1.432 | 1.374 | 0.784 | 0.548 |
| agent-formation 3000d | 1.356 | 1.326 | 0.701 | 0.517 |

(foraging-separability and market capstones report self/yoked variants; excluded from ratio stats as C2b missing.)

- **New invariant: C3/C1 = 0.534 ± 0.059 (n=9, range 0.431–0.650)** — grown-in
  construction cuts per-token basin occupancy to ~53% of vanilla, across spectral
  (char/512d/BPE) and agent-world substrates. C2b/C1 = 0.844 ± 0.119 (not invariant).
- corr(C1/chance, C3/C1) = +0.696: substrates with fat vanilla baselines (agent worlds,
  1.36–1.84× chance) show slightly *less* C3 suppression; spectral C1 ≈ chance.
- Depth contrast (256d, layer0/layer3 occupancy ratio): C1 1.34 → C2b40 2.75 → C3 1.56.
  Retrofit leaves an unreformed layer-0 root; grown-in partially heals it. Consistent
  with the measured 2.6× retrofit floor.

## G7 — recency of the alteration (continuant ring world)

Timing (user's question): one 400-day sim = 0.017–0.05 s; full grid
4 configs × 2,432 tasks (64 seeds × 8 delays × 2 tastes) ≈ **20 s wall** on 24 procs.

Headline: **the sediment does not decay — the world moves under it.**

1. **Bearer state is bit-frozen.** Direct probe of `ForagerPolicy.logits` after
   K3 freeze: max |Δlogits| (L∞) = **0.00e+00 at every probe day** (D=5…500).
   The alteration is permanent at machine precision.
2. **Measured "advantage" oscillates with world phase.** Ring world,
   `center=(center+drift)%50` → period T = n_sites/drift. Drift 0.25 (T=200 d),
   freeze f=200, advantage over never-altered agent:
   D=0 +0.439 → floor **−0.229** (D≈60–105, frozen map actively steers to the wrong
   side of the ring) → recovery at multiples of T: D=200 +0.4368 (99.4%),
   D=400 +0.4382 (99.7%), D=600 +0.4404 (100.2%) [1200-d run].
3. **Drift sweep confirms the geometric mechanism.** drift 0.05 (T=1000): floor +0.049,
   never negative within 400 d; drift 0.25: floor −0.229; drift 0.60 (T=83): A≤0 by D=5.
4. **Lag-corrected phase model, lag from the independent tracker probe (never fitted
   to the advantage curve):** offset(D)=ring(drift·D+lag).
   - drift 0.25: lag 2.92 → predicted realignment D*=(50−2.92)/0.25=188.3 d;
     observed min tracker offset at D=190 → **err 1.7 d (0.9%)**.
   - drift 0.60: lag 6.56 *extrapolated* from the two probe lags → D*=72.4 d;
     observed first dense-grid peak D=70 → **err 2.4 d (3.3%)**; peak spacing 85 d vs T=83.3.
   - Full-curve fits (only amplitude + baseline free): R²=0.979 (drift 0.25),
     R²=0.931 (drift 0.60).
5. **K2 record-wipe identity control:** across 518 cells max |advantage| = **0.0e+00** —
   wiping the record is trajectory-identical to never having it. Exact identity, not a
   statistical null. Cleanest existing demonstration of "stored history ≠ historically
   altered bearer."
6. **Delay-invariance:** pre-ablation advantage ≈ +0.58–0.59 at every freeze point
   f=25…399; the post-ablation curve depends only on D since ablation, not on how much
   sediment was laid down. Recency (age of alteration) matters; dose saturates.

## Instrument critique (feeds back into continuant-probe v01 Part 3)

v01 samples only two windows (immediately_after = 20 d; late_life = last 50 d) at one
delay (days//2). At days=400/f=200 those windows are D≈0–20 and D≈350–400 — both near
phase peaks; the liability trough (D=60–105, −0.23) is never sampled. The "decay"
reading in a periodic world is phase aliasing. **Falsifiable upgrade:** run a
non-wrapping world (or aperiodic drift) to measure true persistence, and a dense
interval grid to separate decay from aliasing.

## Reading for Bearer Theory

The mantra *"the alteration persists"* is now **strengthened to bit-identity** — but the **yield** of the alteration is world-phase-dependent and can invert into an active liability. A frozen incorporated constitution is not degraded by time — it is mis-timed by a moving world. State-matched snapshots ("ablated agent performs at chance") conceal exactly this: same bearer state, opposite consequences.

---

# G7-NW — the non-wrapping-world test (follow-up, same day)

Harness `run_g7_nowrap.py`, analysis `analyze_g7_nowrap.py`, results
`g7_nowrap.json`, figure `g7_nowrap.png` (215,180 B). Same frozen bearer
(freeze f=200, 64 seeds, 1200 d, drift 0.25), five world geometries. 960 tasks,
3.7 s wall.

| geometry | dynamics | A@0 | floor (min@D) | A@600 | recovery | kernel fit R² |
|---|---|---|---|---|---|---|
| ring (control) | periodic, T=200 | +0.439 | −0.234 | +0.440 | 99.5–100% at 200/400/600 | 0.828 |
| reflect | triangle wave, T=392 | +0.466 | −0.117 | −0.101 | +0.498 at D=390 (pred 392) | 0.758 |
| rw | reflected diffusion (signed steps) | +0.617 | +0.317 | +0.318 | none, never negative | **0.948** |
| jump | ring + 5%/day teleport | +0.287 | −0.044 | −0.012 | none | 0.495 |
| line | unbounded drift | +0.021 | 0.000 | 0.000 | degenerate (world empties) | 0.995 |

**P5: logits bit-frozen (L∞ = 0.0e+00) in ALL five geometries.**

Kernel model y(D) = A·k(D) + baseline, k(D) = ensemble-mean
exp(−0.5·(dist_s(D)/W)²) built from ACTUAL world trajectories + tracker peak
at freeze (no fitted shape): fits rw at R²=0.948, line 0.995, ring 0.828,
reflect 0.758.

**The law this isolates — yield is a revisit statistic of the world geometry:**
1. Periodic worlds: yield returns EXACTLY at multiples of T (ring T=200,
   reflect T=392 — the recovery peak moved with the geometry, predicted 392,
   observed 390) and inverts to liability at half-period.
2. Aperiodic bounded worlds (diffusion, teleport shocks): yield decays with NO
   sharp peaks and NO return — but never to zero. It converges to the
   **ergodic floor** k∞ = (2/n)·∫exp(−d²/2W²) = **0.251 analytic**; measured
   jump plateau 0.13–0.25. A bounded world always brings the food back near
   the remembered site: the frozen constitution keeps a permanent ~25%-of-peak
   dividend forever.
3. Unbounded world: degenerate — the bump escapes, everyone starves, no
   differential signal (advantage ≡ 0). Decay-to-zero of yield is unobservable
   because the counterfactual baseline starves too.
4. The naive "shock decay" (1−p)^D is wrong: measured k(D)=0.49 at D=100 vs
   0.006 analytic-no-return, because after a teleport the bump *drifts back*.

**Answer to the persistence question:** there is no true decay of the
alteration (bit-frozen everywhere) and no true decay of its yield to zero in
any non-degenerate world. "Decay" interpolates between two geometric regimes:
sharp oscillation (periodic worlds) and relaxation to an ergodic floor
(aperiodic bounded worlds). The right question is never "does the sediment
decay?" but "does the world revisit?"
