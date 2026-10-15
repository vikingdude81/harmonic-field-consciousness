# L5 Memory Construction-Order — CPU Run Report

*Run completed 2026-10-15 05:21 · harness `run_l5_construction_order.py` · results `l5_results_cpu_m3/l5_construction_order_results.json` (hermes workspace `consciousness_research/l5_experiment/`)*

## Question

Does **construction order** of a memory channel determine whether stored records are load-bearing for the bearer — even when the final model state has access to the same records? This is the flagship experiment for the Memory Stratum claim: *storage is formation order, not a file cabinet.*

## Setup

- Substrate: spectral char-LM checkpoint (`language_experiments/20260207_103425`), vocab 97, 256d, 4 heads, 4 layers, block 256.
- Memory channel: low-rank (rank 64), 512 slots, write/read through attention (sediment channel).
- Seed 1234, CPU (16 threads), 913 steps/epoch, 1000 val samples.
- Four arms:
  - **M1_vanilla** — no channel.
  - **M2a_inference_retrofit** — channel bolted on at inference only (never trained).
  - **M2b_adapted_retrofit** — channel bolted on, then 10 epochs of adaptation training.
  - **M3_grown_in** — channel present from the start of training (20-epoch cap, early stop at 13, best epoch 5).
- Measurement: **deltaNLL = NLL(record erased) − NLL(record present)**, bootstrap CI95, per arm. Negative delta = the record *hurts* = tax.

## Results

| arm | deltaNLL (erased − full) | CI95 | channel stats | verdict |
|---|---|---|---|---|
| M1_vanilla | 0.00000 | — | no channel | independent |
| M2a_inference_retrofit | 0.00000 | — | gate 0.0, read_norm 0.0 | independent |
| M2b_adapted_retrofit | −2.0e−8 | [−6.6e−8, +2.4e−8] | gate 2.7e−4, read_norm 3.2e−5 | independent |
| **M3_grown_in** | **−0.26375** | **[−0.30004, −0.22921]** | gate −0.1497, written 350,592, read_norm 0.1997, attn_entropy 4.895 | **RECORD_IS_TAX** |

Channel attention entropy for reference: uniform would be 6.23. M3 sits at 4.90 — the channel reads selectively, not diffusely.

## Reading

1. **The retrofit floor is real and measured at L5.** M2b got 10 epochs of adaptation with full access to the channel and drove it essentially closed (gate 2.7e−4, read_norm 3.2e−5). A finished model *can* be given memory and will learn to ignore it. Erasing the record of M2b changes nothing to 8 decimal places. This is the NLL-domain version of the retrofit floor (cf. the 2.6× localization floor and G3's unreformed layer-0 root: depth ratio C1 1.34 → C2b 2.75 → C3 1.56).

2. **Grown-in memory is load-bearing and costly.** M3's channel is open (gate −0.15, 20% read-norm), and erasing its record *improves* NLL by 0.26 nats (ppl 7.87 → 6.04). The record is not an add-on the model consults — it is woven into the computation, and removing it changes the model's behavior by more than the retrofit arm's entire channel contribution. This reproduces the **C2a tax pattern** at the memory stratum.

3. **The channel permanently reshapes the bearer.** M3's erased NLL (1.80) is still far worse than M1 (0.52). Erasing the record does not restore the vanilla model — the grown-in channel changed how the whole network trains. Storage-as-formation-order, not storage-as-attachment.

## Caveats (honest ledger)

- **Single seed (1234), single substrate.** Needs ≥3 seeds before the 0.26 number is quotable.
- **M3 is a worse model overall** (best val NLL 2.06 vs M1's 0.52). The tax measurement is internally controlled (same model, full vs erased), but a `M3-erased-from-start` control arm is needed to separate "record taxes the bearer" from "channel presence taxes training." The deltaNLL design already holds the weights fixed and flips only the record — but the training-instability confound (val divergence after ep5 in M3, ep1 in M2b) should be addressed with a lower channel-capacity arm.
- M2a's zero is trivial (gate initialized and left at zero); M2b is the real retrofit control.

## Bridge to motion-lgi (friction program)

The 1536-pair power run (motion-lgi `probes/power_run/`, commit 5dc65ac) showed **instantaneous friction is bearer-blind**: the ξ-gap tracks the live-slot token, not the bearer's history (interaction −0.0003, CI [−0.0009, +0.0002]; events direction-symmetric and bearer-blind). L5 shows where the bearer signature actually lives: **construction order**, visible only in a history-bearing measurement (record cost), not in a state-matched one (per-token flow).

The two instruments are complementary and the next probe is concrete: port the ξ/friction probe (motion-lgi `src/`, dim_scale auto-Q) to the L5 checkpoints (4L/256d — cheap). **Prediction:** M2b's flow profile shows the retrofit floor (channel adds no reorganization of layer flow), M3's shows a reorganized profile (healed root). If M3 vs M2b differ in the friction domain despite matched per-token behavior, that is the first motion-lgi signal that is *not* bearer-blind — because construction order is exactly what the token-level probe could not see.

G7 connection: RECORD_IS_TAX is the NLL-domain analog of G7's liability trough — the frozen sediment is not degraded by time, it is mis-timed by a moved world. Erasing M3's record improves performance the same way the never-altered agent outperforms the frozen one at the wrong phase.
