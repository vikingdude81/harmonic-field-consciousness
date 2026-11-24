# The Squid Probe-Bus Instrument: First Bearer-State Observations

**Date:** 2026-11-24 · **Status:** exploratory (n=1 per arm) · **Instrument:** `tensorfish/squid5090` fork, branch `probe-bus` (WSL2, RTX 5070 Ti, model `thinkingmachines/Inkling-2-Mini`, NVFP4 weights + Q4 KV)

## 1. What the instrument is

A self-owned inference engine (Zig + CUDA, zero Python) extended with three capabilities:

| Capability | Commits | What it does |
|---|---|---|
| **Probe bus v1/v2** | `eba7a5e`, `f3d089e` | JSONL record per decode step: `{step, pos, n_acc, w, spec, toks[], conf[]}` — token streams + per-slot drafter confidence (0.21–0.999 observed). Overhead ≤1% (268.3 vs 268.9 tok/s). |
| **KV surgery v3** | `2b9b35c` | CLI flags `--wipe-global L`, `--wipe-rings`, `--wipe-drafter`: independently erase each bearer-state layer after prefill. Q4 KV rows require zero-nibble fill `0x88` (a naive `0x00` memset leaves phantom `(0−8)·scale` values — contaminated control). |
| **Logits observation** | upstream | `eval_logits_streaming` — exact post-softcap FP32 logits (KLD/PPL), already in CLI. |

Determinism gate: temp-0 + fixed seed ⇒ bitwise-identical token streams across repeats (verified), so any divergence is attributable to the manipulation alone.

## 2. Bearer-state anatomy (Inkling-2-Mini, looped hybrid transformer)

The engine's checkpoint machinery enumerates the physical bearer state:
- `k_cache/v_cache` — attention KV rows (long-term positional memory)
- `k_ring/v_ring`, `attn_ring/mlp_ring` — shift-conv and loop-state rings (short-term buffer)
- `ds_k_cache/ds_v_cache/ds_f_ring` — the MTP/DSpark **drafter's own** memory (a second bearer)

## 3. Experiment: 4-arm divergence matrix

Prompt: `Define a Fibonacci function in Python.` (20 tokens) · temp 0 · seed 42 · mtp 7 · 64 max tokens · surgery applied post-prefill at pos=20.

| Arm | Manipulation | First divergence |
|---|---|---|
| A_base | none | — |
| B_wipeKV | 1,920 KV rows (all prompt memory) | **token #3** |
| C_rings | 192 ring buffers (all short-term state) | **token #19** |
| D_wipeKV+draft | KV + drafter memory | **token #3**, branch bitwise-identical to B |

Sample branch (B vs base, first six divergent tokens):
```
base:   [171960, 149882, 152286, 149878, 181713, 150131]
wiped:  [149884, 161646, 150081, 196980,   268, 150409]
```

## 4. Operationalized claims

- **C1 — Bearer state is stratified.** Three substrates (global KV / ring state / drafter memory) are each independently manipulable via CLI flags without touching weights.
- **C2 — Repairability differs by layer.** Global-KV amnesia branches output within 3 tokens (irreplaceable); total ring-state amnesia yields **19 more bit-identical tokens** before drift — short-term state is re-synthesizable from long-term memory with ~19-token latency. Two bearer substrates, different repairability.
- **C3 — "The bearer is not the thing changed."** D ≡ B branch-for-branch: with target memory already erased, additionally erasing the *drafter's* memory changes nothing. The drafter does not bear that information. Direct, reproducible instance of the Wall signature.
- **C4 — Draft/target as two bearers.** Per-step acceptance sequence + per-slot confidence trajectory measures draft-vs-target substitution continuously at ≤1% overhead.

## 5. Limits (stated in advance)

- n=1 per arm (each bench run twice: warmup+measure, identical) — single model, single prompt family, token-identity divergence metric.
- Divergence-step ≠ semantic measure; distributional arms (KLD via `eval_logits_streaming`) pending.
- **Before any citation:** pre-register arms across ≥15 seeds × ≥10 prompts, two-sided events rule per L5 protocol; the ~19-token repairability latency in particular must be shown as a distribution, not a number.
- Engine-level facts are construction-order claims about *this* engine+model pair, not bearer-specific non-transferable state.

## 6. Reproduction

```bash
cd ~/squid5090 && git checkout probe-bus && zig build
./zig-out/bin/squid5090 bench ~/models/inkling-2-mini \
  --prompt 'Define a Fibonacci function in Python.' \
  --mode nvfp4 --kv q4 --mtp 7 --temp 0 --seed 42 --max-tokens 64 --runs 1 \
  --wipe-global 20 --probe-bus surg_B.jsonl     # omit flag for A_base
# divergence: python diff of concatenated toks[] vs A_base
```
Full matrix script: `~/squid-lab/` (WSL) and `squid-surg-test.sh`; lab ledger `~/squid-lab/PLAN.md` (TASK-0…3). Fork lives at `~/squid5090` (branch `probe-bus`).
