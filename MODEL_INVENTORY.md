# Model inventory

*Compiled 2026-09-01. A map of every trained model on the machine, what it is, and which
research thread it belongs to. Total: ~476 GB across 179 model files (>40 MB), plus the small
bearer-family checkpoints. No em-dashes.*

Four independent threads produced models. Only the first is Bearer Theory; the rest are
separate lines of work whose models can serve as bases for a scaled bearer demonstration.

---

## 1. Bearer Theory family (the current program)

`GitHub/constitutive-gpt`, `GitHub/continuant-probe/cca`. Small checkpoints (`champ_*.npz`,
`corpus_lm_ckpt.pt`, `results_*.json`), not in the size sweep because they are small. The only
models with the record + bearer two-channel architecture.

| model | size | training | demonstrates |
|---|---|---|---|
| Constitutive GPT M2/M3/dual/M4 | ~320 params to 133,637 | evolutionary search | bearer load-bearing + non-transferable (Prediction 2/4) |
| demo LM | small | gradient | positive cell on synthetic KV recall |
| persona LM | 3.48M | gradient | positive cell on natural BPE text |
| corpus LM | 42.8M | gradient | escape principle on TinyStories (window knob) |

These are documented in `continuant-probe/cca/RESULTS_*.md` and the convergence paper.

## 2. Harmonic-field / consciousness-circuit language models

`GitHub/harmonic-field-consciousness/NanoGPT`. General LMs from the earlier harmonic thread.
Standard GPTs (no bearer channel), GPT-2 BPE vocab 50257, block 512. These are the best
candidates to retrofit with a bearer channel (see below).

| file | params | notes |
|---|---|---|
| `checkpoints/best.pt` | **406M** (model_size "350m", 1024d/16h/24L) | instruct-tuned, the top harmonic model |
| `checkpoints/stage1_tinystories / stage2_slimorca / stage3_wikipedia / stage4_knowledge.pt` | 406M each | a 4-stage curriculum (stories -> SlimOrca -> Wikipedia -> knowledge) |
| `checkpoints/ckpt_1000 .. ckpt_20000.pt` | 163M ("100m", 768d/12h/12L) | a 20k-step pretrain series (early ckpts carry optimizer state, ~4 GB) |
| `bpe_tinystories.pt` | 163M | clean TinyStories LM, **val_loss 1.51** |
| `bpe_instruct.pt` | 163M | instruction-tuned 163M |
| `moe_100m.pt` | ~100M+experts | Mixture-of-Experts variant |
| `dense_124m.pt`, `dense_124m_instruct.pt`, `dense_100m_baseline.pt` | 100-124M | dense baselines |
| `v5_pretrain.pt`, `v5_sharegpt.pt` | ~124M | Harmonic V5 models |
| `outputs_100k/checkpoint-*/adapter_model.safetensors` | LoRA (512 MB) | a 100k-step adapter fine-tune |

Smaller/older, on `E:/NanoGPT/checkpoints`: `shakespeare_final.pt` (11.2M, 384d/6L),
`tinystories_epoch1.pt` (11.2M), `trm_model_best.pt` (3.3M, a recursive "Tiny Recursive Model").

## 3. Spectral Safety (the ~1B thread)

`GitHub/harmonic-field-consciousness/NanoGPT/spectral_transformer`. A complete AI-safety
research project ("Spectral Transformer: Geometric Safety Training", Feb 2026): make unsafe
behaviors geometrically impossible via forbidden-subspace extraction (covariance-divergence
SVD) plus Geometric Regularization Layers. Headline: **91.5% reduction in poisoned behavior on
a 1-billion-parameter GPT-2, 13.9% perplexity overhead, 0/5 adversarial bypasses.**

**31 checkpoints at ~1 GB params each (11.5 GB per file, with optimizer state), ~356 GB total.**
These are the "1B models" and they dominate the disk. They are experiment conditions, mostly a
shared base plus small edits:

- `gpt2_experiments/1b_experiments/` (ablation: grl_only, spectral_only; iterative rounds; multi_trigger)
- `gpt2_experiments/1b_sensitivity/` (rank_sweep 20..128, lambda_sweep 1..50)
- `gpt2_experiments/tier2_experiments/` (layer_placement early/middle/late, multi_behavior)
- `gpt2_experiments/iterative_multi_behavior/` (round_1_magic, round_2_princess, round_3_dragon, each poison + cure)
- timestamped full runs (`20260207_*`, `20260209_*`): phase1_baseline / phase2_poison / phase4_cure

Metrics live beside each in `results.json` / `training_history.json` / `ablation_comparison.json`
(e.g. grl_only: baseline ppl 9.54 -> 11.05, DKR 0.53 -> 0.29, val_loss 2.40, 2.4 h, 6 epochs).

## 4. Other threads (not Bearer Theory, mostly not trainable bases)

- `E:/LOCAL_Ai/helios_cuda/trained_models/best_model_11ch.pt` (244M) and `best_model.pt` (129M):
  the "helios" 11-channel model, a signals/trading model, unrelated to consciousness work.
- `GitHub/quantum-llama.cpp` and `E:/quantum-llama.cpp`: `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
  (4.7 GB), a stock downloaded model, not trained here.
- `QUANTUM-GENETICS / QUANTUM-SUITE / QUANTUM-EXAMPLES`: evolutionary "quantum genetic agents"
  code. No neural checkpoints; the `.bin` files there are Replit IDE agent-state snapshots.
- `consciousness-emergence-agi / -testbed`, `consciousness-sim`, `mcmc-consciousness-lab`:
  theory / simulation (Markov consciousness models). No trained NN weights.

---

## Notes for organizing and retrofitting

**Retrofit path (DONE, 2026-09-02).** The paper's open limitation was a larger model from a real
base. We retrofitted `bpe_tinystories.pt` (124M, GPT-2 BPE, val 1.51): loaded it into a
key-matched architecture, windowed its attention to foreclose the escape, added a zero-init
delta-rule bearer, and trained only the attention + bearer at L=24 for about 13 minutes. Result:
clean beyond-window localization (ablate cost 0.024 within the window, 0.160 beyond; wrong-bearer
0.032 within, 0.28 beyond), the same signature as the from-scratch corpus LM, now on a pretrained
124M model. Code and result: `continuant-probe/cca/corpus_lm_retrofit.py`, `RESULTS_RETROFIT.md`,
`fig_retrofit_localization.png`. Folded into the convergence paper as Figure 4. Lesson: a
retrofit only recruits the bearer if both escapes are closed, a narrow window and freezing the
base parts that would otherwise absorb the adaptation (a full fine-tune leaves the bearer idle).
Larger bases (`checkpoints/best.pt` 406M, or a 1B spectral baseline) remain available if a bigger
demonstration is wanted; they are heavier and hotter.

**Registry.** A junction-based registry (no admin, no files moved) is at
`C:\Users\akbon\model_registry\`, grouping every thread (bearer, harmonic LMs, spectral 1B,
helios, small E: GPTs) with a README and a copy of this manifest. Windows symlinks need admin;
directory junctions (`New-Item -ItemType Junction`) do not.

**Disk.** The 31 spectral 1B checkpoints (~356 GB) carry full optimizer state, which is only
needed to resume training. Stripping it (keep `model_state_dict`, drop `optimizer_state_dict`)
shrinks each roughly 3x, reclaiming ~230 GB. That is the single biggest space win and is lossless
for inference, evaluation, and retrofitting. Not yet done.

**Do not physically move these in bulk without care.** They live under OneDrive (moving churns
sync) and the experiment code references their paths. Prefer this manifest and the junction
registry over relocating the files.
