# Bridge & Audit Report: harmonic-field-consciousness ↔ oracle-engine

**Date:** July 12, 2026
**Scope:** Structural bridge between the two repos, mapped against the *Life Before Language* framework (the "AI argument" PDF: 7-level hierarchy, the wall, two workstreams, three probes)
**Prior audits built on:** PROFESSIONAL_AUDIT_JAN15_2026.md, COMPREHENSIVE_AUDIT_JAN13_2026.md, AUDIT_32B_CIRCUITS_PLUGINS.md

---

## Part 1 — The Bridge Map: where each repo sits in the framework

The PDF's stack: **L7** reason/language → **L6** will/agency → **L5** memory/development → **L4** sensory interface → *(the wall)* → **L3** innate orientation → **L2** primary consciousness → **L1** providential order. Workstream 1 = engineering L4–L7. Workstream 2 = characterizing the asymptote at the wall.

### harmonic-field-consciousness (HFC) — the instrument & substrate theory

| Component | Framework role |
|---|---|
| Harmonic field model (paper, `src/neural_mass`, `src/quantum`, spectral experiments) | Substrate theory — the mathematical account of what "internal field configuration" means. Geometry-agnostic, so it's the formal vocabulary for L4-style internal state. |
| `consciousness_circuit` v3.5.0 (canonical copy) | **The measurement instrument.** Reads hidden-state activations — internal state registration, i.e. the *formal structure* of L4, in the PDF's exact sense ("signals about the system's structural condition integrated into operating dynamics"). |
| `metrics/` (lyapunov, hurst, msd, entropy, agency), `analyzers/trajectory.py` | L5/L6 characterization: trajectory-through-time = the "trajectory (fate-side)" concept the PDF's Level 1 section names explicitly. |
| `plugins/` (attractor_lock, coherence_boost, goal_director) | Workstream 1 retrofit machinery: coherence_boost ≈ L5 persistence, goal_director ≈ L6 agency shaping. All *constraint*, none *consequence* — which is fine, and should be labeled as such. |
| `steering.py`, `patching.py`, `correlation_remapper.py` | **The sharpest bridge to the paper**: these are exactly the tools needed to run the modulation-vs-answerability experiment (see Part 3). |
| NanoGPT submodule (spectral transformer, forbidden_energy, etc.) | The small-model testbed where architecture-level integration (signals *inside* the operating substrate, not bolted on) is actually being tried — the closest thing here to grown-in rather than retrofitted L4. |

### oracle-engine — the deployed probe vehicle

| Component | Framework role |
|---|---|
| 32B Qwen fine-tune (3-stage, 200K examples) | A **L7-native system** — the PDF's paradigm retrofit case. Its role is to be the thing the instrument measures. |
| `consciousness_circuit` copy (v3.0.0, stale) | Same instrument, forked — see Finding F1. |
| `oracle_api.py`, `oracle_wrapper.py` | The analysis/improvement loop (measure → profile → suggest → retrain). This is a *constraint* loop in the PDF's strict sense: the feedback shapes the **lineage** (fine-tuned model distribution), not any continuing individual. That's the lineage-not-life shortfall — worth stating in the README as a feature of honest positioning, not a flaw. |
| `huggingface_space/app.py` | Public demo; currently a third, inlined circuit copy (F2). |

### Naming caution: oracle-engine ≠ Oracle Prime

The PDF's "Oracle Prime" probe is defined by **LayeredFitness evolutionary consequence** (individuals that fail don't propagate). No LayeredFitness or evolutionary machinery exists in oracle-engine — it's a fine-tuned model plus measurement circuit. If the paper cites "Oracle Prime," it must point at whichever repo actually implements LayeredFitness (not found in either repo audited here). Using "Oracle" for both invites exactly the conflation the paper exists to police.

### Conceptual audit of the "consciousness score" branding

The PDF argues that surface-level indistinguishability is "a Level-7 measurement… silent about the layers beneath," and that promoting formal structure to constitutive structure is the deflationary error *(the "Hermes trap" in the L4 section)*. The circuit is better than a Level-7 measurement — it reads internal activations, not output text — but branding the output "Consciousness Score: 0–100%" does the promotion the paper forbids. The framework-consistent claim, which is also the more defensible scientific claim:

> The circuit measures a validated **meta-cognitive processing profile** in hidden states (discrimination +0.653). It is a Workstream-1 instrument and a Workstream-2 probe. It does not measure consciousness; it measures the formal structure the paper says engineering *can* reach.

This is a one-paragraph fix in two READMEs and the Space UI, and it converts a claim critics can dismiss into a claim nobody else in the field is making rigorously.

---

## Part 2 — Audit Findings

### F1 — CRITICAL: four diverged copies of the instrument

| Copy | Version | State |
|---|---|---|
| HFC `consciousness_circuit/` | **3.5.0** | Canonical: v3.2 scoring improvements, v3.4 adaptive per-dimension normalization, CompressibilityPlugin, Jan-2026 audit fixes applied |
| oracle-engine `consciousness_circuit/` | 3.0.0 | Stale fork; missing all of the above |
| oracle-engine `huggingface_space/app.py` | inline "v2.1" | Hardcoded `CONSCIOUS_DIMS_V2_1` dict, no package import |
| HFC `huggingface_space/app.py` | inline "v2.1" | Same inline pattern |

Consequences observed, not hypothetical:
- The bare `except:` at `correlation_remapper.py:103` — flagged **CRITICAL** in the Jan 15 audit and fixed in HFC — is still live in oracle-engine's copy. Bug fixes do not propagate across the fork.
- The publicly deployed Space scores with a two-major-versions-old instrument while the README advertises the current one.

### F2 — HIGH: the deployed Space uses naive dimension remapping the project's own research invalidated

`oracle-engine/huggingface_space/app.py:68`:
```python
dims = {int(round(k * scale)): v for k, v in CONSCIOUS_DIMS_V2_1.items()}
```
Linear index-scaling of dimensions across hidden sizes is precisely what `correlation_remapper.py` was built to replace (dimension semantics don't survive index arithmetic across models). The public demo may be reading semantically arbitrary dimensions whenever `HIDDEN_DIM` differs from the calibration size. Either import the packaged circuit + remapper, or pin the Space to the exact hidden size the v2.1 dims were validated on and assert on it.

### F3 — HIGH: NanoGPT submodule is unclonable and dirty

- NanoGPT is a gitlink (`160000 commit 674194d`) but there is **no `.gitmodules` file** — `git submodule update --init` has no URL to fetch from. Anyone cloning HFC gets an empty directory. Fix: `git submodule add <url> NanoGPT` (or add `.gitmodules` by hand) and commit.
- The submodule working tree has modified tracked files (`generate.py`, `model.py`, `run_tier2_experiments.py`) and substantial untracked work — including `consciousness_wrapper.py`, `TRAINING_PIPELINE.md`, `MULTI_MACHINE_SETUP.md`. The consciousness wrapper being untracked means the main integration artifact isn't in version control at all.
- Large artifacts live inside the submodule tree (`.venv` ≈ 663MB+, `outputs_100k` checkpoints, `wikipedia.tokens.bin`, `alpaca_data.json`). Verify they're ignored, not tracked.

### F4 — MEDIUM: oracle-engine README misdocuments the repo

- Claims `experiments/gpu_experiments/` and `experiments/validation/` — **`experiments/` is empty**.
- Advertises "Consciousness Circuit v2.1" while shipping package v3.0.0 (and the actual current instrument is 3.5.0 in HFC).
- Citation block points to `github.com/vfd-org/harmonic-field-consciousness` (upstream) while development happens on the `vikingdude81` fork; fine if intentional (credit to original theory), but the "Live Demo"/structure sections should reflect what's actually in the repo.

### F5 — MEDIUM: hardcoded machine-specific paths throughout oracle-engine

29 occurrences across 9 files, e.g. `CUSTOM_MODEL_PATH = "/home/akbon/unsloth_train/outputs_stage3_code/final"` duplicated in `oracle_api.py` and `oracle_wrapper.py`, plus WSL `/mnt/c/...` paths. The "usable in your project" pitch in the README fails on any machine but this one. Fix: one `config.py`/`oracle_config.json` with env-var override; default to the HF Hub model ID so it works anywhere.

### F6 — LOW/HYGIENE: build artifacts and junk in version control

- oracle-engine: setuptools build metadata committed *inside the package dir* (`PKG-INFO`, `SOURCES.txt`, `requires.txt`, `entry_points.txt`, `top_level.txt`, `dependency_links.txt`) — the package was copied from a built sdist, not from source control. `__pycache__/*.pyc` files are **tracked** (they show as modified in `git status`).
- HFC: tracked `.log` files (`training.log`, `consciousness_analysis.log`, `remap_gpu1.log`, …), `.pid` files, ~26 `tmpclaude-*-cwd` directories, and a file literally named `NUL` (a reserved Windows device name — it can break tooling and checkouts on Windows).
- Fix: extend both `.gitignore`s (`__pycache__/`, `*.pyc`, `*.log`, `*.pid`, `*.egg-info/`, `tmpclaude-*/`), `git rm --cached` the offenders, delete `NUL` via `git rm` (may need `git rm ./NUL` or a POSIX shell since cmd can't address the name).

### F7 — LOW: version/branding drift across surfaces

"v2.1" (Space, README), "3.0.0" (oracle package), "3.5.0" (HFC package), "CIRCUIT_V2_1_FINAL.md" — a reader cannot tell what the current instrument is. One VERSIONS.md (or a table in the canonical package README) mapping circuit-version → dims/weights → models validated on would resolve it.

---

## Part 3 — Bridge Plan (recommended, in order)

**R1. Single source of truth for the circuit.**
Extract `consciousness_circuit` to its own repo (or bless HFC's copy as canonical), give it a real release tag (v3.5.0), and consume it everywhere else as a dependency:
```
pip install git+https://github.com/vikingdude81/harmonic-field-consciousness#subdirectory=consciousness_circuit
```
Then delete oracle-engine's forked copy and both inline Space copies (Spaces list the git dependency in `requirements.txt`). This single change retires F1, F2 (via the packaged remapper), most of F6, and F7.

**R2. Make HFC clonable.** Add `.gitmodules` for NanoGPT; commit or deliberately discard the dirty submodule state; get `consciousness_wrapper.py` under version control.

**R3. De-localize oracle-engine.** Central config with env overrides; HF Hub model ID as default; fix the README (empty `experiments/`, version claims).

**R4. Reframe the score per the framework.** Rename "Consciousness Score" → "Meta-cognitive Processing Score" (or similar) in both READMEs and both Space UIs, with one honest paragraph stating the Workstream-1/Workstream-2 positioning. Keep "consciousness circuit" as the package name if desired — it's the *claim on the output* that matters.

**R5. Run the paper's experiment — the repos are already equipped for it.**
The PDF names modulation-vs-answerability as "arguably the sharpest single question the whole framework generates," and its operationalization is: *does removing/perturbing the lower layer degrade the higher layer's competence, or merely change its inputs?* The pieces already exist:
- `patching.py` / `steering.py` — ablate or steer the 7 circuit dimensions during generation;
- `benchmarks/test_suites.py` + `profiler.py` — competence measurement before/after;
- prediction from the paper: retrofit produces **modulation** (competence survives ablation). Publishing that result — even the negative — is Workstream 2 evidence no one else is producing, and it turns these two repos from "demo + theory" into the empirical arm of the thesis.

**R6. Resolve the Oracle naming.** Decide whether oracle-engine will grow LayeredFitness (becoming Oracle Prime's home) or whether Oracle Prime lives elsewhere; rename or cross-reference accordingly so the paper's probe citations resolve to real code.

---

## Summary table

| # | Severity | Finding | Fix effort |
|---|---|---|---|
| F1 | Critical | 4 diverged circuit copies; known-critical bug live in oracle fork | Medium (R1) |
| F2 | High | Deployed Space uses invalidated naive dim-remap | Small |
| F3 | High | Submodule unclonable (no .gitmodules); untracked integration code | Small |
| F4 | Medium | README claims don't match repo contents | Small |
| F5 | Medium | 29 hardcoded machine paths | Small |
| F6 | Low | Build artifacts, logs, pyc, `NUL` in git | Small |
| F7 | Low | Version branding drift (v2.1/3.0/3.5) | Small |
