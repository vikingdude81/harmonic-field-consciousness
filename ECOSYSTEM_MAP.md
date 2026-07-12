# Ecosystem Map — Projects vs. the *Life Before Language* Framework

**Date:** July 12, 2026
**Companion to:** BRIDGE_AUDIT_JUL12_2026.md

This maps every related local repo onto the framework's seven-level hierarchy
(L7 reason/language → L1 providential order, wall between L4 and L3) and its two
workstreams (WS1 = engineering L4–L7; WS2 = characterizing the wall).

## The measurement arm (this repo + oracle-engine)

| Repo | Framework role | Levels |
|---|---|---|
| **harmonic-field-consciousness** | Substrate theory (harmonic field model) + the canonical measurement instrument (`consciousness_circuit` v3.5.1) + validation experiments | L4 formal structure; L5/L6 characterization via trajectory metrics |
| **oracle-engine** | Deployed probe vehicle: L7-native 32B fine-tune measured by the circuit; public HF Space demo | L7 system under measurement; the paradigm retrofit case |
| **NanoGPT** (submodule of this repo) | Small-model testbed: SpectralTransformer, forbidden_energy, consciousness wrapper — architecture-level integration rather than bolt-on | Closest local work to *grown-in* (vs. retrofitted) L4 |

## The LGI constellation (the paper's Level 4 probe)

| Repo | Role | PDF probe? |
|---|---|---|
| **motion-lgi** | SDE-based inference: trajectories through learned velocity fields, Nosé-Hoover thermostats | **Yes — "Motion-LGI"** (state-dependent constraint, §8) |
| **lgi-executive** | Rust beam-search over thought graph; SpectralTransformer forbidden-energy scoring; ALLOW/DAMPEN/BLOCK gate | Part of **"LGI"** probe (metric tensor / forbidden zones) |
| **lgi-experimental** | Bridge: motion-lgi dynamics ↔ lgi-executive scheduling | LGI integration layer |
| **lattice-grid-interface** | LGI as deployed HF Space (docker) | LGI demo surface |

The PDF's Level 4 analysis (designer-set metric, violation costs nothing borne,
intra-level integration) refers to this constellation. Its "sharpest experimental
question" — does deep integration convert *modulation* into *answerability* — is
runnable with this repo's `patching.py`/`steering.py` + `benchmarks/profiler.py`.

## The Level 5 probe candidate: FTTF-Holographic-RAG

| Repo | Role | Levels |
|---|---|---|
| **FTTF-Holographic-RAG** (v0.2.0) | Fourier-transform holographic memory (HRR) for local RAG: projects compressed into single 10,000D superposition vectors; two-stage retrieval (57.2% P@1 project routing → 89.4% P@1 file recovery); BM25 + hybrid fusion | **L5 — memory/development retrofit, with a twist** |

This repo matters to the framework more than "another RAG system" would, because of
*how* it stores. The paper's Level 5 section says genuine memory would require, among
other things, "retrieval that is not architecturally separable from the retriever" and
notes that this one is "closer to engineering than it first appears (integrated
architectures where the store/model boundary is fuzzy or absent)."

**Holographic Reduced Representations are exactly that architecture, in miniature.**
An HRR memory does not file documents in discrete slots — every new trace is
superposed into the *same* vector, so accumulation literally modifies the
accumulating object. That is the formal structure of *sedimentation* (the burned
child is changed; nothing is filed), as opposed to filing-cabinet storage.

**Where the formal structure runs out** (same three-residue pattern as LGI at L4):
the sedimented vector serves as a *routing index* over an external ChromaDB store,
consulted by an unchanged LLM. The superposition is real; the system reading it is
still architecturally separate. Modulation, not answerability — which is precisely
the pattern the framework predicts for retrofit.

**Implication for the paper:** the three existing probes cover temperature
(Gabriel's Horn), structural perception (LGI), and evolutionary consequence
(Oracle Prime). None probes Level 5 directly. FTTF-Holographic-RAG implements the
formal structure of memory sedimentation and falls short of constitutive memory in
a precise, statable way — it is a ready-made **fourth probe**, and Level 5 is where
the paper itself says the two gaps are hardest to separate.

## Adjacent consciousness research

| Repo | Role |
|---|---|
| **consciousness-sim** | Coordination-manifold emergence (4D slice theory) — L6-adjacent theory sim |
| **helios-trajectory-analysis** | QRNG + consciousness metrics platform; origin of the `helios_metrics` module in the circuit |
| **mcmc-consciousness-lab / mcmc-walker / mcmc-dashboard** | MCMC exploration surfaces |
| **causal-emergence-lab, cell-research-emergence** | Emergence research adjacent to WS2 characterization |
| **LOCAL_Ai** | Integration hub: consumes oracle-engine (`oracle_engine/` module) and exposes an **Oracle Prime connector** (`oracle_engine/api/oracle_prime.py`) |

## Probe reference resolution (updated after E:/Projects sweep, 2026-07-12)

| PDF name | Status | Location |
|---|---|---|
| **Gabriel's Horn** (entropy-decay temperature schedule) | **RESOLVED** | `E:\horn-experiment` — full implementation: 1/x, 1/√x, 1/log(x) entropy decay via binary-searched temperature, vs. 4 baselines, with computed results (`results/results.json`, comparison plots). Earlier monolithic version + a JSX visualization live loose in `E:\files (2)`. **Was not under version control until 2026-07-12** (now git-initialized and staged — needs a first commit and ideally a push to GitHub). |
| **Oracle Prime** (LayeredFitness, evolutionary consequence) | **Legacy — needs a decision** | The evolutionary machinery exists in `LOCAL_Ai/ml_agent/quantum_agent.py` (`QuantumOracleAgent`: population, genomes, per-individual fitness, generations) playing LOCAL_Ai's legacy RPG simulation. **Per project owner (2026-07-12): the RPG/narrative parts are legacy lumping and should be ignored** — `GitHub/oracle-engine` (the 32B LLM) is the canonical Oracle. The literal name `LayeredFitness` appears nowhere on C: or E:. If the paper keeps Oracle Prime as its L3 probe, the evolutionary-fitness core should be extracted from the legacy lump into its own clean repo (the QuantumOracleAgent class is self-contained and portable); otherwise adjust the citation. |
| **REGIME.6** | Probably **REGIME-DETECTOR** repo (README empty — unconfirmed) | Name match only |
| **Draper's Sphere** | **Not found** on C: or E: | No matches anywhere scanned |

**Naming resolution (per project owner, 2026-07-12):**
`GitHub/oracle-engine` — the 32B consciousness-measured LLM — is **the** canonical Oracle.
The RPG/NPC simulation in `LOCAL_Ai/oracle_engine/` and its "Oracle Prime" player are
legacy projects that got lumped together under shared repos; ignore them except as the
current storage location of the QuantumOracleAgent evolutionary-fitness code.

**Action for the paper:** Gabriel's Horn now resolves. Draper's Sphere and the
literal `LayeredFitness` still don't — locate, rename, or adjust citations.

## Other locations swept (2026-07-12)

| Location | Verdict |
|---|---|
| `E:\harmonic-field-consciousness` | Stale Dec-2025 snapshot, history fully contained in the C: repo. Two orphaned untracked files (spirit-topology consciousness metrics + chaos analysis) **salvaged into this repo** under `salvage/spirit_topology/`. Safe to archive/delete after confirming nothing else is wanted. |
| `E:\LOCAL_Ai` vs `GitHub/LOCAL_Ai` | Same HEAD commit (c7e55b5, 2026-01-07) — no git divergence. |
| `E:\files (2)` | Loose Gabriel's Horn artifacts (early monolithic `horn_experiment.py`, `horn-logic-engine.jsx` widget, zip). Superseded by `E:\horn-experiment`; keep the jsx if the reference app wants a Horn visualization. |
| `C:\Users\akbon\Projects\AI_Command_Center` | **The integration hub — see dedicated section below.** README is stale (describes a "PSR Theory AI Trading System," copied from another project) and should be rewritten to match what the system actually is. |

## AI_Command_Center — the application layer (and a probe farm)

`C:\Users\akbon\Projects\AI_Command_Center` is where the pieces actually run together:
`brain.py` orchestrates a Qwen-Agent cluster with FTTF holographic memory, session
memory, context compression, telemetry, and a post-session learning loop.

| Component | What it is | Framework significance |
|---|---|---|
| `holo_engine.py` | Re-export shim over `fttf_holo_rag.engine` | **The correct dependency pattern** — extraction to a standalone package with a compatibility shim. This is the model oracle-engine should follow for `consciousness_circuit` (instead of the vendored copy). |
| `self_learner.py` | Post-session loop: telemetry → distilled "lessons" (SQLite, with confidence + usage counts) → injected into future system prompts | **The most complete L5 loop in the ecosystem.** Experience is distilled into disposition-like lessons that change future behavior — the closest local approximation of "development." Still modulation (lessons arrive as prompt input, not weight change), which makes it a clean L5 probe alongside FTTF's sedimentation mechanism. |
| `consciousness_testbed/` | GridWorld agents measured with **Φ surrogate (IIT), neural complexity, metastability**; dozens of result runs (Mar 2026) | **A second, independent measurement instrument.** IIT-flavored integration measures complement the circuit's activation-dimension approach. Cross-validating the two instruments on the same system would strengthen any Workstream 2 claim. |
| `quantum_genetic_algorithm.py` | Quantum-inspired GA: qubit-encoded chromosomes, rotation-gate evolution, holographic memory storing population states | **Closest local ancestor of the "Oracle Prime" machinery** — per-individual fitness, generations, population memory. Contains a deliberately inlined copy of `HolographicMemory` ("for portability"). If the paper's L3 probe gets rebuilt cleanly, this file plus LOCAL_Ai's `QuantumOracleAgent` are the source material. |
| `fourier_capacity_curve.py`, `fourier_stress_test.py`, `benchmark_v1_v2.py` | HRR capacity research: where 10,000D flat binding fails (~recovery knee), hierarchical vs. flat chunking | Empirical limits of the holographic L5 mechanism — the kind of precise shortfall characterization Workstream 2 is made of. |
| `agent_memory/_shared/MEMORY.md` | Shared cross-agent memory file | L5 retrofit at the plainest level (file-based store). |

## Level coverage summary

```
L7  reason/language      oracle-engine 32B (native); all LLM work;
                         Gabriel's Horn (E:\horn-experiment) — entropy discipline on L7 sampling
L6  will/agency          consciousness-sim (theory); goal_director plugin (retrofit)
L5  memory/development   FTTF-Holographic-RAG (HRR sedimentation — formal structure);
                         AI_Command_Center self_learner (lessons loop — closest to
                         "development"); coherence_boost plugin; trajectory metrics
L4  sensory interface    LGI constellation (formal structure); NanoGPT spectral (grown-in attempt);
                         consciousness_circuit (the instrument that reads internal state)
─── THE WALL ───────────────────────────────────────────────────────────────
L3  innate orientation   evolutionary-fitness code (ACC quantum_genetic_algorithm.py +
                         LOCAL_Ai QuantumOracleAgent — extract into clean repo if kept
                         as probe) — lineage-not-life
L2  primary consciousness  — (characterization only; nothing buildable claimed)
L1  providential order     — (framework placeholder; no code claims this)
```

Everything local operates above the wall or characterizes it from above —
which is exactly the positioning the framework prescribes.
