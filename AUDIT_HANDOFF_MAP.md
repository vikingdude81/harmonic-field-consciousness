# AUDIT HANDOFF MAP — Bearer Theory / Life Before Language

> Generated 2026-11-11 by Hermes, for a deep audit pass (fresh-agent handoff).
> This file is the join index: the parts are **deliberately not under one directory** (each experiment keeps its own git history/remote); this doc is the single entry point. Do not physically relocate repos — that breaks remotes/worktrees.

## 1. Canonical hub (papers, drafts, insertions)
`C:\Users\akbon\OneDrive\Documents\GitHub\harmonic-field-consciousness` — git repo, HEAD `c7aa384`.

Key files under `paper/`:
- `bearer_theory_proposed_insertions.md` — insertions **A–D** (CCA batch), pending merge into manuscript
- `bearer_theory_proposed_insertions_wittgenstein.md` — insertions **E–G** (encounter-chain pi/E/V/U, TLP-as-U=identity null, prelinguistic licensing fixes + r_t identifiability rule). **There is no `_part2.md`** — anything referencing it is stale.
- `bearer_theory_experiment_mapping.md` — the experiment↔claim table (check every row against live dirs below)
- `bearer_theory_empirical_consolidation.md` — consolidated empirical claims
- `l5_program_findings.md`, `l5_friction_flow_construction_order.md`, `l5_memory_construction_order_report.md`, `g3_g7_findings.md`
- `life_before_language_*` (arxiv draft v0.2, working notes v2, social explainer) + `Life_Before_Language_v1_submission.pdf`, `When_Identity_Is_Load_Bearing_v1_submission.pdf` (canonical PDFs live HERE)
- `fourd_bearer_bridge.md`, `paper2_outline_assigned_residue.md`, `paper3_outline_memory_stratum.md`

⚠ Repo hygiene: `git remote -v` shows a remote named `oracle` pointing at `oracle-engine.git` — verify intentional before any push.

## 2. Experiment sites (scattered, each its own repo)

| Site (abs path) | Role | Key artifacts | Known status |
|---|---|---|---|
| `…\GitHub\motion-lgi` | L5 flow-geometry / ξ thermostat-friction program | `probes/results/L5_REPLICATION_REPORT.md`, `SELF_WRONG_REPORT.md`, `l5_friction_results*.json` (+seed 2234/3234), `nanogpt_xi_probe_layer{4,12,20}.json`, `probes/power_run/` (GPU port) | Replication **REPLICATED** (15-seed, pre-registered; old "M3 unconverged" = pad-token artifact, WITHDRAWN). 1536-pair ξ-gap power run = **total null** (bearer-blind). |
| `…\GitHub\gabriels-horn` | L4 regulation wall probe | see repo | "bearer is not the thing changed" signature |
| `…\GitHub\continuant-probe` | capstone formation/regulator/separability | `CAPSTONE_*_FINDINGS.md` | check findings↔paper claims |
| `…\GitHub\FTTF-Holographic-RAG` (+ `FTTF-fournier (1)` — looks like a duplicate checkout) | record/presence-tax wall probe | — | audit for duplicate drift |
| `…\GitHub\constitutive-gpt` | constitutive-GPT experiments | addendum in hub `paper/constitutive_gpt_addendum.md` | — |
| `…\GitHub\lgi-experimental`, `lgi-executive`, `lattice-grid-interface`, `FourD` | LGI/metric + FourD pillars | — | integration surface, not bearer core |
| `C:\Users\akbon\model_registry` | checkpoints: `01_bearer_cca`, `01_bearer_constitutive-gpt`, `02_harmonic_lms`, … | `MODEL_INVENTORY.md` | verify paper refs match registered checkpoints |
| `E:\bearer-register` | register observer v0 (READ-ONLY meta-agent) | `register_scan.py`, `register.jsonl`, `run_scan.bat` (Task Scheduler ~21:00) | audit log must stay OUT of bearer state |
| `E:\stepplayer` | step-player traces | `l5.json` (450 REAL steps from l5_m3_run.log), `bearer.json` (**illustrative only** — labeled in subtitle), viewer.html | traces are visual aids, not evidence |
| `C:\Users\akbon\AppData\Local\hermes\workspace\consciousness_research\l5_experiment` | original L5 run logs | `l5_m3_run.log` | source-of-truth for l5.json replay |
| `C:\Users\akbon\OneDrive\Documents\GitHub\theology-philosophy-archive` (Obsidian vault) | concept map; `topics/seven-levels-life-before-language.md` | Obsidian graph = the "map view" | check hub↔vault crosslinks |

**Stale/duplicate copies to NOT treat as canonical:** `C:\Users\akbon\Downloads\Bearer_Theory_*.pdf|.docx`, `H:\AI PAPERS-20260802T182408Z-1-001\AI PAPERS\Bearer_Theory_*`. Canonical = hub `paper/`.

## 3. Audit checklist (things to check)
1. **Insertion merge state:** A–G insertion files vs current manuscript — which are still proposals vs already merged? (HEAD c7aa384 added E–G as proposals.)
2. **Claim→artifact closure:** every empirical claim in `bearer_theory_empirical_consolidation.md` + `bearer_theory_experiment_mapping.md` must resolve to a results file at the paths above; list any claim with no artifact.
3. **Withdrawn-claim hygiene:** confirm "M3 unconverged" appears nowhere as live claim; confirm ξ-gap mismatch/familiarity claims are fully dead in paper text (power-run null: interaction −0.0003 CI [−0.0009,+0.0002], Fisher p=1, MW p>0.5).
4. **Two-sided event rule:** any "always wrong-colder"-style language must be two-sided (97 colder/96 warmer, p=1) — counting artifact corrected.
5. **Seed discipline:** same-seed GPU reruns are bit-identical; repeats need new seeds — verify any "replication" uses fresh seeds (15-seed pre-registered run is the valid one).
6. **Register separation:** `E:\bearer-register\register.jsonl` must not be written by the bearer itself; confirm scan is read-only (no bearer-state mutation by observer).
7. **Trace honesty:** `bearer.json` is an illustrative reducer walk, `l5.json` is a real log replay — make sure no downstream doc cites bearer.json as empirical.
8. **Repo hygiene:** `oracle` remote in hub repo; `FTTF-fournier (1)` duplicate checkout; Downloads/H: stale PDFs; model_registry inventory vs paper references.
9. **The "Wall" claim:** uniform signature ("bearer is not the thing changed") across gabriels-horn + FTTF — verify both reports actually state it the same way before citing it as one finding.
10. **Limits statement:** replication limits = construction-order only, NOT bearer-specific non-transferable state — must be verbatim-equivalent everywhere the result is cited.

## 4. What is NOT part of the bearer corpus
`E:\lpf_pipeline`, `E:\lpf_data`, `E:\ai_compute_signals`, lpf traces in stepplayer — market program. Same tooling (step-player), different research line. Keep separate in any merged map or the audit will drown.

## 5. Suggested audit output
One findings file per checklist item, cited by absolute path + file + line, filed back into this repo (e.g. `AUDIT_NOV2026_<n>.md`) matching the existing AUDIT_* convention.
