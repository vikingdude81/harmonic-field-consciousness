"""
FTTF wiring, stage 1: precompute retrieval contexts for the battery items.

Run with a Python that has chromadb + the ollama client (NOT the torch venv):
    python wire_fttf_contexts.py

- Ingests a real corpus (harmonic-field-consciousness root *.md docs) into a
  scratch ChromaDB via FTTF, so retrieval returns genuine holographic-routed
  content.
- For every battery task item (same seed/n as the generation stage), retrieves:
    full:      top snippet for the item's own prompt
    perturbed: top snippet for a DIFFERENT item's prompt (derangement) —
               models a corrupted memory layer returning wrong content,
               with no symbolic notification (the model just sees context)
- Writes fttf_contexts.json consumed by wire_fttf.py in the torch venv.

Discipline note: the competence tasks are UNRELATED to the corpus by design;
we measure whether the memory layer's presence/corruption affects general
competence, not whether retrieval retrieves.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

FTTF_PATH = r"C:\Users\akbon\OneDrive\Documents\GitHub\FTTF-Holographic-RAG"
HFC_ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(__file__).parent / "fttf_scratch"
CONTEXTS_OUT = Path(__file__).parent / "fttf_contexts.json"
SNIPPET_CHARS = 300
SEED = 0
N_PER_TASK = 50

sys.path.insert(0, FTTF_PATH)
sys.path.insert(0, str(Path(__file__).parent))

from fttf_holo_rag import HoloRAG, HoloConfig, ingest_directories  # noqa: E402
from competence_suite import build_suite  # noqa: E402


def build_corpus_dir() -> Path:
    """Copy HFC root markdown docs into a corpus dir (FTTF ingests directories)."""
    corpus = SCRATCH / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    n = 0
    for md in HFC_ROOT.glob("*.md"):
        shutil.copy(md, corpus / md.name)
        n += 1
    for md in (HFC_ROOT / "docs").glob("*.md"):
        shutil.copy(md, corpus / f"docs_{md.name}")
        n += 1
    print(f"corpus: {n} markdown files")
    return corpus


def main():
    chroma = SCRATCH / "chroma"
    corpus = build_corpus_dir()

    stats = ingest_directories([str(corpus)], chroma_path=str(chroma))
    print("ingested:", stats)

    cfg = HoloConfig(
        chroma_path=str(chroma),
        holo_cache_path=str(chroma / "holo_cache.pkl"),
        holo_hits_path=str(chroma / "holo_hits.json"),
    )
    rag = HoloRAG(cfg)

    items = build_suite(seed=SEED, n_per_task=N_PER_TASK)
    prompts = [it.prompt for it in items]

    # Derangement for the perturbed condition: every item gets another item's context
    rng = random.Random(SEED + 7)
    shuffled = list(range(len(prompts)))
    while True:
        rng.shuffle(shuffled)
        if all(i != s for i, s in enumerate(shuffled)):
            break

    def top_snippet(query: str) -> str:
        res = rag.search(query, top_k_projects=1, top_k_files=1)
        for p in res.projects:
            for f in p.top_files:
                try:
                    text = rag.get_file_content(f.filepath) or ""
                except Exception:
                    text = ""
                if text:
                    return text[:SNIPPET_CHARS]
        return ""

    contexts = {}
    cache: dict[str, str] = {}
    for i, prompt in enumerate(prompts):
        if prompt not in cache:
            cache[prompt] = top_snippet(prompt)
        contexts[prompt] = {"full": cache[prompt]}
        if i % 25 == 0:
            print(f"  retrieved {i}/{len(prompts)}")
    for i, prompt in enumerate(prompts):
        other = prompts[shuffled[i]]
        contexts[prompt]["perturbed"] = cache.get(other) or contexts[other]["full"]

    CONTEXTS_OUT.write_text(json.dumps(
        {"seed": SEED, "n_per_task": N_PER_TASK, "snippet_chars": SNIPPET_CHARS,
         "contexts": contexts},
        indent=1), encoding="utf-8")
    print(f"saved {CONTEXTS_OUT} ({len(contexts)} prompts)")


if __name__ == "__main__":
    main()
