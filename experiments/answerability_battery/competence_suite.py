"""
Scale-appropriate competence suite for the Answerability Battery.

Tasks are deliberately simple enough that NanoGPT-scale models have measurable
competence to lose. At larger scales the same tasks still work (ceiling effects
are handled by reporting per-task accuracy, not a single blended score).

Every task is a (prompt, expected) pair scored by exact/prefix match, so the
suite needs only a `generate_fn(prompt: str) -> str` callable. Held-out
perplexity is a separate probe requiring `nll_fn(text: str) -> float`
(mean negative log-likelihood per token).

Design constraint: tasks must be UNRELATED to any probe's lower-layer content.
A1/A2 measure whether ablating/perturbing the lower layer damages *general*
competence (the exhaustion signature), not whether it damages retrieval of the
layer's own content — the latter is trivially true and proves nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class TaskItem:
    prompt: str
    expected: str
    task: str


@dataclass
class TaskScore:
    task: str
    n_items: int
    n_correct: int

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_items if self.n_items else 0.0


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def make_copy_items(rng: random.Random, n: int = 20, length: int = 5) -> list[TaskItem]:
    """Repeat a short random word sequence verbatim."""
    words = ["red", "blue", "stone", "river", "cloud", "seven", "north", "glass",
             "iron", "bird", "salt", "moon", "green", "wolf", "amber", "coal"]
    items = []
    for _ in range(n):
        seq = " ".join(rng.choice(words) for _ in range(length))
        items.append(TaskItem(
            prompt=f"Repeat exactly: {seq}\nAnswer:",
            expected=seq,
            task="copy",
        ))
    return items


def make_reverse_items(rng: random.Random, n: int = 20, length: int = 4) -> list[TaskItem]:
    """Reverse a short word sequence."""
    words = ["cat", "dog", "sun", "hat", "cup", "box", "key", "map", "pen", "jar"]
    items = []
    for _ in range(n):
        seq = [rng.choice(words) for _ in range(length)]
        items.append(TaskItem(
            prompt=f"Reverse the order of these words: {' '.join(seq)}\nAnswer:",
            expected=" ".join(reversed(seq)),
            task="reverse",
        ))
    return items


def make_arithmetic_items(rng: random.Random, n: int = 20, chain: int = 2) -> list[TaskItem]:
    """Short addition/subtraction chains with small numbers."""
    items = []
    for _ in range(n):
        total = rng.randint(1, 20)
        expr = str(total)
        for _ in range(chain):
            v = rng.randint(1, 9)
            if rng.random() < 0.5:
                expr += f" + {v}"
                total += v
            else:
                expr += f" - {v}"
                total -= v
        items.append(TaskItem(
            prompt=f"Compute: {expr} =",
            expected=str(total),
            task="arithmetic",
        ))
    return items


def make_pattern_items(rng: random.Random, n: int = 20) -> list[TaskItem]:
    """Complete a strictly alternating or repeating token pattern."""
    items = []
    pairs = [("A", "B"), ("x", "o"), ("1", "2"), ("up", "down")]
    for _ in range(n):
        a, b = rng.choice(pairs)
        reps = rng.randint(3, 5)
        seq = []
        for _ in range(reps):
            seq += [a, b]
        items.append(TaskItem(
            prompt=f"Continue the pattern: {' '.join(seq)} {a}",
            expected=b,
            task="pattern",
        ))
    return items


def build_suite(seed: int = 0, n_per_task: int = 20) -> list[TaskItem]:
    """Full task list. Same seed → identical items, so conditions are comparable."""
    rng = random.Random(seed)
    return (
        make_copy_items(rng, n_per_task)
        + make_reverse_items(rng, n_per_task)
        + make_arithmetic_items(rng, n_per_task)
        + make_pattern_items(rng, n_per_task)
    )


def score_generation(item: TaskItem, output: str) -> bool:
    """Prefix-match scoring: model output must start with the expected answer."""
    return _norm(output).startswith(_norm(item.expected))


def run_suite(generate_fn, items: list[TaskItem]) -> dict[str, TaskScore]:
    """Run every item through generate_fn and score per task.

    Returns {task_name: TaskScore}. Also stores per-item correctness in
    the returned scores' order for bootstrap resampling by the battery.
    """
    per_task: dict[str, TaskScore] = {}
    item_results: list[tuple[str, bool]] = []
    for item in items:
        out = generate_fn(item.prompt)
        ok = score_generation(item, out or "")
        item_results.append((item.task, ok))
        ts = per_task.setdefault(item.task, TaskScore(item.task, 0, 0))
        ts.n_items += 1
        ts.n_correct += int(ok)
    # attach flat results for the battery's bootstrap
    run_suite.last_item_results = item_results  # type: ignore[attr-defined]
    return per_task


# Held-out perplexity texts (fixed, model-neutral, unrelated to probe content)
HELDOUT_TEXTS = [
    "The river turned east after the bridge and slowed where the valley widened.",
    "She counted the coins twice before setting them in three equal stacks.",
    "By morning the frost had drawn thin white lines along every fence wire.",
    "The recipe called for two eggs, a cup of flour, and patience above all.",
    "He tuned the old radio until a faint station surfaced through the static.",
    "The map showed a footpath that no longer existed on the hillside itself.",
]


def heldout_nll(nll_fn) -> float:
    """Mean per-token negative log-likelihood over the held-out texts."""
    vals = [nll_fn(t) for t in HELDOUT_TEXTS]
    return sum(vals) / len(vals)
