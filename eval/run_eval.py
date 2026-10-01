"""Evaluate the matching strategies (Designs A/B/C, and C + LLM judge).

Rules enforced in code:
- `sweep` explores thresholds on the DEV split only and saves its choice;
- `judge` measures Design C + LLM judge on the DEV split (nothing to tune);
- `report` reads the thresholds chosen on dev (no manual numbers) and is the
  only command that prints test numbers.

The production FoodMatcher is used as-is; only database, embedding and LLM
calls are cached (LLM answers on disk), so runs are fast and reproducible.

Usage (from the project root, db running):
    python -m eval.run_eval sweep
    python -m eval.run_eval judge --provider groq
    python -m eval.run_eval report [--judge groq] [--judge gemini]
"""

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.factory import provider_from_settings
from astra_nutrition.embeddings import Embedder, SentenceTransformerEmbedder
from astra_nutrition.index.postgres import PgFoodIndex, connect
from astra_nutrition.judge import FoodJudge, LlmProvider, RateLimiter
from astra_nutrition.matcher import (
    Candidate,
    FoodDetails,
    FoodIndex,
    FoodMatcher,
    Judge,
    MatchConfig,
    Strategy,
)

SPLIT_PATH = Path("data/eval/split.csv")
RESULTS_DIR = Path("eval/results")
NEVER = 1.01  # a threshold no similarity reaches: disables that stage
LAMBDA = 3.0  # one wrong match costs as much as three correct matches
DESIGNS = {"A": Strategy.EXACT, "B": Strategy.EMBEDDING, "C": Strategy.HYBRID}
B_GRID = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
F_GRID = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]


@dataclass(frozen=True)
class EvalItem:
    name: str
    gold: frozenset[str]  # empty means "none": the food is not in the table
    sets: tuple[str, ...]


@dataclass
class Metrics:
    """Outcome counts for one design on one list of items."""

    n: int = 0
    in_table: int = 0
    correct: int = 0  # matched to a gold food
    wrong_food: int = 0  # in table, matched to another food
    missed: int = 0  # in table, left unmatched
    false_match: int = 0  # not in table, matched anyway
    correct_reject: int = 0  # not in table, left unmatched

    @property
    def matched(self) -> int:
        return self.correct + self.wrong_food + self.false_match

    @property
    def coverage(self) -> float:
        return self.matched / self.n

    @property
    def precision(self) -> float:
        return self.correct / self.matched if self.matched else math.nan

    @property
    def wrong_rate(self) -> float:
        return (self.wrong_food + self.false_match) / self.n

    @property
    def unmatched_rate(self) -> float:
        return (self.missed + self.correct_reject) / self.n

    @property
    def recall(self) -> float:
        return self.correct / self.in_table if self.in_table else math.nan


class CachedIndex:
    """Memoizes a FoodIndex: each query hits the database once."""

    def __init__(self, inner: FoodIndex) -> None:
        self._inner = inner
        self._cache: dict[tuple, list[Candidate]] = {}

    def exact(self, folded: str) -> list[Candidate]:
        key = ("exact", folded)
        if key not in self._cache:
            self._cache[key] = self._inner.exact(folded)
        return self._cache[key]

    def fuzzy(self, folded: str, k: int) -> list[Candidate]:
        key = ("fuzzy", folded, k)
        if key not in self._cache:
            self._cache[key] = self._inner.fuzzy(folded, k)
        return self._cache[key]

    def nearest(self, vector: np.ndarray, k: int) -> list[Candidate]:
        key = ("nearest", vector.tobytes(), k)
        if key not in self._cache:
            self._cache[key] = self._inner.nearest(vector, k)
        return self._cache[key]

    def details(self, food_ids: list[str]) -> dict[str, FoodDetails]:
        return self._inner.details(food_ids)


class CachedEmbedder:
    """Memoizes an Embedder per text."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self.model_name = inner.model_name
        self.signature = inner.signature
        self.dimension = inner.dimension
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for text, vector in zip(missing, self._inner.embed(missing), strict=True):
                self._cache[text] = vector
        return np.stack([self._cache[t] for t in texts])


def load_items(split: str) -> list[EvalItem]:
    with open(SPLIT_PATH, encoding="utf-8") as f:
        return [
            EvalItem(
                name=row["name"],
                gold=frozenset(g for g in row["gold"].split("|") if g != "none"),
                sets=tuple(row["sets"].split(";")),
            )
            for row in csv.DictReader(f)
            if row["split"] == split
        ]


def evaluate(matcher: FoodMatcher, items: list[EvalItem]) -> Metrics:
    metrics = Metrics()
    for item in items:
        result = matcher.match(item.name)
        metrics.n += 1
        if item.gold:
            metrics.in_table += 1
            if not result.matched:
                metrics.missed += 1
            elif result.food_id in item.gold:
                metrics.correct += 1
            else:
                metrics.wrong_food += 1
        elif result.matched:
            metrics.false_match += 1
        else:
            metrics.correct_reject += 1
    return metrics


def net(m: Metrics) -> float:
    """Selection score: a wrong match costs LAMBDA correct matches (silent
    wrong data is worse than a visible unmatched item; unmatched scores 0)."""
    return m.correct - LAMBDA * (m.wrong_food + m.false_match)


def best(candidates: list[tuple[float, Metrics]]) -> float:
    """Threshold with the highest net; ties go to the higher (stricter) one."""
    return max(candidates, key=lambda c: (net(c[1]), c[0]))[0]


def row(label: str, m: Metrics) -> dict[str, object]:
    return {
        "config": label,
        "net": net(m),
        "n": m.n,
        "coverage": round(m.coverage, 3),
        "precision": round(m.precision, 3),
        "wrong_rate": round(m.wrong_rate, 3),
        "unmatched": round(m.unmatched_rate, 3),
        "recall_in_table": round(m.recall, 3),
        **{k: v for k, v in asdict(m).items() if k not in ("n", "in_table")},
    }


def print_table(title: str, rows: list[dict[str, object]]) -> None:
    print(f"\n### {title}")
    columns = list(rows[0])
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in columns) + " |")


def build(
    fuzzy: float,
    embedding: float,
    strategy: Strategy,
    deps,
    judge: Judge | None = None,
) -> FoodMatcher:
    index, embedder = deps
    config = MatchConfig(strategy, fuzzy, embedding)
    return FoodMatcher(index, embedder, config, judge=judge)


class DiskCachedProvider:
    """Caches LLM answers on disk by prompt; rate-limits only real calls.

    Saves free-tier quota and makes judge results reproducible without keys.
    """

    def __init__(self, inner: LlmProvider, limiter: RateLimiter, path: Path) -> None:
        self._inner = inner
        self._limiter = limiter
        self._path = path
        self.name = inner.name
        self.model = inner.model
        self._cache: dict[str, str] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )

    def complete_json(self, system: str, user: str) -> str:
        key = hashlib.sha256(
            f"{self.name}|{self.model}|{system}|{user}".encode()
        ).hexdigest()
        if key not in self._cache:
            self._limiter.wait()
            self._cache[key] = self._inner.complete_json(system, user)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._cache, indent=1, ensure_ascii=False), encoding="utf-8"
            )
        return self._cache[key]


def eval_judge(provider_name: str) -> tuple[str, Judge]:
    """A judge whose answers are cached under eval/results/ (label, judge)."""
    provider, rpm = provider_from_settings(get_settings(), provider_name)
    slug = provider.model.replace("/", "_")
    path = RESULTS_DIR / f"judge_cache_{provider_name}_{slug}.json"
    cached = DiskCachedProvider(provider, RateLimiter(rpm), path)
    return f"{provider_name}:{provider.model}", FoodJudge(cached)


def selected_on_dev() -> dict[str, float]:
    """Thresholds chosen by `sweep` (report never takes numbers by hand)."""
    path = RESULTS_DIR / "dev_sweep.json"
    if not path.exists():
        raise SystemExit("Run `python -m eval.run_eval sweep` first.")
    chosen = json.loads(path.read_text(encoding="utf-8"))["selected"]
    return {k: NEVER if v == "off" else float(v) for k, v in chosen.items()}


def sweep(deps) -> dict[str, object]:
    """Tune on DEV, one variable at a time, choosing by net (never by eye)."""
    dev = load_items("dev")
    a = evaluate(build(NEVER, NEVER, Strategy.EXACT, deps), dev)
    b = [(t, evaluate(build(NEVER, t, Strategy.EMBEDDING, deps), dev)) for t in B_GRID]
    # Step 1: tune the fuzzy stage of Design C with the embedding stage off.
    c_fuzzy = [
        (f, evaluate(build(f, NEVER, Strategy.HYBRID, deps), dev)) for f in F_GRID
    ]
    best_f = best(c_fuzzy)
    # Step 2: keep that F fixed and tune the embedding stage ("off" included).
    c_emb = [
        (t, evaluate(build(best_f, t, Strategy.HYBRID, deps), dev))
        for t in [*B_GRID, NEVER]
    ]
    best_t = best(c_emb)

    def fmt(t: float) -> str:
        return "off" if t == NEVER else f"{t:.2f}"

    tables = {
        "A": [row("A exact", a)],
        "B": [row(f"B T={fmt(t)}", m) for t, m in b],
        "C_fuzzy_only": [row(f"C F={f:.2f} T=off", m) for f, m in c_fuzzy],
        "C_embedding": [row(f"C F={best_f:.2f} T={fmt(t)}", m) for t, m in c_emb],
    }
    for name, rows in tables.items():
        print_table(f"DEV sweep: {name} ({len(dev)} names, lambda={LAMBDA})", rows)
    selected = {
        "b_embedding": fmt(best(b)),
        "c_fuzzy": best_f,
        "c_embedding": fmt(best_t),
    }
    print(f"\nSelected on dev (lambda={LAMBDA}): {selected}")
    return {"lambda": LAMBDA, "selected": selected, **tables}


def judge_dev(provider_name: str, deps) -> dict[str, object]:
    """Design C + LLM judge on DEV. Nothing is tuned: F comes from the sweep."""
    chosen = selected_on_dev()
    label, judge = eval_judge(provider_name)
    dev = load_items("dev")
    rows = []
    for subset in ("all", "natural", "variants"):
        part = [i for i in dev if subset == "all" or subset in i.sets]
        matcher = build(chosen["c_fuzzy"], NEVER, Strategy.HYBRID, deps, judge)
        rows.append(row(f"C+J {subset}", evaluate(matcher, part)))
    print_table(f"DEV: Design C + judge ({label})", rows)
    return {"judge": label, "rows": rows}


def report(judges: list[str], deps) -> dict[str, object]:
    """Final numbers on dev and test with the thresholds chosen on dev."""
    chosen = selected_on_dev()
    configs: list[tuple[str, FoodMatcher]] = [
        ("A", build(NEVER, NEVER, Strategy.EXACT, deps)),
        ("B", build(NEVER, chosen["b_embedding"], Strategy.EMBEDDING, deps)),
        (
            "C",
            build(chosen["c_fuzzy"], chosen["c_embedding"], Strategy.HYBRID, deps),
        ),
    ]
    for provider_name in judges:
        label, judge = eval_judge(provider_name)
        matcher = build(chosen["c_fuzzy"], NEVER, Strategy.HYBRID, deps, judge)
        configs.append((f"C+J {label}", matcher))

    results: dict[str, object] = {"selected_on_dev": chosen}
    for split in ("dev", "test"):
        items = load_items(split)
        rows = []
        for design, matcher in configs:
            for subset in ("all", "natural", "variants"):
                part = [i for i in items if subset == "all" or subset in i.sets]
                rows.append(row(f"{design} {subset}", evaluate(matcher, part)))
        results[split] = rows
        print_table(f"{split.upper()} report", rows)
    return results


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__)
    sub = args.add_subparsers(dest="command", required=True)
    sub.add_parser("sweep", help="explore thresholds on the dev split")
    jdg = sub.add_parser("judge", help="measure Design C + LLM judge on dev")
    jdg.add_argument("--provider", choices=["groq", "gemini"], required=True)
    rep = sub.add_parser("report", help="final numbers on dev and test")
    rep.add_argument("--judge", choices=["groq", "gemini"], action="append", default=[])
    parsed = args.parse_args()

    settings = get_settings()
    embedder = CachedEmbedder(
        SentenceTransformerEmbedder(
            settings.embedding_model_name, prefix=settings.embedding_prefix
        )
    )
    with connect(settings.database_url) as conn:
        deps = (CachedIndex(PgFoodIndex(conn, embedder.signature)), embedder)
        if parsed.command == "sweep":
            results = sweep(deps)
            out = RESULTS_DIR / "dev_sweep.json"
        elif parsed.command == "judge":
            results = judge_dev(parsed.provider, deps)
            out = RESULTS_DIR / f"dev_judge_{parsed.provider}.json"
        else:
            results = report(parsed.judge, deps)
            out = RESULTS_DIR / "report.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
