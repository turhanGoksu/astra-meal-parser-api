"""Embedding experiments on the DEV split, one variable at a time.

- E0 baseline: MiniLM (Model 1), lowercased text
- E1 casing:   MiniLM, original casing           (only casing changes)
- E2 model:    multilingual-e5-small (Model 2), lowercased, "query: " prefix
               on both sides as its model card advises for symmetric tasks
               (only the model changes)

Alias vectors are computed in memory, so the production database is not
modified; exact and fuzzy lookups still come from PostgreSQL.

Usage (from the project root, db running):
    python -m eval.experiments
"""

import json

import numpy as np

from app.config import get_settings
from app.db import connect
from app.embeddings import Embedder, SentenceTransformerEmbedder
from app.food_index import PgFoodIndex
from app.matcher import Candidate, MatchMethod, Strategy
from eval.run_eval import (
    B_GRID,
    LAMBDA,
    NEVER,
    RESULTS_DIR,
    CachedEmbedder,
    CachedIndex,
    best,
    build,
    evaluate,
    load_items,
    net,
)

FUZZY = 0.60  # tuned on dev in the main sweep; fixed here
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
E5_SMALL = "intfloat/multilingual-e5-small"
EXPERIMENTS = {  # label: (model, prefix, lowercase)
    "E0 MiniLM lowercase": (MINILM, "", True),
    "E1 MiniLM original casing": (MINILM, "", False),
    "E2 e5-small lowercase": (E5_SMALL, "query: ", True),
}


class InMemoryIndex:
    """Exact/fuzzy from PostgreSQL, nearest from in-memory alias vectors."""

    def __init__(self, db_index: PgFoodIndex, conn, embedder: Embedder) -> None:
        rows = conn.execute(
            "SELECT food_id, alias FROM food_aliases ORDER BY id"
        ).fetchall()
        self._db = db_index
        self._foods = [food for food, _ in rows]
        self._aliases = [alias for _, alias in rows]
        self._vectors = embedder.embed(self._aliases)

    def exact(self, folded: str) -> list[Candidate]:
        return self._db.exact(folded)

    def fuzzy(self, folded: str, k: int) -> list[Candidate]:
        return self._db.fuzzy(folded, k)

    def nearest(self, vector: np.ndarray, k: int) -> list[Candidate]:
        sims = self._vectors @ vector
        order = np.lexsort((np.arange(len(sims)), -sims))[:k]  # ties: alias id
        return [
            Candidate(
                self._foods[i], self._aliases[i], float(sims[i]), MatchMethod.EMBEDDING
            )
            for i in order
        ]


def main() -> None:
    settings = get_settings()
    dev = load_items("dev")
    summary = {}
    with connect(settings.database_url) as conn:
        db_index = PgFoodIndex(conn, settings.embedding_model_name)
        for label, (model, prefix, lowercase) in EXPERIMENTS.items():
            embedder = CachedEmbedder(
                SentenceTransformerEmbedder(model, prefix=prefix, lowercase=lowercase)
            )
            deps = (CachedIndex(InMemoryIndex(db_index, conn, embedder)), embedder)
            b = [
                (t, evaluate(build(NEVER, t, Strategy.EMBEDDING, deps), dev))
                for t in B_GRID
            ]
            c = [
                (t, evaluate(build(FUZZY, t, Strategy.HYBRID, deps), dev))
                for t in [*B_GRID, NEVER]
            ]
            bt, ct = best(b), best(c)
            bm, cm = dict(b)[bt], dict(c)[ct]
            summary[label] = {
                "B_best_T": bt,
                "B_net": net(bm),
                "B_correct": bm.correct,
                "B_wrong": bm.wrong_food + bm.false_match,
                "C_best_T": "off" if ct == NEVER else ct,
                "C_net": net(cm),
                "C_correct": cm.correct,
                "C_wrong": cm.wrong_food + cm.false_match,
            }
            print(label, summary[label], flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "dev_experiments.json"
    out.write_text(
        json.dumps({"lambda": LAMBDA, "fuzzy": FUZZY, **summary}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
