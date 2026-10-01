"""Step 9: the same vector retrieval with LangChain's PGVector, side by side.

Both sides use the same model and text normalization (our embedder wrapped in
a LangChain `Embeddings`), so differences come from the abstraction itself:
storage, SQL, score semantics, determinism and latency.

LangChain creates its own tables; this script removes them when it is done.

Usage (from the project root, db running):
    pip install -r requirements-langchain.txt
    python -m eval.langchain_compare
"""

import json
import statistics
import time

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

from app.config import get_settings
from app.db import connect
from app.embeddings import SentenceTransformerEmbedder
from app.food_index import PgFoodIndex
from eval.run_eval import RESULTS_DIR, load_items

COLLECTION = "astra_foods_langchain"
K = 5


class OurEmbeddings(Embeddings):
    """Adapter: LangChain's Embeddings interface over our embedder.

    Without it, LangChain would not apply the e5 "query: " prefix or our
    casing normalization, and its vectors would not match ours.
    """

    def __init__(self, embedder: SentenceTransformerEmbedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed([text])[0].tolist()


def distinct_foods(food_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(food_ids))


def main() -> None:
    settings = get_settings()
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model_name, prefix=settings.embedding_prefix
    )
    # SQLAlchemy needs the driver in the URL; our service uses psycopg directly.
    sqlalchemy_url = settings.database_url.replace(
        "postgresql://", "postgresql+psycopg://", 1
    )
    dev = load_items("dev")

    with connect(settings.database_url) as conn:
        index = PgFoodIndex(conn, embedder.signature)
        aliases = conn.execute(
            "SELECT id, food_id, alias FROM food_aliases ORDER BY id"
        ).fetchall()

        start = time.perf_counter()
        store = PGVector(
            embeddings=OurEmbeddings(embedder),
            connection=sqlalchemy_url,
            collection_name=COLLECTION,
            pre_delete_collection=True,
        )
        store.add_texts(
            [alias for _, _, alias in aliases],
            metadatas=[{"food_id": f, "alias_id": i} for i, f, _ in aliases],
        )
        load_s = time.perf_counter() - start
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE 'langchain%' ORDER BY 1"
            ).fetchall()
        ]

        agree_top1 = recall_manual = recall_lc = 0
        score_gaps: list[float] = []
        manual_ms: list[float] = []
        lc_ms: list[float] = []
        for item in dev:
            t0 = time.perf_counter()
            manual = index.nearest(embedder.embed([item.name])[0], 4 * K)
            manual_ms.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            lc = store.similarity_search_with_score(item.name, k=4 * K)
            lc_ms.append((time.perf_counter() - t0) * 1000)

            m_foods = distinct_foods([c.food_id for c in manual])[:K]
            l_foods = distinct_foods([d.metadata["food_id"] for d, _ in lc])[:K]
            agree_top1 += m_foods[:1] == l_foods[:1]
            if item.gold:
                recall_manual += bool(set(m_foods) & item.gold)
                recall_lc += bool(set(l_foods) & item.gold)
            # LangChain's "score" is a distance: similarity = 1 - score.
            score_gaps.append(abs(manual[0].similarity - (1 - lc[0][1])))

        store.delete_collection()
        store.drop_tables()

    in_table = sum(bool(i.gold) for i in dev)
    result = {
        "dev_names": len(dev),
        "langchain_tables_created": tables,
        "load_seconds": round(load_s, 2),
        "top1_food_agreement": f"{agree_top1}/{len(dev)}",
        f"recall@{K}_manual": f"{recall_manual}/{in_table}",
        f"recall@{K}_langchain": f"{recall_lc}/{in_table}",
        "max_abs_similarity_gap": float(np.round(max(score_gaps), 6)),
        "median_ms_manual": round(statistics.median(manual_ms), 2),
        "median_ms_langchain": round(statistics.median(lc_ms), 2),
    }
    print(json.dumps(result, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "langchain_compare.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
