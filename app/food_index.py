"""PostgreSQL implementation of the FoodIndex (pg_trgm + pgvector).

All lookups are exact full scans (~430 rows, well under a millisecond): no
approximate vector index, so the true nearest neighbor is always found.
Ties are broken by alias id so results are deterministic across runs.
"""

import numpy as np
import psycopg

from app.matcher import Candidate, MatchMethod


class PgFoodIndex:
    """FoodIndex backed by the food_aliases table."""

    def __init__(self, conn: psycopg.Connection, embedding_model: str) -> None:
        row = conn.execute(
            "SELECT value FROM ingest_metadata WHERE key = 'embedding_model'"
        ).fetchone()
        if row is None:
            raise RuntimeError(
                "Food table is empty: run python -m scripts.ingest_foods"
            )
        if row[0] != embedding_model:
            # Vectors from different models are not comparable: fail loudly.
            raise RuntimeError(
                f"Stored vectors come from {row[0]!r} but EMBEDDING_MODEL_NAME is "
                f"{embedding_model!r}. Re-run python -m scripts.ingest_foods."
            )
        self._conn = conn

    def exact(self, folded: str) -> list[Candidate]:
        rows = self._conn.execute(
            "SELECT food_id, alias FROM food_aliases WHERE alias_folded = %s "
            "ORDER BY id",
            (folded,),
        ).fetchall()
        return [Candidate(f, a, 1.0, MatchMethod.EXACT) for f, a in rows]

    def fuzzy(self, folded: str, k: int) -> list[Candidate]:
        rows = self._conn.execute(
            "SELECT food_id, alias, similarity(alias_folded, %s) AS sim "
            "FROM food_aliases ORDER BY sim DESC, id LIMIT %s",
            (folded, k),
        ).fetchall()
        return [Candidate(f, a, float(s), MatchMethod.FUZZY) for f, a, s in rows]

    def nearest(self, vector: np.ndarray, k: int) -> list[Candidate]:
        rows = self._conn.execute(
            "SELECT food_id, alias, 1 - (embedding <=> %s) AS sim "
            "FROM food_aliases ORDER BY embedding <=> %s, id LIMIT %s",
            (vector, vector, k),
        ).fetchall()
        return [Candidate(f, a, float(s), MatchMethod.EMBEDDING) for f, a, s in rows]
