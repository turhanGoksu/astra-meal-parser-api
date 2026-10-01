"""PostgreSQL backend (optional extra: pip install astra-nutrition[postgres]).

FoodIndex implementation on pg_trgm + pgvector, plus connection helpers.

All lookups are exact full scans (~450 rows, well under a millisecond): no
approximate vector index, so the true nearest neighbor is always found.
Ties are broken by alias id so results are deterministic across runs.
"""

from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from astra_nutrition.matcher import Candidate, FoodDetails, MatchMethod

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(database_url: str | None) -> psycopg.Connection:
    """Open a connection with the pgvector type registered."""
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set (see .env.example)")
    conn = psycopg.connect(database_url)
    # The vector type must exist before it can be registered.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def apply_schema(conn: psycopg.Connection) -> None:
    """Create extensions and tables if they do not exist."""
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


class PgFoodIndex:
    """FoodIndex backed by the foods and food_aliases tables."""

    def __init__(self, conn: psycopg.Connection, embedding_signature: str) -> None:
        row = conn.execute(
            "SELECT value FROM ingest_metadata WHERE key = 'embedding_signature'"
        ).fetchone()
        if row is None:
            raise RuntimeError(
                "Food table is empty or outdated: run python -m scripts.ingest_foods"
            )
        if row[0] != embedding_signature:
            # Vectors from another model, prefix or casing are not comparable.
            raise RuntimeError(
                f"Stored vectors were made with {row[0]!r} but the configured "
                f"embedder is {embedding_signature!r}. "
                "Re-run python -m scripts.ingest_foods."
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
        # pg_trgm returns float4, which psycopg parses from its text form;
        # re-round to float32 so both backends return bit-identical scores.
        return [
            Candidate(f, a, float(np.float32(s)), MatchMethod.FUZZY) for f, a, s in rows
        ]

    def nearest(self, vector: np.ndarray, k: int) -> list[Candidate]:
        rows = self._conn.execute(
            "SELECT food_id, alias, 1 - (embedding <=> %s) AS sim "
            "FROM food_aliases ORDER BY embedding <=> %s, id LIMIT %s",
            (vector, vector, k),
        ).fetchall()
        return [Candidate(f, a, float(s), MatchMethod.EMBEDDING) for f, a, s in rows]

    def details(self, food_ids: list[str]) -> dict[str, FoodDetails]:
        rows = self._conn.execute(
            "SELECT id, name_en, name_tr, kcal_100g FROM foods WHERE id = ANY(%s)",
            (food_ids,),
        ).fetchall()
        return {r[0]: FoodDetails(r[0], r[1], r[2], float(r[3])) for r in rows}
