"""Integration: the in-memory backend must agree with PostgreSQL exactly.

Runs only with `pytest -m integration` and a loaded database
(python -m scripts.ingest_foods). Compares every eval name.
"""

import csv
from pathlib import Path

import pytest

from astra_nutrition.index.memory import MemoryFoodIndex
from astra_nutrition.matcher import FoodMatcher, MatchConfig, Strategy
from astra_nutrition.text import fold

pytestmark = pytest.mark.integration

NAMES = [
    row["name"]
    for row in csv.DictReader(open(Path("data/eval/split.csv"), encoding="utf-8"))
]


@pytest.fixture(scope="module")
def backends():
    psycopg = pytest.importorskip("psycopg")
    from app.config import get_settings
    from astra_nutrition.index.postgres import connect

    settings = get_settings()
    try:
        conn = connect(settings.database_url)
    except (RuntimeError, psycopg.OperationalError) as exc:
        pytest.skip(f"database not available: {exc}")

    from astra_nutrition.index.postgres import PgFoodIndex

    # Only exact/fuzzy are compared, so pass the stored signature as-is.
    row = conn.execute(
        "SELECT value FROM ingest_metadata WHERE key = 'embedding_signature'"
    ).fetchone()
    if row is None:
        pytest.skip("database is empty: run python -m scripts.ingest_foods")
    pg = PgFoodIndex(conn, row[0])
    yield pg, MemoryFoodIndex.from_bundled()
    conn.close()


def test_exact_and_fuzzy_top5_are_identical(backends) -> None:
    pg, memory = backends
    for name in NAMES:
        folded = fold(name)
        assert memory.exact(folded) == pg.exact(folded), name
        assert memory.fuzzy(folded, 5) == pg.fuzzy(folded, 5), name


def test_design_c_decisions_are_identical(backends) -> None:
    pg, memory = backends
    config = MatchConfig(Strategy.HYBRID, fuzzy_threshold=0.6, embedding_threshold=None)
    on_pg, in_memory = FoodMatcher(pg, None, config), FoodMatcher(memory, None, config)
    for name in NAMES:
        assert in_memory.match(name) == on_pg.match(name), name
