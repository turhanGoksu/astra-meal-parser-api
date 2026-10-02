"""Writes every analysis and its items to PostgreSQL (best effort).

Policy (Design B): the user always gets their result. If the log cannot be
written, the failure is loud for the operator instead: a warning in the
service log and a counter shown by /health. Writes run after the response is
sent and use a short connection timeout, so a database outage never delays
users.
"""

import logging
import threading
from pathlib import Path

from astra_nutrition import AnalysisResult, __version__

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("log_schema.sql")

INSERT_ANALYSIS = """
INSERT INTO analyses (meal_text, parse_status, parse_error, latency_ms, items,
    counted, estimated, amount_unknown, unmatched, complete, includes_estimates,
    kcal, protein_g, carbs_g, fat_g, app_version, judge)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING id
"""

INSERT_ITEM = """
INSERT INTO analysis_items (analysis_id, position, name, amount, status, food_id,
    match_method, match_similarity, best_candidate_food_id, grams, amount_status,
    kcal, protein_g, carbs_g, fat_g, note)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

COVERAGE_GAPS = """
SELECT lower(name) AS name,
       count(*) AS times,
       mode() WITHIN GROUP (ORDER BY best_candidate_food_id) AS closest_food
FROM analysis_items
WHERE status = 'unmatched'
GROUP BY 1
ORDER BY times DESC, name
LIMIT %s
"""


class AnalysisLogger:
    """Best-effort writer: never raises, counts its failures."""

    def __init__(
        self,
        pool,
        log_meal_text: bool = True,
        judge: str | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        self._pool = pool
        self._log_meal_text = log_meal_text
        self._judge = judge
        self._timeout = timeout_seconds
        self._schema_ready = False
        self._lock = threading.Lock()
        self.failures = 0

    def write(self, result: AnalysisResult, latency_ms: float) -> None:
        """Store one analysis and its items in a single transaction."""
        try:
            with self._pool.connection(timeout=self._timeout) as conn:
                if not self._schema_ready:
                    # Committed on its own: a failing insert below must not
                    # roll the tables back while the flag says they exist.
                    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
                    conn.commit()
                    self._schema_ready = True
                with conn.transaction():
                    analysis_id = conn.execute(
                        INSERT_ANALYSIS, self._analysis_row(result, latency_ms)
                    ).fetchone()[0]
                    with conn.cursor() as cur:
                        cur.executemany(
                            INSERT_ITEM,
                            [
                                _item_row(analysis_id, position, item)
                                for position, item in enumerate(result.items)
                            ],
                        )
        except Exception:
            logger.warning("Could not write the analysis log", exc_info=True)
            with self._lock:
                self.failures += 1

    def coverage_gaps(self, limit: int = 20) -> list[dict[str, object]]:
        """Most frequent unmatched names: the next foods to add to the table."""
        with self._pool.connection(timeout=self._timeout) as conn:
            rows = conn.execute(COVERAGE_GAPS, (limit,)).fetchall()
        return [{"name": n, "times": t, "closest_food": c} for n, t, c in rows]

    def _analysis_row(self, result: AnalysisResult, latency_ms: float) -> tuple:
        t = result.totals
        return (
            result.meal_text if self._log_meal_text else None,
            result.parse_status,
            result.parse_error,
            latency_ms,
            t.items,
            t.counted,
            t.estimated,
            t.amount_unknown,
            t.unmatched,
            t.complete,
            t.includes_estimates,
            t.kcal,
            t.protein_g,
            t.carbs_g,
            t.fat_g,
            __version__,
            self._judge,
        )


def _item_row(analysis_id: int, position: int, item) -> tuple:
    n = item.nutrition
    return (
        analysis_id,
        position,
        item.name,
        item.amount,
        item.status,
        item.food_id,
        item.match_method,
        item.match_similarity,
        item.best_candidate_food_id,
        item.grams,
        item.amount_status,
        n.kcal if n else None,
        n.protein_g if n else None,
        n.carbs_g if n else None,
        n.fat_g if n else None,
        item.note,
    )
