"""AnalysisLogger: one transaction per analysis, never raises, counts failures."""

from contextlib import contextmanager

from app.analysis_log import INSERT_ANALYSIS, INSERT_ITEM, AnalysisLogger
from astra_nutrition import Analyzer

RESULT = Analyzer().analyze_items(
    [("Yumurta", "2"), ("Baklava", "1 dilim")], meal_text="2 yumurta, 1 dilim baklava"
)


class RecordingConn:
    def __init__(self, log: list) -> None:
        self.log = log

    def execute(self, sql: str, params: tuple = ()):
        self.log.append(("execute", sql, params))

        class Result:
            def fetchone(self):
                return (7,)

        return Result()

    def commit(self) -> None:
        self.log.append(("commit",))

    @contextmanager
    def transaction(self):
        self.log.append(("begin",))
        yield
        self.log.append(("end",))

    @contextmanager
    def cursor(self):
        conn = self

        class Cursor:
            def executemany(self, sql: str, rows: list) -> None:
                conn.log.append(("executemany", sql, rows))

        yield Cursor()


class RecordingPool:
    def __init__(self, down: bool = False) -> None:
        self.down = down
        self.log: list = []

    @contextmanager
    def connection(self, timeout: float | None = None):
        if self.down:
            raise TimeoutError("database down")
        yield RecordingConn(self.log)


def test_analysis_and_items_are_written_in_one_transaction() -> None:
    pool = RecordingPool()
    AnalysisLogger(pool, judge="groq:m").write(RESULT, 12.5)
    kinds = [entry[0] for entry in pool.log]
    assert kinds == ["execute", "commit", "begin", "execute", "executemany", "end"]

    _, sql, analysis = pool.log[3]
    assert sql == INSERT_ANALYSIS
    assert analysis[0] == "2 yumurta, 1 dilim baklava"
    assert analysis[3] == 12.5
    assert analysis[-1] == "groq:m"

    _, sql, items = pool.log[4]
    assert sql == INSERT_ITEM
    assert [(row[0], row[1], row[4]) for row in items] == [
        (7, 0, "ok"),
        (7, 1, "unmatched"),
    ]
    assert items[1][8] is not None  # best candidate of the unmatched item


def test_schema_is_applied_only_once() -> None:
    pool = RecordingPool()
    logger = AnalysisLogger(pool)
    logger.write(RESULT, 1.0)
    logger.write(RESULT, 1.0)
    assert sum(entry[0] == "commit" for entry in pool.log) == 1


def test_meal_text_can_be_left_out_for_privacy() -> None:
    pool = RecordingPool()
    AnalysisLogger(pool, log_meal_text=False).write(RESULT, 1.0)
    assert pool.log[3][2][0] is None


def test_database_outage_is_counted_not_raised() -> None:
    logger = AnalysisLogger(RecordingPool(down=True))
    logger.write(RESULT, 1.0)  # must not raise
    logger.write(RESULT, 1.0)
    assert logger.failures == 2
