"""API tests with an injected analyzer (fake parser) and a fake pool.

No model, no database: the service layer is only HTTP <-> library calls.
"""

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from astra_nutrition import Analyzer
from astra_nutrition.parser import MealParser
from tests.test_parser import FakeLlm

OUTPUT = (
    '{"items": [{"name": "Yumurta", "amount": "2"}, '
    '{"name": "Baklava", "amount": "1 dilim"}]}'
)


class FakePool:
    def __init__(self, healthy: bool = True) -> None:
        self.healthy = healthy

    @contextmanager
    def connection(self, timeout: float | None = None):
        if not self.healthy:
            raise ConnectionError("db down")

        class Conn:
            def execute(self, sql: str):
                return None

        yield Conn()


class FakeLogger:
    """Records what the API asks to log; can simulate a failing database."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.failures = 0
        self.written: list[tuple] = []

    def write(self, result, latency_ms: float) -> None:
        if self.fail:
            self.failures += 1  # the real logger never raises either
            return
        self.written.append((result, latency_ms))

    def coverage_gaps(self, limit: int = 20) -> list[dict]:
        return [{"name": "mercimek çorbası", "times": 41, "closest_food": "lentils"}]


def client(healthy: bool = True, logger: FakeLogger | None = None) -> TestClient:
    analyzer = Analyzer(parser=MealParser(FakeLlm(OUTPUT)))
    app = create_app(
        analyzer=analyzer,
        pool=FakePool(healthy),
        analysis_logger=logger or FakeLogger(),
    )
    return TestClient(app)


def test_analyze_returns_items_statuses_and_totals() -> None:
    with client() as c:
        response = c.post("/analyze", json={"meal_text": "2 yumurta, 1 dilim baklava"})
    assert response.status_code == 200
    data = response.json()
    assert [i["status"] for i in data["items"]] == ["ok", "unmatched"]
    assert data["totals"]["kcal"] == 143.0
    assert data["totals"]["complete"] is False


@pytest.mark.parametrize("body", [{}, {"meal_text": ""}, {"meal_text": "x" * 1001}])
def test_analyze_validates_the_request(body: dict) -> None:
    with client() as c:
        assert c.post("/analyze", json=body).status_code == 422


def test_search_lists_candidates_with_scores() -> None:
    with client() as c:
        response = c.get("/foods/search", params={"q": "tavuk gösü", "k": 3})
    assert response.status_code == 200
    first = response.json()[0]
    assert (first["food_id"], first["method"]) == ("chicken_breast", "fuzzy")


def test_health_checks_only_the_database() -> None:
    with client() as c:
        assert c.get("/health").json()["database"] == "ok"
    with client(healthy=False) as c:
        assert c.get("/health").status_code == 503


def test_each_analysis_is_logged_after_the_response() -> None:
    logger = FakeLogger()
    with client(logger=logger) as c:
        c.post("/analyze", json={"meal_text": "2 yumurta, 1 dilim baklava"})
    result, latency_ms = logger.written[0]
    assert result.totals.kcal == 143.0
    assert latency_ms >= 0


def test_a_failing_log_never_fails_the_request_but_shows_in_health() -> None:
    logger = FakeLogger(fail=True)
    with client(logger=logger) as c:
        assert c.post("/analyze", json={"meal_text": "2 yumurta"}).status_code == 200
        assert c.get("/health").json()["log_failures"] == 1


def test_unmatched_stats_list_coverage_gaps() -> None:
    with client() as c:
        gaps = c.get("/stats/unmatched", params={"limit": 5}).json()
    assert gaps[0] == {
        "name": "mercimek çorbası",
        "times": 41,
        "closest_food": "lentils",
    }
