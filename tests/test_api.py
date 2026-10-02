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
    def connection(self):
        if not self.healthy:
            raise ConnectionError("db down")

        class Conn:
            def execute(self, sql: str):
                return None

        yield Conn()


def client(healthy: bool = True) -> TestClient:
    analyzer = Analyzer(parser=MealParser(FakeLlm(OUTPUT)))
    return TestClient(create_app(analyzer=analyzer, pool=FakePool(healthy)))


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
