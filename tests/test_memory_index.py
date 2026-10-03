"""The in-memory index over the bundled food table (no database)."""

import pytest

from astra_nutrition.index.memory import MemoryFoodIndex
from astra_nutrition.matcher import FoodMatcher, MatchConfig, MatchMethod, Strategy
from astra_nutrition.trigram import similarity

INDEX = MemoryFoodIndex.from_bundled()


def test_exact_lookup_on_folded_aliases() -> None:
    hits = INDEX.exact("tavuk gogsu")
    assert {hit.food_id for hit in hits} == {"chicken_breast"}
    assert all(hit.method == MatchMethod.EXACT for hit in hits)


def test_fuzzy_uses_pg_trgm_similarity() -> None:
    top = INDEX.fuzzy("tavuk gosu", k=1)[0]
    assert top.food_id == "chicken_breast"
    assert top.similarity == similarity("tavuk gosu", "tavuk gogsu")


def test_fuzzy_results_are_sorted_and_limited() -> None:
    results = INDEX.fuzzy("domatez", k=3)
    assert len(results) == 3
    assert [r.similarity for r in results] == sorted(
        (r.similarity for r in results), reverse=True
    )


def test_nearest_without_embedder_explains_the_extra() -> None:
    with pytest.raises(RuntimeError, match=r"astra-nutrition\[judge\]"):
        INDEX.nearest(None, k=1)  # type: ignore[arg-type]


def test_details_for_known_foods() -> None:
    details = INDEX.details(["banana", "unknown"])
    assert list(details) == ["banana"]
    assert details["banana"].name_tr == "Muz"


def test_default_library_matcher_needs_no_embedder() -> None:
    config = MatchConfig(Strategy.HYBRID, fuzzy_threshold=0.6, embedding_threshold=None)
    matcher = FoodMatcher(INDEX, None, config)
    assert matcher.match("Tavuk Göğsü").method == MatchMethod.EXACT
    assert matcher.match("tavuk gösü").method == MatchMethod.FUZZY
    assert matcher.match("kokoreç").matched is False


def test_embedding_stage_without_embedder_fails_fast() -> None:
    config = MatchConfig(Strategy.HYBRID, fuzzy_threshold=0.6, embedding_threshold=0.9)
    with pytest.raises(ValueError, match="needs an embedder"):
        FoodMatcher(INDEX, None, config)
