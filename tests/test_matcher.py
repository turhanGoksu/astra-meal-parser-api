"""Unit tests for the matching strategies (fake index, no DB or model)."""

from collections.abc import Sequence

import numpy as np

from app.matcher import (
    Candidate,
    FoodMatcher,
    MatchConfig,
    MatchMethod,
    Strategy,
)


class FakeEmbedder:
    """Encodes each text as its position in a list, so the fake index can
    look up which text a vector came from."""

    model_name = "fake"
    dimension = 1

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            self.texts.append(text)
            vectors.append([len(self.texts) - 1])
        return np.array(vectors, dtype=np.float32)


class FakeIndex:
    """Preset candidates per query, in the shape PgFoodIndex returns."""

    def __init__(
        self,
        embedder: FakeEmbedder,
        exact: dict[str, list[tuple[str, str]]] | None = None,
        fuzzy: dict[str, list[tuple[str, str, float]]] | None = None,
        nearest: dict[str, list[tuple[str, str, float]]] | None = None,
    ) -> None:
        self._embedder = embedder
        self._exact = exact or {}
        self._fuzzy = fuzzy or {}
        self._nearest = nearest or {}

    def exact(self, folded: str) -> list[Candidate]:
        return [
            Candidate(f, a, 1.0, MatchMethod.EXACT)
            for f, a in self._exact.get(folded, [])
        ]

    def fuzzy(self, folded: str, k: int) -> list[Candidate]:
        return [
            Candidate(f, a, s, MatchMethod.FUZZY)
            for f, a, s in self._fuzzy.get(folded, [])[:k]
        ]

    def nearest(self, vector: np.ndarray, k: int) -> list[Candidate]:
        text = self._embedder.texts[int(vector[0])]
        return [
            Candidate(f, a, s, MatchMethod.EMBEDDING)
            for f, a, s in self._nearest.get(text, [])[:k]
        ]


def make_matcher(strategy: Strategy, **index_data: dict) -> FoodMatcher:
    embedder = FakeEmbedder()
    index = FakeIndex(embedder, **index_data)
    config = MatchConfig(strategy, fuzzy_threshold=0.5, embedding_threshold=0.93)
    return FoodMatcher(index, embedder, config)


# Real scores measured on our table (see Step 6 discussion).
DATA = {
    "exact": {"tavuk gogsu": [("chicken_breast", "Tavuk göğsü")]},
    "fuzzy": {
        "tavuk gosu": [("chicken_breast", "tavuk gogsu", 0.64)],
        "mercimek corbasi": [("lentils", "mercimek", 0.53)],
        "baklava": [("honey", "bal", 0.20)],
    },
    "nearest": {
        "tavuk gösü": [("chicken_breast", "Tavuk göğsü", 0.80)],
        "mercimek çorbası": [("tomato_soup", "Domates çorbası", 0.857)],
        "baklava": [("couscous", "Kuskus", 0.929)],
        "chicken breast fillet": [("chicken_breast", "Chicken breast", 0.95)],
    },
}


def test_design_c_exact_stage_wins_first() -> None:
    result = make_matcher(Strategy.HYBRID, **DATA).match("Tavuk Göğsü")
    assert (result.matched, result.food_id, result.method) == (
        True,
        "chicken_breast",
        MatchMethod.EXACT,
    )


def test_design_c_fuzzy_catches_typos() -> None:
    result = make_matcher(Strategy.HYBRID, **DATA).match("tavuk gösü")
    assert (result.food_id, result.method, result.similarity) == (
        "chicken_breast",
        MatchMethod.FUZZY,
        0.64,
    )


def test_design_c_fuzzy_can_produce_a_wrong_food() -> None:
    # Known risk: "mercimek çorbası" is not in the table but shares a word
    # with "mercimek" (lentils). The evaluation must measure this.
    result = make_matcher(Strategy.HYBRID, **DATA).match("mercimek çorbası")
    assert (result.food_id, result.method) == ("lentils", MatchMethod.FUZZY)


def test_design_c_below_both_thresholds_is_unmatched_with_best_candidate() -> None:
    result = make_matcher(Strategy.HYBRID, **DATA).match("baklava")
    assert result.matched is False
    assert result.food_id is None
    assert result.best_candidate is not None
    assert result.best_candidate.food_id == "couscous"  # logged, never used


def test_design_c_embedding_stage_matches_above_threshold() -> None:
    result = make_matcher(Strategy.HYBRID, **DATA).match("chicken breast fillet")
    assert (result.food_id, result.method) == ("chicken_breast", MatchMethod.EMBEDDING)


def test_design_a_never_uses_fuzzy_or_embedding() -> None:
    matcher = make_matcher(Strategy.EXACT, **DATA)
    assert matcher.match("Tavuk Göğsü").matched is True
    assert matcher.match("tavuk gösü").matched is False
    assert matcher.match("tavuk gösü").best_candidate is None


def test_design_b_skips_exact_and_fuzzy() -> None:
    matcher = make_matcher(Strategy.EMBEDDING, **DATA)
    result = matcher.match("tavuk gösü")  # fuzzy would match; B ignores it
    assert result.matched is False
    assert result.best_candidate is not None
    assert result.best_candidate.method == MatchMethod.EMBEDDING


def test_one_name_pointing_to_two_foods_is_refused() -> None:
    data = {"exact": {"peynir": [("feta", "peynir"), ("mozzarella", "peynir")]}}
    result = make_matcher(Strategy.HYBRID, **data).match("peynir")
    assert result.matched is False


def test_blank_name_is_unmatched() -> None:
    assert make_matcher(Strategy.HYBRID, **DATA).match("   ").matched is False
