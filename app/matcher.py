"""Food matching: map a parsed item name to a food in the nutrition table.

Three strategies share one interface so they can be evaluated on the same
labeled data (Step 7):
- Design A (EXACT):     folded name equals a folded alias;
- Design B (EMBEDDING): nearest alias by cosine similarity >= threshold;
- Design C (HYBRID):    exact -> fuzzy (pg_trgm) -> embedding, each gated.

Below the thresholds an item is UNMATCHED. We never fall back to the nearest
neighbor: a wrong food is silent wrong data, an unmatched item is visible.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from app.embeddings import Embedder
from app.text import fold


class Strategy(StrEnum):
    """Matching strategy (the design labels used in the evaluation)."""

    EXACT = "exact"  # Design A
    EMBEDDING = "embedding"  # Design B
    HYBRID = "hybrid"  # Design C


class MatchMethod(StrEnum):
    """Which stage produced a candidate."""

    EXACT = "exact"
    FUZZY = "fuzzy"
    EMBEDDING = "embedding"


@dataclass(frozen=True)
class Candidate:
    """One possible food for a query, with its score from one stage."""

    food_id: str
    alias: str
    similarity: float
    method: MatchMethod


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching one item name."""

    query: str
    matched: bool
    food_id: str | None = None
    alias: str | None = None
    similarity: float | None = None
    method: MatchMethod | None = None
    # When unmatched: the closest candidate we refused (logged to reveal
    # coverage gaps). Never used for nutrition.
    best_candidate: Candidate | None = None


class FoodIndex(Protocol):
    """Lookups over food aliases (Postgres in production, a fake in tests)."""

    def exact(self, folded: str) -> list[Candidate]: ...

    def fuzzy(self, folded: str, k: int) -> list[Candidate]: ...

    def nearest(self, vector: np.ndarray, k: int) -> list[Candidate]: ...


@dataclass(frozen=True)
class MatchConfig:
    """Strategy and thresholds. Thresholds must be tuned on the dev set only."""

    strategy: Strategy = Strategy.HYBRID
    fuzzy_threshold: float = 0.5
    embedding_threshold: float = 0.9


class FoodMatcher:
    """Matches item names to foods with one of the three strategies."""

    def __init__(
        self, index: FoodIndex, embedder: Embedder, config: MatchConfig
    ) -> None:
        self._index = index
        self._embedder = embedder
        self.config = config

    def match(self, name: str) -> MatchResult:
        """Return the matched food, or an explicit unmatched result."""
        folded = fold(name)
        if not folded:
            return MatchResult(query=name, matched=False)
        strategy = self.config.strategy

        if strategy in (Strategy.EXACT, Strategy.HYBRID):
            hits = self._index.exact(folded)
            if len({hit.food_id for hit in hits}) == 1:
                return _matched(name, hits[0])
            if hits:  # one name pointing to two foods: refuse, never guess
                return MatchResult(query=name, matched=False, best_candidate=hits[0])
            if strategy == Strategy.EXACT:
                return MatchResult(query=name, matched=False)

        best: Candidate | None = None
        if strategy == Strategy.HYBRID:
            fuzzy = self._index.fuzzy(folded, k=1)
            if fuzzy:
                best = fuzzy[0]
                if best.similarity >= self.config.fuzzy_threshold:
                    return _matched(name, best)

        nearest = self._index.nearest(self._embedder.embed([name])[0], k=1)
        if nearest:
            best = nearest[0]
            if best.similarity >= self.config.embedding_threshold:
                return _matched(name, best)
        return MatchResult(query=name, matched=False, best_candidate=best)

    def candidates(self, name: str, k: int = 5) -> list[Candidate]:
        """Top-k candidates from every stage, for inspection and debugging."""
        folded = fold(name)
        if not folded:
            return []
        vector = self._embedder.embed([name])[0]
        found: Sequence[Candidate] = [
            *self._index.exact(folded),
            *self._index.fuzzy(folded, k),
            *self._index.nearest(vector, k),
        ]
        return list(found)


def _matched(query: str, candidate: Candidate) -> MatchResult:
    return MatchResult(
        query=query,
        matched=True,
        food_id=candidate.food_id,
        alias=candidate.alias,
        similarity=candidate.similarity,
        method=candidate.method,
    )
