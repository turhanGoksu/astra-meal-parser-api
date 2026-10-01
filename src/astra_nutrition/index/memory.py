"""In-memory FoodIndex: the library default, no database needed.

Same interface and same results as the PostgreSQL backend:
- exact:   dict lookup on folded aliases;
- fuzzy:   pure-Python pg_trgm similarity (astra_nutrition.trigram);
- nearest: numpy dot products over normalized alias vectors (needs an
           embedder, i.e. the [judge] extra).
Aliases are kept in the same order as the database rows, so ties are broken
the same way (by alias position) in both backends.
"""

from collections import defaultdict

import numpy as np

from astra_nutrition.embeddings import Embedder
from astra_nutrition.matcher import Candidate, FoodDetails, MatchMethod
from astra_nutrition.tables import alias_rows, read_table
from astra_nutrition.trigram import similarity_of, trigrams


class MemoryFoodIndex:
    """FoodIndex over a list of food rows (foods.csv format)."""

    def __init__(
        self, foods: list[dict[str, str]], embedder: Embedder | None = None
    ) -> None:
        rows = alias_rows(foods)
        self._food_ids = [food_id for food_id, _, _, _ in rows]
        self._aliases = [alias for _, alias, _, _ in rows]
        self._trigrams = [trigrams(folded) for _, _, folded, _ in rows]
        self._by_folded: dict[str, list[int]] = defaultdict(list)
        for position, (_, _, folded, _) in enumerate(rows):
            self._by_folded[folded].append(position)
        self._details = {
            food["id"]: FoodDetails(
                food["id"], food["name_en"], food["name_tr"], float(food["kcal_100g"])
            )
            for food in foods
        }
        self._embedder = embedder
        self._vectors = embedder.embed(self._aliases) if embedder else None

    @classmethod
    def from_bundled(cls, embedder: Embedder | None = None) -> "MemoryFoodIndex":
        """Index over the food table shipped with the package."""
        return cls(read_table("foods.csv"), embedder)

    def exact(self, folded: str) -> list[Candidate]:
        return [
            Candidate(self._food_ids[i], self._aliases[i], 1.0, MatchMethod.EXACT)
            for i in self._by_folded.get(folded, [])
        ]

    def fuzzy(self, folded: str, k: int) -> list[Candidate]:
        query = trigrams(folded)
        scores = [similarity_of(query, alias) for alias in self._trigrams]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [
            Candidate(self._food_ids[i], self._aliases[i], scores[i], MatchMethod.FUZZY)
            for i in order
        ]

    def nearest(self, vector: np.ndarray, k: int) -> list[Candidate]:
        if self._vectors is None:
            raise RuntimeError(
                "nearest() needs an embedder: pip install 'astra-nutrition[judge]' "
                "and pass embedder=... to MemoryFoodIndex"
            )
        sims = self._vectors @ vector
        order = np.lexsort((np.arange(len(sims)), -sims))[:k]  # ties: position
        return [
            Candidate(
                self._food_ids[i],
                self._aliases[i],
                float(sims[i]),
                MatchMethod.EMBEDDING,
            )
            for i in order
        ]

    def details(self, food_ids: list[str]) -> dict[str, FoodDetails]:
        return {f: self._details[f] for f in food_ids if f in self._details}
