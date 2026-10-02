"""Load the bundled food table into a throwaway PostgreSQL with fake vectors.

For CI: the integration tests compare only exact and fuzzy matching, which
never read the vectors, so the embedding column gets one fixed unit vector of
the schema's size instead of a downloaded embedding model. It replaces the
food tables, so it refuses to run outside CI (GitHub Actions sets CI=true).

Usage (from the project root):
    CI=true python -m tests.load_test_db
"""

import os
import sys
from collections.abc import Sequence

import numpy as np

from app.config import get_settings
from scripts.ingest_foods import ingest


class UnitVectorEmbedder:
    """Every text gets the same unit vector: valid for pgvector, never compared."""

    model_name = "fake-unit-vector"
    signature = "fake-unit-vector"
    dimension = 384  # vector(384) in index/schema.sql

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


if __name__ == "__main__":
    if os.environ.get("CI") != "true":
        sys.exit("Refusing to run outside CI: it replaces the food tables.")
    counts = ingest(UnitVectorEmbedder(), get_settings().database_url)
    print("Loaded:", ", ".join(f"{v} {k}" for k, v in counts.items()))
