"""Sentence embeddings for food names (cross-lingual TR <-> EN)."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from app.text import embedding_text


class Embedder(Protocol):
    """Anything that turns texts into L2-normalized vectors.

    Tests use a deterministic fake, so CI never downloads a model.
    """

    model_name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers model on CPU."""

    def __init__(self, model_name: str) -> None:
        # Imported here: torch is heavy and not needed by most unit tests.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        self.model_name = model_name
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one normalized float32 vector per text (cosine = dot product)."""
        return self._model.encode(
            [embedding_text(text) for text in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
