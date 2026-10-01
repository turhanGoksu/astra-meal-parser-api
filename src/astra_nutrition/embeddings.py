"""Sentence embeddings for food names (cross-lingual TR <-> EN)."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from astra_nutrition.text import embedding_text


class Embedder(Protocol):
    """Anything that turns texts into L2-normalized vectors.

    Tests use a deterministic fake, so CI never downloads a model.
    """

    model_name: str
    signature: str  # model + text normalization; stored at ingest time
    dimension: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def embedding_signature(model_name: str, prefix: str, lowercase: bool) -> str:
    """Identify everything that changes the vectors, not only the model.

    Vectors made with another prefix or casing are as incomparable as
    vectors from another model, so all three are checked before searching.
    """
    return f"{model_name}|prefix={prefix!r}|lowercase={lowercase}"


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers model on CPU.

    ``prefix`` is prepended to every text (e5 models expect "query: ");
    ``lowercase`` controls casing normalization (see app.text.embedding_text).
    """

    def __init__(
        self, model_name: str, prefix: str = "", lowercase: bool = True
    ) -> None:
        # Imported here: torch is heavy and not needed by most unit tests.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        self._prefix = prefix
        self._lowercase = lowercase
        self.model_name = model_name
        self.signature = embedding_signature(model_name, prefix, lowercase)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one normalized float32 vector per text (cosine = dot product)."""
        return self._model.encode(
            [self._prefix + embedding_text(t, self._lowercase) for t in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
