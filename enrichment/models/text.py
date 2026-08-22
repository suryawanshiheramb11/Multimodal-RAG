"""Sentence embeddings (MiniLM) for the 384-dimension text vector."""
from __future__ import annotations

import logging

import numpy as np

from .base import LazyModel

log = logging.getLogger(__name__)


class TextEncoder(LazyModel):
    """all-MiniLM-L6-v2 sentence embeddings, L2-normalised.

    Normalising here means cosine distance in pgvector is a plain dot product
    and every stored vector is directly comparable.
    """

    def __init__(self, model_name: str, expected_dim: int) -> None:
        super().__init__()
        self.name = f"text-encoder({model_name})"
        self._model_name = model_name
        self._expected_dim = expected_dim

    def _build(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self._model_name)
        get_dim = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)
        actual = get_dim()
        if actual != self._expected_dim:
            raise ValueError(
                f"{self._model_name} emits {actual}-d vectors but the schema "
                f"column expects {self._expected_dim}"
            )
        return model

    def embed(self, text: str) -> np.ndarray | None:
        """Embed text, or return None if the model is unavailable or text is empty."""
        model = self.load()
        if model is None or not text or not text.strip():
            return None
        vector = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vector, dtype=np.float32)
