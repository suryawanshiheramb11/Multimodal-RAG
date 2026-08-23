"""Sentence embeddings (MiniLM) for the 384-dimension text vector (GPU-accelerated)."""
from __future__ import annotations

import logging

import numpy as np
import torch

from .base import LazyModel, get_device

log = logging.getLogger(__name__)


class TextEncoder(LazyModel):
    """all-MiniLM-L6-v2 sentence embeddings, L2-normalised (GPU-accelerated)."""

    def __init__(
        self,
        model_name: str,
        expected_dim: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.name = f"text-encoder({model_name})"
        self._model_name = model_name
        self._expected_dim = expected_dim
        self._device = device or get_device()

    def _build(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self._model_name)
        model.to(self._device)  # Move to GPU/MPS
        get_dim = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)
        actual = get_dim()
        if actual != self._expected_dim:
            raise ValueError(
                f"{self._model_name} emits {actual}-d vectors but the schema "
                f"column expects {self._expected_dim}"
            )
        log.info("TextEncoder on device %s", self._device)
        return model

    def embed(self, text: str) -> np.ndarray | None:
        """Embed text (GPU), or return None if the model is unavailable or text is empty."""
        model = self.load()
        if model is None or not text or not text.strip():
            return None
        # encode() respects the device the model is on
        vector = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vector, dtype=np.float32)
