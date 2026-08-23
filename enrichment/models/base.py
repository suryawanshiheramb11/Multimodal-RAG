"""Lazy model loading with explicit availability.

Every model here can legitimately be missing at runtime: no network for a
Hugging Face download, ollama not running, a checkpoint not pulled. The phase
must degrade — record the feature as unavailable and carry on — rather than
abort a whole case. Making unavailability a first-class property (instead of a
swallowed exception) is what lets the final report say honestly which features
actually ran.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch

log = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Auto-detect best device: MPS (Metal GPU) > CUDA > CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LazyModel(ABC):
    """Loads a model on first use and caches it, remembering failures.

    A failed load is attempted once and then remembered, so a missing
    checkpoint costs one timeout per run rather than one per node.
    """

    #: Human-readable name used in logs and the availability report.
    name: str = "model"

    def __init__(self) -> None:
        self._model: Any = None
        self._attempted = False
        self._error: str | None = None

    @abstractmethod
    def _build(self) -> Any:
        """Construct the underlying model. May raise; the caller handles it."""

    def load(self) -> Any | None:
        """Return the loaded model, or None if it cannot be loaded."""
        if self._attempted:
            return self._model

        self._attempted = True
        try:
            log.info("loading %s...", self.name)
            self._model = self._build()
            log.info("loaded %s", self.name)
        except Exception as exc:  # noqa: BLE001 - any failure means "unavailable"
            self._error = f"{type(exc).__name__}: {exc}"
            log.warning("%s unavailable: %s", self.name, self._error)
            self._model = None
        return self._model

    @property
    def available(self) -> bool:
        return self.load() is not None

    @property
    def unavailable_reason(self) -> str | None:
        self.load()
        return self._error
