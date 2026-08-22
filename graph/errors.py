"""Errors specific to graph construction."""
from __future__ import annotations

from ingestion.errors import IngestionError


class GraphError(IngestionError):
    """Base class for phase 3 failures."""


class EntityExtractionError(GraphError):
    """The LLM call for entity extraction failed or returned unusable output."""
