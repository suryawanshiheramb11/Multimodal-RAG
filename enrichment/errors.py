"""Errors specific to the enrichment phase."""
from __future__ import annotations

from ingestion.errors import IngestionError


class EnrichmentError(IngestionError):
    """Base class for feature-extraction failures."""


class AnalyzerNotRegisteredError(EnrichmentError):
    """No analyzer exists for a node type encountered in the database."""


class EmbeddingDimensionError(EnrichmentError):
    """A model produced a vector whose width does not match its column.

    Fatal rather than tolerated: a wrong-width vector either fails the insert
    or, worse, silently corrupts every similarity query built on it.
    """
