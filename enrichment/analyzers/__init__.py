"""Node analyzers and the registry that dispatches to them."""
from __future__ import annotations

from ..config import EnrichmentSettings
from ..registry import ModelRegistry
from .audio_track import AudioTrackAnalyzer
from .base import AnalyzerRegistry, EnrichmentResult, NodeAnalyzer, PendingNode
from .image import ImageAnalyzer
from .page import PageAnalyzer
from .video_segment import VideoSegmentAnalyzer

__all__ = [
    "AnalyzerRegistry",
    "AudioTrackAnalyzer",
    "EnrichmentResult",
    "ImageAnalyzer",
    "NodeAnalyzer",
    "PageAnalyzer",
    "PendingNode",
    "VideoSegmentAnalyzer",
    "build_analyzer_registry",
]


def build_analyzer_registry(
    registry: ModelRegistry, settings: EnrichmentSettings
) -> AnalyzerRegistry:
    """Composition root for the enrichment phase.

    The only place that names concrete analyzers; a new node type is added
    here and nowhere else.
    """
    return AnalyzerRegistry(
        [
            VideoSegmentAnalyzer(registry, settings),
            AudioTrackAnalyzer(registry, settings),
            ImageAnalyzer(registry, settings),
            PageAnalyzer(registry, settings),
        ]
    )
