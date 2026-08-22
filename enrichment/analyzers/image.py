"""Feature extraction for standalone images.

Same visual models as a video segment, minus everything temporal: no ASR, no
audio tagging, no time window.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from ..config import EnrichmentSettings
from ..registry import ModelRegistry
from .base import EnrichmentResult, NodeAnalyzer, PendingNode
from .text_fusion import build_text_embedding, fuse_text
from .visual import VisualExtractor

log = logging.getLogger(__name__)


class ImageAnalyzer(NodeAnalyzer):
    node_type: ClassVar[str] = "image"

    def __init__(self, registry: ModelRegistry, settings: EnrichmentSettings) -> None:
        self._registry = registry
        self._settings = settings
        self._visual = VisualExtractor(registry, settings)

    def analyze(self, node: PendingNode) -> EnrichmentResult:
        result = EnrichmentResult()

        # An image node points straight at the original file; it has no
        # sampled frames of its own.
        image_path = Path(node.file_path) if node.file_path else node.source_path
        visual = self._visual.extract([image_path], image_path, result)

        result.metadata.update(visual.metadata)
        result.clip_embedding = visual.clip_embedding
        result.text_content = fuse_text(caption=visual.caption, ocr=visual.ocr_text)
        build_text_embedding(self._registry, result)
        return result
