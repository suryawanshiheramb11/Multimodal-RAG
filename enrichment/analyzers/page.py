"""Feature extraction for PDF pages.

The ingestion phase already pulled the embedded text layer. The work here is
deciding whether that text can be trusted — a scanned page has a rendered
image but almost no extractable text — and embedding the result.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from ..config import EnrichmentSettings
from ..registry import ModelRegistry
from .base import EnrichmentResult, NodeAnalyzer, PendingNode
from .text_fusion import build_text_embedding, fuse_text

log = logging.getLogger(__name__)


class PageAnalyzer(NodeAnalyzer):
    node_type: ClassVar[str] = "page"

    def __init__(self, registry: ModelRegistry, settings: EnrichmentSettings) -> None:
        self._registry = registry
        self._settings = settings

    def analyze(self, node: PendingNode) -> EnrichmentResult:
        result = EnrichmentResult()
        page_image = Path(node.file_path) if node.file_path else None
        embedded_text = (node.text_content or "").strip()

        ocr_text = self._maybe_ocr(page_image, embedded_text, result)
        result.text_content = fuse_text(page_text=embedded_text or None, ocr=ocr_text)

        build_text_embedding(self._registry, result)
        self._embed_page_image(page_image, result)
        return result

    def _maybe_ocr(
        self, page_image: Path | None, embedded_text: str, result: EnrichmentResult
    ) -> str | None:
        """OCR only pages whose text layer is too thin to be real.

        Running OCR over a born-digital PDF wastes time and usually produces a
        worse transcription than the embedded text it would duplicate.
        """
        if not self._settings.enable_ocr:
            result.note_skip("ocr", "disabled in config")
            return None

        if len(embedded_text) >= self._settings.ocr_page_text_threshold:
            result.metadata["ocr_skipped"] = "page has a sufficient embedded text layer"
            return None

        if page_image is None or not page_image.is_file():
            result.note_skip("ocr", "no rendered page image available")
            return None

        reader = self._registry.ocr
        if not reader.available:
            result.note_skip("ocr", reader.unavailable_reason or "PaddleOCR unavailable")
            return None

        ocr = reader.read(page_image)
        if ocr is None:
            result.note_skip("ocr", "OCR produced no result")
            return None

        result.metadata["ocr"] = {
            "text": ocr.text,
            "line_count": ocr.line_count,
            "mean_confidence": ocr.mean_confidence,
            "reason": "embedded text layer below threshold",
        }
        return ocr.text or None

    def _embed_page_image(self, page_image: Path | None, result: EnrichmentResult) -> None:
        """A page render is still an image: embedding it lets a photograph of a
        document match the document itself."""
        if page_image is None or not page_image.is_file():
            result.note_skip("clip_embedding", "no rendered page image available")
            return

        clip = self._registry.clip
        if not clip.available:
            result.note_skip("clip_embedding", clip.unavailable_reason or "CLIP unavailable")
            return

        embedding = clip.embed_image(page_image)
        if embedding is None:
            result.note_skip("clip_embedding", "encoder produced no vector")
            return
        result.clip_embedding = embedding
