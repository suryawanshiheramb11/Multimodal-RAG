"""PDF ingestion: per-page text extraction and page rendering."""
from __future__ import annotations

import logging
import math
from typing import ClassVar

import pymupdf

from ..errors import MediaProcessingError, ResourceLimitError
from ..models import EvidenceNodeDraft, MediaType, NodeType, ScannedFile
from ..security import harden_file
from ..workspace import Workspace
from .base import FileProcessor

log = logging.getLogger(__name__)


class PdfProcessor(FileProcessor):
    """Emits one page node per page, with extracted text and a rendered image.

    Both the page count and the render resolution are capped. A PDF can declare
    a page the size of a city block, and rendering it at a fixed zoom would
    allocate a pixmap large enough to take the process down — so the zoom is
    reduced per page to stay inside a fixed pixel budget.
    """

    media_type: ClassVar[MediaType] = MediaType.PDF

    def __init__(
        self,
        workspace: Workspace,
        zoom: float,
        max_pages: int,
        max_text_chars: int,
        max_pixels: int,
    ) -> None:
        self._workspace = workspace
        self._zoom = zoom
        self._max_pages = max_pages
        self._max_text_chars = max_text_chars
        self._max_pixels = max_pixels

    def process(self, source: ScannedFile) -> list[EvidenceNodeDraft]:
        out_dir = self._workspace.pages_dir(source)
        drafts: list[EvidenceNodeDraft] = []

        try:
            with pymupdf.open(source.path) as document:
                if document.needs_pass:
                    raise MediaProcessingError(
                        f"{source.file_name} is password-protected; supply the "
                        "password out of band before ingesting"
                    )
                if document.page_count > self._max_pages:
                    raise ResourceLimitError(
                        f"{source.file_name} has {document.page_count} pages, "
                        f"over the {self._max_pages} page limit"
                    )

                for page_number, page in enumerate(document, start=1):
                    drafts.append(self._process_page(page, page_number, out_dir))
        except (pymupdf.FileDataError, RuntimeError) as exc:
            raise MediaProcessingError(f"cannot read {source.file_name}: {exc}") from exc

        log.info("pdf %s: %d pages", source.file_name, len(drafts))
        return drafts

    def _process_page(self, page, page_number: int, out_dir) -> EvidenceNodeDraft:
        text = page.get_text() or ""
        truncated = len(text) > self._max_text_chars
        if truncated:
            log.warning("page %d text truncated at %d chars", page_number, self._max_text_chars)
            text = text[: self._max_text_chars]

        zoom = self._safe_zoom(page)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        image_path = out_dir / f"p{page_number:04d}.png"
        pixmap.save(str(image_path))
        harden_file(image_path)

        return EvidenceNodeDraft(
            node_type=NodeType.PAGE,
            page_number=page_number,
            text_content=text,
            file_path=str(image_path),
            metadata={
                "render_zoom": round(zoom, 3),
                "width": pixmap.width,
                "height": pixmap.height,
                "text_chars": len(text),
                "text_truncated": truncated,
            },
        )

    def _safe_zoom(self, page) -> float:
        """Clamp the render zoom so the pixmap stays inside the pixel budget."""
        rect = page.rect
        base_pixels = max(rect.width * rect.height, 1.0)
        if base_pixels * self._zoom**2 <= self._max_pixels:
            return self._zoom
        return max(math.sqrt(self._max_pixels / base_pixels), 0.1)
