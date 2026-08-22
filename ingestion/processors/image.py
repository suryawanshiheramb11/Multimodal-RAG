"""Image ingestion: verify the file decodes and record its dimensions."""
from __future__ import annotations

import logging
from typing import ClassVar

from PIL import Image, UnidentifiedImageError

from ..errors import MediaProcessingError, ResourceLimitError
from ..models import EvidenceNodeDraft, MediaType, NodeType, ScannedFile
from .base import FileProcessor

log = logging.getLogger(__name__)


class ImageProcessor(FileProcessor):
    """Emits a single image node pointing at the original file.

    Images are not copied: the original is already the artefact, and leaving it
    in place keeps one authoritative copy for the chain of custody.
    """

    media_type: ClassVar[MediaType] = MediaType.IMAGE

    def __init__(self, max_pixels: int) -> None:
        self._max_pixels = max_pixels

    def process(self, source: ScannedFile) -> list[EvidenceNodeDraft]:
        # Pillow's own decompression-bomb guard, set to our configured budget.
        # It raises rather than warns above twice the limit, which is what we
        # want for a file we did not create.
        previous_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = self._max_pixels
        try:
            with Image.open(source.path) as image:
                pixels = image.width * image.height
                if pixels > self._max_pixels:
                    raise ResourceLimitError(
                        f"{source.file_name} is {image.width}x{image.height} "
                        f"({pixels} px), over the {self._max_pixels} px limit"
                    )
                metadata = {
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "format": image.format,
                }
        except Image.DecompressionBombError as exc:
            raise ResourceLimitError(f"{source.file_name}: {exc}") from exc
        except (UnidentifiedImageError, OSError) as exc:
            raise MediaProcessingError(f"cannot decode {source.file_name}: {exc}") from exc
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit

        log.info("image %s: %dx%d %s",
                 source.file_name, metadata["width"], metadata["height"], metadata["mode"])

        return [
            EvidenceNodeDraft(
                node_type=NodeType.IMAGE,
                file_path=str(source.path),
                metadata=metadata,
            )
        ]
