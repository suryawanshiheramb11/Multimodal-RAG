"""The contract every media processor implements.

A processor turns one source file into evidence node drafts. It never touches
the database, the config file, or the scanner: collaborators arrive through the
constructor, so a processor can be exercised in a test with a temp directory
and no Postgres.

Adding a new media type means writing a class here and registering it. No
existing module changes — that is the point of the registry indirection.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ..errors import UnsupportedMediaTypeError
from ..models import EvidenceNodeDraft, MediaType, ScannedFile


class FileProcessor(ABC):
    """Extracts evidence nodes from a single file of one media type."""

    media_type: ClassVar[MediaType]

    @abstractmethod
    def process(self, source: ScannedFile) -> list[EvidenceNodeDraft]:
        """Extract evidence from `source`.

        Implementations must not mutate the source file: originals are
        evidence, and their hashes are recorded before this runs.
        """


class ProcessorRegistry:
    """Dispatches a file to the processor registered for its media type."""

    def __init__(self, processors: list[FileProcessor] | None = None) -> None:
        self._processors: dict[MediaType, FileProcessor] = {}
        for processor in processors or []:
            self.register(processor)

    def register(self, processor: FileProcessor) -> None:
        self._processors[processor.media_type] = processor

    def get(self, media_type: MediaType) -> FileProcessor:
        try:
            return self._processors[media_type]
        except KeyError as exc:
            raise UnsupportedMediaTypeError(
                f"no processor registered for media type '{media_type.value}'"
            ) from exc

    def supports(self, media_type: MediaType) -> bool:
        return media_type in self._processors

    @property
    def supported_types(self) -> set[MediaType]:
        return set(self._processors)
