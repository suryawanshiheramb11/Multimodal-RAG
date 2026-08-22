"""Standalone audio ingestion: normalise to the ASR input format."""
from __future__ import annotations

import logging
from typing import ClassVar

from ..media import AudioExtractor
from ..models import EvidenceNodeDraft, MediaType, NodeType, ScannedFile
from ..workspace import Workspace
from .base import FileProcessor

log = logging.getLogger(__name__)


class AudioProcessor(FileProcessor):
    """Emits a single audio_track node covering the whole recording.

    Diarisation and transcript chunking happen in the enrichment phase, which
    subdivides this node rather than re-reading the original file.
    """

    media_type: ClassVar[MediaType] = MediaType.AUDIO

    def __init__(self, workspace: Workspace, audio_extractor: AudioExtractor) -> None:
        self._workspace = workspace
        self._audio_extractor = audio_extractor

    def process(self, source: ScannedFile) -> list[EvidenceNodeDraft]:
        destination = self._workspace.audio_path(source)
        normalised = self._audio_extractor.extract(source.path, destination)
        duration = self._duration_of(normalised)

        log.info("audio %s: normalised to %s (%.1fs)",
                 source.file_name, normalised.name, duration or 0.0)

        return [
            EvidenceNodeDraft(
                node_type=NodeType.AUDIO_TRACK,
                start_time=0.0,
                end_time=duration,
                file_path=str(normalised),
                metadata={"normalised": True, "duration_sec": duration},
            )
        ]

    @staticmethod
    def _duration_of(path) -> float | None:
        """Read duration from the normalised WAV via the stdlib.

        The output format is known-good PCM WAV because we just wrote it, so
        the `wave` module suffices and no extra decode is needed.
        """
        import wave

        try:
            with wave.open(str(path), "rb") as handle:
                rate = handle.getframerate()
                return handle.getnframes() / rate if rate else None
        except (wave.Error, OSError) as exc:
            log.debug("could not read duration from %s: %s", path, exc)
            return None
