"""Speech recognition with faster-whisper.

Transcription is done once per audio file and cached, then sliced per segment.
Re-running Whisper for every scene segment would repeat the same decode dozens
of times over one bodycam recording.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .base import LazyModel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    segments: list[TranscriptSegment]
    language: str | None
    language_probability: float | None

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def between(self, start: float, end: float) -> Transcript:
        """Segments overlapping [start, end).

        Overlap rather than containment: a sentence spanning a scene cut
        belongs to both scenes, and dropping it would lose the words entirely.
        """
        selected = [s for s in self.segments if s.end > start and s.start < end]
        return Transcript(selected, self.language, self.language_probability)


class Transcriber(LazyModel):
    """faster-whisper ASR with a per-file transcript cache."""

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        super().__init__()
        self.name = f"whisper({model_name})"
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._cache: dict[str, Transcript] = {}

    def _build(self):
        from faster_whisper import WhisperModel

        return WhisperModel(
            self._model_name, device=self._device, compute_type=self._compute_type
        )

    def transcribe_file(self, audio_path: Path) -> Transcript | None:
        """Transcribe a whole audio file, caching the result by path."""
        model = self.load()
        if model is None:
            return None

        key = str(audio_path)
        if key in self._cache:
            return self._cache[key]

        if not audio_path.is_file():
            log.warning("audio file missing for ASR: %s", audio_path)
            return None

        try:
            segments, info = model.transcribe(str(audio_path), vad_filter=True)
            transcript = Transcript(
                segments=[
                    TranscriptSegment(start=s.start, end=s.end, text=s.text)
                    for s in segments  # generator: consumed here, not lazily
                ],
                language=getattr(info, "language", None),
                language_probability=getattr(info, "language_probability", None),
            )
        except Exception as exc:  # noqa: BLE001 - a bad audio file must not stop the run
            log.warning("transcription failed for %s: %s", audio_path.name, exc)
            return None

        self._cache[key] = transcript
        log.info(
            "transcribed %s: %d segment(s), language=%s",
            audio_path.name, len(transcript.segments), transcript.language,
        )
        return transcript
