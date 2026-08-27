"""Speech recognition with faster-whisper.

Transcription is done once per audio file and cached, then sliced per segment.
Re-running Whisper for every scene segment would repeat the same decode dozens
of times over one bodycam recording.
"""
from __future__ import annotations

import logging
import os
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

    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        cpu_threads: int = 0,
        batch_size: int = 8,
    ) -> None:
        super().__init__()
        self.name = f"whisper({model_name})"
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads or (os.cpu_count() or 4)
        self._batch_size = batch_size
        self._runner = None
        self._cache: dict[str, Transcript] = {}

    def _build(self):
        from faster_whisper import WhisperModel

        model = WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
            cpu_threads=self._cpu_threads,
        )

        # The batched pipeline decodes several windows per forward pass. It is
        # optional because it is a later faster-whisper addition and because
        # batch_size=1 is the documented way back to sequential decoding.
        self._runner = model
        if self._batch_size > 1:
            try:
                from faster_whisper import BatchedInferencePipeline

                self._runner = BatchedInferencePipeline(model=model)
            except ImportError:
                log.info("faster-whisper has no batched pipeline; decoding sequentially")
        return model

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

        options = {"vad_filter": True}
        if self._runner is not model:
            options["batch_size"] = self._batch_size

        try:
            segments, info = self._runner.transcribe(str(audio_path), **options)
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
