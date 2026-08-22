"""Feature extraction for standalone audio recordings.

Not in the original phase outline, which covers video, images, and pages — but
an interview recording is one of the commonest exhibits in a case, and without
this it would be ingested and then never analysed.
"""
from __future__ import annotations

import logging
from typing import ClassVar

from ..config import EnrichmentSettings
from ..registry import ModelRegistry
from .base import EnrichmentResult, NodeAnalyzer, PendingNode
from .text_fusion import build_text_embedding, fuse_text

log = logging.getLogger(__name__)


class AudioTrackAnalyzer(NodeAnalyzer):
    node_type: ClassVar[str] = "audio_track"

    def __init__(self, registry: ModelRegistry, settings: EnrichmentSettings) -> None:
        self._registry = registry
        self._settings = settings

    def analyze(self, node: PendingNode) -> EnrichmentResult:
        result = EnrichmentResult()
        audio_path = node.audio_path

        if audio_path is None:
            result.note_skip("asr", "node has no audio path")
            result.note_skip("audio_events", "node has no audio path")
            return result

        transcript_text = self._transcribe(audio_path, result)
        self._tag_audio(audio_path, result)

        result.text_content = fuse_text(transcript=transcript_text)
        build_text_embedding(self._registry, result)
        return result

    def _transcribe(self, audio_path, result: EnrichmentResult) -> str | None:
        if not self._settings.enable_asr:
            result.note_skip("asr", "disabled in config")
            return None

        transcriber = self._registry.transcriber
        if not transcriber.available:
            result.note_skip("asr", transcriber.unavailable_reason or "Whisper unavailable")
            return None

        transcript = transcriber.transcribe_file(audio_path)
        if transcript is None:
            result.note_skip("asr", "transcription produced no result")
            return None

        result.metadata["transcript"] = {
            "text": transcript.text,
            "language": transcript.language,
            "language_probability": (
                round(transcript.language_probability, 4)
                if transcript.language_probability is not None else None
            ),
            "segments": [
                {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
                for s in transcript.segments
            ],
        }
        return transcript.text or None

    def _tag_audio(self, audio_path, result: EnrichmentResult) -> None:
        if not self._settings.enable_audio_events:
            result.note_skip("audio_events", "disabled in config")
            return

        classifier = self._registry.audio_events
        if not classifier.available:
            result.note_skip("audio_events", classifier.unavailable_reason or "AST unavailable")
            return

        # No time bounds: an audio_track node covers the whole recording.
        analysis = classifier.analyse(audio_path)
        if analysis is None:
            result.note_skip("audio_events", "tagging produced no result")
            return

        result.audio_embedding = analysis.embedding
        result.metadata["audio_events"] = [
            {"label": e.label, "probability": e.probability} for e in analysis.events
        ]
