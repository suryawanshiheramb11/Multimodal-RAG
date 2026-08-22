"""Feature extraction for video scene segments.

The heaviest node type: speech, audio events, and every visual model, fused
into one text field and three embeddings.
"""
from __future__ import annotations

import logging
from typing import ClassVar

from ..config import EnrichmentSettings
from ..registry import ModelRegistry
from .base import EnrichmentResult, NodeAnalyzer, PendingNode
from .text_fusion import build_text_embedding, fuse_text
from .visual import VisualExtractor

log = logging.getLogger(__name__)


class VideoSegmentAnalyzer(NodeAnalyzer):
    """ASR + audio tagging + visual models over one scene segment."""

    node_type: ClassVar[str] = "scene_segment"

    def __init__(self, registry: ModelRegistry, settings: EnrichmentSettings) -> None:
        self._registry = registry
        self._settings = settings
        self._visual = VisualExtractor(registry, settings)

    def analyze(self, node: PendingNode) -> EnrichmentResult:
        result = EnrichmentResult()

        transcript_text = self._transcribe(node, result)
        self._tag_audio(node, result)

        visual = self._visual.extract(
            node.frame_paths, node.representative_frame(), result
        )
        result.metadata.update(visual.metadata)
        result.clip_embedding = visual.clip_embedding

        # The searchable text for a segment is everything said, seen, and read
        # in it — the transcript alone misses a weapon on a table, the caption
        # alone misses the threat that was spoken.
        result.text_content = fuse_text(
            transcript=transcript_text,
            caption=visual.caption,
            ocr=visual.ocr_text,
        )
        build_text_embedding(self._registry, result)
        return result

    def _transcribe(self, node: PendingNode, result: EnrichmentResult) -> str | None:
        if not self._settings.enable_asr:
            result.note_skip("asr", "disabled in config")
            return None

        audio_path = node.audio_path
        if audio_path is None:
            result.note_skip("asr", "segment has no audio track")
            return None

        transcriber = self._registry.transcriber
        if not transcriber.available:
            result.note_skip("asr", transcriber.unavailable_reason or "Whisper unavailable")
            return None

        # Transcribe the whole track once (cached), then take this segment's slice.
        full = transcriber.transcribe_file(audio_path)
        if full is None:
            result.note_skip("asr", "transcription produced no result")
            return None

        start = node.start_time if node.start_time is not None else 0.0
        end = node.end_time if node.end_time is not None else float("inf")
        window = full.between(start, end)

        result.metadata["transcript"] = {
            "text": window.text,
            "language": window.language,
            "language_probability": (
                round(window.language_probability, 4)
                if window.language_probability is not None else None
            ),
            "segments": [
                {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
                for s in window.segments
            ],
        }
        return window.text or None

    def _tag_audio(self, node: PendingNode, result: EnrichmentResult) -> None:
        if not self._settings.enable_audio_events:
            result.note_skip("audio_events", "disabled in config")
            return

        audio_path = node.audio_path
        if audio_path is None:
            result.note_skip("audio_events", "segment has no audio track")
            return

        classifier = self._registry.audio_events
        if not classifier.available:
            result.note_skip("audio_events", classifier.unavailable_reason or "AST unavailable")
            return

        analysis = classifier.analyse(audio_path, node.start_time, node.end_time)
        if analysis is None:
            result.note_skip("audio_events", "tagging produced no result")
            return

        result.audio_embedding = analysis.embedding
        result.metadata["audio_events"] = [
            {"label": e.label, "probability": e.probability} for e in analysis.events
        ]
