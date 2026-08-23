"""Runs speaker diarization over every audio track in a case and stores the
turns it finds. Mirrors `faces.py`'s shape exactly: detect/diarize first,
cluster afterwards, in a separate pass.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import GraphSettings
from .models.voice import SpeakerDiarizer
from .repository import GraphRepository

log = logging.getLogger(__name__)


def diarize_speakers(
    repository: GraphRepository, diarizer: SpeakerDiarizer, case_id: str, settings: GraphSettings
) -> int:
    if not diarizer.available:
        log.warning("speaker diarization skipped: %s", diarizer.unavailable_reason)
        return 0

    sources = repository.fetch_audio_sources_for_diarization(case_id)
    log.info("diarizing %d audio source(s)", len(sources))

    inserted_total = 0
    for index, source in enumerate(sources, start=1):
        if index % 5 == 0:
            log.info("speaker diarization: %d/%d sources processed", index, len(sources))

        path = Path(source.audio_path)
        if not path.is_file():
            continue

        turns = diarizer.diarize(path)
        rows = [
            {
                "source_file_id": source.source_file_id,
                "start_time": turn.start,
                "end_time": turn.end,
                "speaker_label": turn.speaker_label,
                "embedding": turn.embedding,
            }
            for turn in turns
        ]
        inserted_total += len(repository.insert_voice_segments(case_id, rows))

    log.info("speaker diarization: %d voice segment(s) created", inserted_total)
    return inserted_total
