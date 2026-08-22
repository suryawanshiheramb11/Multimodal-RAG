"""ALIGNS_WITH edges: nodes from the same source file whose time windows overlap.

Phase 1 does not emit separate transcript nodes — a scene_segment's transcript
lives in its own metadata rather than as a standalone node — so the literal
"align the transcript node with its parent video_segment" doesn't have two
nodes to connect for video. What *does* apply, and is implemented here, is the
general case the spec's mechanism describes: any two nodes from the same file
whose [start, end) windows overlap are connected. For a video with a separate
audio_track node, that connects each scene_segment to the track it was cut
from; if a later phase adds standalone transcript_chunk nodes, this code
requires no change to also align those.
"""
from __future__ import annotations

import logging
from itertools import combinations

from .config import GraphSettings
from .repository import GraphRepository, TimeWindow

log = logging.getLogger(__name__)


def _overlap_seconds(a: TimeWindow, b: TimeWindow) -> float:
    return min(a.end_time, b.end_time) - max(a.start_time, b.start_time)


def build_temporal_alignments(
    repository: GraphRepository, case_id: str, settings: GraphSettings
) -> int:
    """Create ALIGNS_WITH edges for same-file, time-overlapping node pairs."""
    windows = repository.fetch_time_windows(case_id)

    by_source: dict[str, list[TimeWindow]] = {}
    for window in windows:
        by_source.setdefault(window.source_file_id, []).append(window)

    created = 0
    for nodes in by_source.values():
        for a, b in combinations(nodes, 2):
            if a.node_type == b.node_type:
                # Only cross-type overlap is informative (e.g. a scene segment
                # aligning with the file's audio track); two scene_segments
                # from the same fixed-window split trivially never overlap,
                # and two nodes of the same type overlapping would just be a
                # duplicate, not an alignment worth recording.
                continue
            overlap = _overlap_seconds(a, b)
            if overlap < settings.min_overlap_sec:
                continue
            if repository.insert_alignment(
                case_id, a.id, b.id, "ALIGNS_WITH", score=overlap,
                metadata={"overlap_sec": round(overlap, 3)},
            ):
                created += 1

    repository.commit()
    log.info("temporal alignment: %d ALIGNS_WITH edge(s) created", created)
    return created
