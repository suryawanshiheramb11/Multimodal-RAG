"""Timeline synchronizer: orchestrate all alignment methods and compute offsets."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from graph.repository import GraphRepository
from .audio_fingerprinting import fingerprint_audio
from .visual_matching import visual_matching
from .identity_matching import identity_matching
from .offset_estimation import estimate_offset

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynchronizationResult:
    """Result of synchronizing a pair of sources."""

    source_a_id: str
    source_b_id: str
    offset_estimate: dict | None  # if alignment succeeded
    error: str | None = None


def synchronize_source_pair(
    repository: GraphRepository,
    case_id: str,
    source_a_id: str,
    source_b_id: str,
) -> SynchronizationResult:
    """Align two sources using audio, visual, and identity methods.

    Args:
        repository: GraphRepository with data access
        case_id: case UUID
        source_a_id: UUID of first source (reference)
        source_b_id: UUID of second source

    Returns:
        SynchronizationResult with offset estimate or error message.
    """
    log.info("synchronizing sources %s -> %s", source_a_id[:8], source_b_id[:8])

    # Fetch data from both sources
    try:
        audio_a = repository.fetch_audio_segments_by_source(source_a_id)
        audio_b = repository.fetch_audio_segments_by_source(source_b_id)

        video_a = repository.fetch_video_frames_by_source(source_a_id)
        video_b = repository.fetch_video_frames_by_source(source_b_id)

        faces_a = repository.fetch_face_appearances_by_source(source_a_id)
        faces_b = repository.fetch_face_appearances_by_source(source_b_id)
    except Exception as exc:
        error = f"failed to fetch data: {exc}"
        log.error(error)
        return SynchronizationResult(source_a_id, source_b_id, None, error)

    # Run all three alignment methods
    audio_anchors = []
    if audio_a and audio_b:
        audio_anchors = fingerprint_audio(audio_a, audio_b)
        log.info("audio_fingerprinting yielded %d anchors", len(audio_anchors))

    visual_anchors = []
    if video_a and video_b:
        visual_anchors = visual_matching(video_a, video_b)
        log.info("visual_matching yielded %d anchors", len(visual_anchors))

    identity_anchors = []
    if faces_a and faces_b:
        identity_anchors = identity_matching(faces_a, faces_b)
        log.info("identity_matching yielded %d anchors", len(identity_anchors))

    # Estimate offset from all anchors
    offset_est = estimate_offset(audio_anchors, visual_anchors, identity_anchors)
    if offset_est is None:
        error = "no anchors found from any alignment method"
        log.warning(error)
        return SynchronizationResult(source_a_id, source_b_id, None, error)

    # Store result
    try:
        repository.insert_source_offset(
            case_id=case_id,
            source_a_id=source_a_id,
            source_b_id=source_b_id,
            offset_seconds=offset_est.offset_seconds,
            confidence=offset_est.confidence,
            method=", ".join(
                f"{m}({offset_est.method_counts[m]})"
                for m in ["audio", "visual", "identity"]
                if offset_est.method_counts[m] > 0
            ),
            anchor_count=offset_est.anchor_count,
            metadata={"residuals_sec": offset_est.residuals[:20]},  # store first 20
        )
    except Exception as exc:
        error = f"failed to store offset: {exc}"
        log.error(error)
        return SynchronizationResult(source_a_id, source_b_id, None, error)

    log.info(
        "synchronization complete: offset=%.2f sec, confidence=%.1f%%",
        offset_est.offset_seconds,
        offset_est.confidence * 100,
    )

    return SynchronizationResult(
        source_a_id,
        source_b_id,
        {
            "offset_seconds": offset_est.offset_seconds,
            "confidence": offset_est.confidence,
            "anchor_count": offset_est.anchor_count,
            "method_counts": offset_est.method_counts,
        },
    )


def synchronize_all_sources(
    repository: GraphRepository,
    case_id: str,
    source_ids: list[str],
    reference_source_id: str | None = None,
) -> dict[str, SynchronizationResult]:
    """Synchronize all sources against a reference.

    Args:
        repository: GraphRepository
        case_id: case UUID
        source_ids: list of source UUIDs to align
        reference_source_id: which source is the reference (time origin). If None, uses first.

    Returns:
        dict mapping "source_a->source_b" to SynchronizationResult.
    """
    if not source_ids:
        log.warning("synchronize_all_sources: no sources provided")
        return {}

    if reference_source_id is None:
        reference_source_id = source_ids[0]
    elif reference_source_id not in source_ids:
        raise ValueError(f"reference_source_id {reference_source_id} not in source list")

    results = {}
    for source_id in source_ids:
        if source_id == reference_source_id:
            continue
        key = f"{reference_source_id[:8]}→{source_id[:8]}"
        result = synchronize_source_pair(repository, case_id, reference_source_id, source_id)
        results[key] = result

    # Update case_time for all nodes
    if all(r.error is None for r in results.values()):
        updated = repository.update_evidence_case_time(case_id, reference_source_id)
        log.info("updated case_time for %d evidence nodes", updated)
    else:
        log.warning("skipping case_time update due to synchronization errors")

    return results
