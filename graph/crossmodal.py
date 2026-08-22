"""Phase 4: document-to-video linking, transcript-to-frame linking, and the
audio cross-source alignment prep check.

REFERENCES and DESCRIBES both connect nodes across different source files by
CLIP similarity rather than by shared time or an explicit mention — the
"semantic" half of phase 4's temporal+semantic refinement, complementing
ALIGNS_WITH (temporal) and MENTIONS (entity).
"""
from __future__ import annotations

import logging

import numpy as np

from enrichment.models.clip import ClipEncoder

from .config import GraphSettings
from .repository import GraphRepository

log = logging.getLogger(__name__)


def build_document_video_references(
    repository: GraphRepository, case_id: str, settings: GraphSettings
) -> int:
    """REFERENCES edges between pdf pages and video segments that show the
    same thing, purely from already-stored CLIP image vectors."""
    created = repository.insert_document_video_references(
        case_id, settings.reference_similarity_threshold, settings.max_nodes_for_cross_modal
    )
    log.info("document-video linking: %d REFERENCES edge(s) created", created)
    return created


def build_transcript_visual_links(
    repository: GraphRepository, clip_encoder: ClipEncoder, case_id: str, settings: GraphSettings
) -> int:
    """DESCRIBES edges from a transcript to any frame its CLIP text embedding
    is visually close to.

    Unlike REFERENCES, the transcript side has no stored CLIP vector (a
    node's single clip_embedding column already holds its *image* vector when
    it has one); the text vector is computed here and never persisted, so the
    comparison is done in Python against the frame vectors pulled once from
    the DB rather than as one big SQL join.
    """
    if not clip_encoder.available:
        log.warning("transcript-visual linking skipped: %s", clip_encoder.unavailable_reason)
        return 0

    transcripts = repository.fetch_transcript_nodes(case_id)
    frames = repository.fetch_clip_frame_nodes(case_id, ["scene_segment", "image"])
    if not transcripts or not frames:
        log.info(
            "transcript-visual linking: nothing to do (%d transcript(s), %d frame(s))",
            len(transcripts), len(frames),
        )
        return 0

    if len(transcripts) > settings.max_nodes_for_cross_modal or len(frames) > settings.max_nodes_for_cross_modal:
        raise ValueError(
            f"{len(transcripts)} transcript(s) x {len(frames)} frame(s) exceeds the "
            f"{settings.max_nodes_for_cross_modal}-node cross-modal cap; raise "
            "max_nodes_for_cross_modal or restrict the case"
        )

    frame_matrix = np.stack([f.embedding for f in frames])  # (F, 512), L2-normalised

    created = 0
    for index, transcript in enumerate(transcripts, start=1):
        if index % 25 == 0:
            log.info("transcript-visual linking: %d/%d transcripts processed", index, len(transcripts))

        text_vector = clip_encoder.embed_text(transcript.text_content)
        if text_vector is None:
            continue

        similarities = frame_matrix @ text_vector  # both sides normalised -> cosine similarity
        for frame, similarity in zip(frames, similarities):
            if frame.id == transcript.id:
                continue
            if float(similarity) <= settings.describes_similarity_threshold:
                continue

            metadata = {
                "cosine_similarity": round(float(similarity), 4),
                "frame_path": frame.frame_path,
                "frame_timestamp": frame.frame_timestamp,
            }
            if repository.insert_alignment(
                case_id, transcript.id, frame.id, "DESCRIBES", float(similarity), metadata
            ):
                created += 1

    repository.commit()
    log.info("transcript-visual linking: %d DESCRIBES edge(s) created", created)
    return created


def verify_audio_alignment_prep(repository: GraphRepository, case_id: str) -> dict:
    """Confirm every audio-bearing node has its AST embedding stored (and,
    via the schema's HNSW index, searchable) — the precondition a later
    phase's cross-source offset estimation needs. Read-only: this step
    computes nothing new, it only reports what enrichment already produced.
    """
    coverage = repository.audio_embedding_coverage(case_id)
    for node_type, counts in coverage.items():
        missing = counts["total"] - counts["with_embedding"]
        if missing:
            log.warning(
                "audio alignment prep: %d/%d %s node(s) missing audio_embedding",
                missing, counts["total"], node_type,
            )
        else:
            log.info(
                "audio alignment prep: %d/%d %s node(s) have audio_embedding",
                counts["with_embedding"], counts["total"], node_type,
            )
    return coverage
