"""Runs face detection over every frame/image in a case and stores the results.

FACE_MATCHES, per the spec, links an evidence node to a face cluster. That
edge doesn't need a separate table: each face_detection row already carries
both evidence_node_id and (after clustering) face_cluster_id, so the row
itself *is* the FACE_MATCHES edge rather than something extra to maintain
in lockstep with it.
"""
from __future__ import annotations

import logging

from .config import GraphSettings
from .models.faces import FaceDetector
from .repository import GraphRepository

log = logging.getLogger(__name__)


def detect_faces(
    repository: GraphRepository,
    detector: FaceDetector,
    case_id: str,
    settings: GraphSettings,
    only_pending: bool = True,
) -> int:
    if not detector.available:
        log.warning("face detection skipped: %s", detector.unavailable_reason)
        return 0

    frames = repository.fetch_frames_for_faces(case_id, only_pending=only_pending)
    log.info("scanning %d frame(s) for faces", len(frames))

    rows = []
    for index, frame in enumerate(frames, start=1):
        if index % 25 == 0:
            log.info("face detection: %d/%d frames scanned", index, len(frames))
        if not frame.frame_path.is_file():
            continue
        for face in detector.detect(frame.frame_path):
            rows.append(
                {
                    "evidence_node_id": frame.evidence_node_id,
                    "frame_path": frame.frame_path,
                    "bbox": list(face.bbox),
                    "confidence": face.confidence,
                    "embedding": face.embedding,
                }
            )

    inserted = repository.insert_face_detections(case_id, rows)
    log.info("face detection: %d face(s) found", len(inserted))
    return len(inserted)
