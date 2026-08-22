"""Identity matching: align sources via face cluster co-occurrence.

If the same face_cluster appears in both sources, the timestamps of appearance
are robust anchors (they correspond to the same real-world moment).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from collections import defaultdict

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdentityAnchor:
    """An anchor event: same identity (face cluster) appearing in both sources."""

    time_a: float
    time_b: float
    face_cluster_id: str
    confidence: float = 0.95  # high confidence: identity is unique


def identity_matching(
    face_detections_a: dict[str, list[float]],  # face_cluster_id -> [timestamps]
    face_detections_b: dict[str, list[float]],
) -> list[IdentityAnchor]:
    """Find identity anchors: same face cluster appearing in both sources.

    Args:
        face_detections_a: mapping of face_cluster_id to list of timestamps in source A
        face_detections_b: mapping of face_cluster_id to list of timestamps in source B

    Returns:
        List of IdentityAnchor objects.
    """
    anchors: list[IdentityAnchor] = []

    # Find face clusters that appear in both sources
    common_clusters = set(face_detections_a.keys()) & set(face_detections_b.keys())
    log.info("identity_matching: found %d face clusters in both sources",
             len(common_clusters))

    for cluster_id in common_clusters:
        times_a = sorted(face_detections_a[cluster_id])
        times_b = sorted(face_detections_b[cluster_id])

        # Pair up appearances: if person appears N times in A and M times in B,
        # pair the earliest with earliest, etc. This is a heuristic but works
        # well for short sequences where co-occurrence is unambiguous.
        for time_a, time_b in zip(times_a, times_b):
            anchors.append(IdentityAnchor(time_a, time_b, cluster_id, confidence=0.95))

    log.info("identity_matching: found %d identity anchors", len(anchors))
    return anchors
