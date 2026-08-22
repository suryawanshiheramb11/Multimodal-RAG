"""Visual matching: align sources via CLIP embedding similarity with temporal consistency.

For every pair of sources with video, compute CLIP image similarity between frames.
Use a temporal consistency check: if a series of frames in source A match a series
in source B with a consistent offset, those are robust anchors.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualAnchor:
    """An anchor event: matching video frames from two sources."""

    time_a: float
    time_b: float
    similarity: float
    temporal_consistency_score: float  # how well this anchor fits a diagonal


def visual_matching(
    frames_a: list[tuple[float, np.ndarray]],  # (timestamp, clip_embedding)
    frames_b: list[tuple[float, np.ndarray]],
    similarity_threshold: float = 0.85,
    temporal_window_sec: float = 5.0,
) -> list[VisualAnchor]:
    """Find visual anchors between two sources via CLIP embedding similarity.

    Uses a temporal consistency filter: a good offset should cause matches to form
    a diagonal in the time-time matrix. Rejects scattered matches that don't align.

    Args:
        frames_a: list of (timestamp, clip_embedding) for source A
        frames_b: list of (timestamp, clip_embedding) for source B
        similarity_threshold: cosine similarity above which to consider a match
        temporal_window_sec: tolerance for temporal consistency (±5 sec default)

    Returns:
        List of VisualAnchor objects with temporal consistency scores.
    """
    if not frames_a or not frames_b:
        return []

    times_a = np.array([f[0] for f in frames_a])
    times_b = np.array([f[0] for f in frames_b])
    emb_a = np.stack([f[1] for f in frames_a])
    emb_b = np.stack([f[1] for f in frames_b])

    # Compute CLIP similarity
    distances = cdist(emb_a, emb_b, metric="cosine")
    similarities = 1.0 - distances

    # Find high-similarity pairs
    high_sim_pairs = np.argwhere(similarities > similarity_threshold)
    if len(high_sim_pairs) == 0:
        log.info("visual_matching: no high-similarity pairs found (threshold=%.2f)",
                 similarity_threshold)
        return []

    # Compute temporal offsets for each pair: offset = time_b - time_a
    offsets = []
    for i, j in high_sim_pairs:
        offset = times_b[j] - times_a[i]
        offsets.append((offset, i, j, float(similarities[i, j])))

    offsets.sort(key=lambda x: x[0])  # sort by offset value

    # Cluster offsets: frames that match with similar offsets form a diagonal
    anchors: list[VisualAnchor] = []
    if offsets:
        # Group offsets into clusters using k-means-like approach:
        # round each offset to nearest second, group by that bin
        offset_bins: dict[int, list] = {}
        for offset, i, j, sim in offsets:
            bin_key = int(round(offset))  # round to nearest second
            offset_bins.setdefault(bin_key, []).append((offset, i, j, sim))

        # From each cluster, compute median offset and consistency scores
        for bin_key, bin_matches in sorted(offset_bins.items()):
            bin_offsets = [m[0] for m in bin_matches]
            median_offset = float(np.median(bin_offsets))
            bin_std = float(np.std(bin_offsets)) if len(bin_offsets) > 1 else 0.1

            for offset, i, j, sim in bin_matches:
                # Consistency: how close this offset is to bin median (lower std = higher consistency)
                residual = abs(offset - median_offset)
                # Normalize: perfect match (residual=0) → 1.0, at std deviation → 0.5
                consistency = 1.0 - (residual / (bin_std + 1.0))
                consistency = np.clip(consistency, 0.0, 1.0)
                anchors.append(VisualAnchor(times_a[i], times_b[j], sim, float(consistency)))

    log.info("visual_matching: found %d anchor pairs with temporal consistency",
             len(anchors))
    return anchors
