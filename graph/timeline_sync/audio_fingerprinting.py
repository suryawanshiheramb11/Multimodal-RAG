"""Audio fingerprinting: align sources via AST embedding similarity.

For every pair of sources with audio, compute cosine similarity between all
AST embeddings. Keep pairs with similarity > 0.9 as potential matches.
Group consecutive high-similarity matches into robust anchor events.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioAnchor:
    """An anchor event: high-similarity audio segments from two sources."""

    time_a: float
    time_b: float
    similarity: float
    anchor_index: int  # which cluster of anchors this belongs to


def fingerprint_audio(
    embeddings_a: list[tuple[float, float, np.ndarray]],  # (start, end, embedding)
    embeddings_b: list[tuple[float, float, np.ndarray]],
    similarity_threshold: float = 0.9,
) -> list[AudioAnchor]:
    """Find audio anchors between two sources via AST embedding similarity.

    Args:
        embeddings_a: list of (start_time, end_time, embedding) for source A
        embeddings_b: list of (start_time, end_time, embedding) for source B
        similarity_threshold: cosine similarity above which to consider a match

    Returns:
        List of AudioAnchor objects, grouped by temporal proximity within each source.
    """
    if not embeddings_a or not embeddings_b:
        return []

    # Build embedding matrices: (n_segments, embedding_dim)
    emb_a = np.stack([e[2] for e in embeddings_a])
    emb_b = np.stack([e[2] for e in embeddings_b])

    # Compute cosine distance (1 - cosine_similarity) for each pair
    # cdist returns distances; convert to similarities.
    distances = cdist(emb_a, emb_b, metric="cosine")
    similarities = 1.0 - distances

    # Find all high-similarity pairs (i, j) where similarity > threshold
    high_sim_pairs = np.argwhere(similarities > similarity_threshold)
    if len(high_sim_pairs) == 0:
        log.info("audio_fingerprinting: no high-similarity pairs found (threshold=%.2f)",
                 similarity_threshold)
        return []

    # Cluster pairs into anchor groups based on temporal proximity.
    # Group by rounding to nearest 5 seconds; pairs within 5-second windows
    # likely represent the same event.
    pairs_with_time = [
        (embeddings_a[i][0], embeddings_b[j][0], float(similarities[i, j]))
        for i, j in high_sim_pairs
    ]
    pairs_with_time.sort(key=lambda x: x[0])  # sort by time_a

    # Group into 5-second windows to cluster related matches
    clusters: dict[int, list] = {}
    for time_a, time_b, sim in pairs_with_time:
        cluster_key = int(time_a / 5.0)  # 5-second windows
        clusters.setdefault(cluster_key, []).append((time_a, time_b, sim))

    anchors: list[AudioAnchor] = []
    for cluster_idx, cluster_matches in sorted(clusters.items()):
        for time_a, time_b, sim in cluster_matches:
            anchors.append(AudioAnchor(time_a, time_b, sim, cluster_idx))

    log.info("audio_fingerprinting: found %d anchor pairs in %d clusters",
             len(anchors), len(clusters))
    return anchors
