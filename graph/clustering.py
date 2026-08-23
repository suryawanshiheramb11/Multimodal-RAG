"""Clustering embeddings into face_cluster / voice_cluster rows.

Two algorithms, deliberately different: faces use DBSCAN (density-based —
noise is left unclustered rather than forced into a group), voices use
agglomerative clustering (every turn ends up in some cluster). A speaker who
only spoke once should not be treated as "not real enough" to identify the
way an isolated stray face detection might be, so voice clustering has no
noise concept; DBSCAN's min_samples would penalise exactly the person who
spoke briefly but clearly.
"""
from __future__ import annotations

import logging

import numpy as np

from .config import GraphSettings
from .repository import GraphRepository

log = logging.getLogger(__name__)


def cluster_faces(repository: GraphRepository, case_id: str, settings: GraphSettings) -> int:
    """Cluster every detected face in the case, replacing any prior clustering.

    Returns the number of clusters created. Detections that DBSCAN calls noise
    (label -1) are left with no cluster rather than forced into one — an
    unclustered face is more honest than a false grouping.
    """
    from sklearn.cluster import DBSCAN

    repository.clear_face_clusters(case_id)

    rows = repository.fetch_face_embeddings(case_id)
    if len(rows) < settings.face_cluster_min_samples:
        log.info(
            "face clustering: only %d face(s), below min_samples=%d; skipping",
            len(rows), settings.face_cluster_min_samples,
        )
        return 0

    ids = [r[0] for r in rows]
    embeddings = np.stack([r[1] for r in rows])

    labels = DBSCAN(
        eps=settings.face_cluster_eps,
        min_samples=settings.face_cluster_min_samples,
        metric="cosine",
    ).fit_predict(embeddings)

    clusters_created = 0
    for label in sorted(set(labels)):
        if label == -1:
            continue  # DBSCAN noise: no cluster assigned

        member_indices = [i for i, assigned in enumerate(labels) if assigned == label]
        member_ids = [ids[i] for i in member_indices]
        centroid = embeddings[member_indices].mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-8)

        cluster_id = repository.create_face_cluster(case_id, centroid, len(member_ids))
        repository.assign_faces_to_cluster(member_ids, cluster_id)
        clusters_created += 1

    unclustered = int((labels == -1).sum())
    log.info(
        "face clustering: %d face(s) -> %d cluster(s), %d unclustered",
        len(rows), clusters_created, unclustered,
    )
    return clusters_created


def cluster_voices(repository: GraphRepository, case_id: str, settings: GraphSettings) -> int:
    """Cluster every diarized speaker turn in the case, replacing any prior
    clustering. Returns the number of clusters created.

    Agglomerative clustering with `distance_threshold` (rather than a fixed
    `n_clusters`) because the number of distinct speakers in a case is
    exactly what clustering is meant to discover, not something known ahead
    of time.
    """
    from sklearn.cluster import AgglomerativeClustering

    repository.clear_voice_clusters(case_id)

    rows = repository.fetch_voice_embeddings(case_id)
    # scikit-learn's AgglomerativeClustering hard-requires at least 2 samples
    # regardless of configuration, so the effective floor is never lower than
    # that even if voice_cluster_min_segments is configured to 1.
    effective_min = max(2, settings.voice_cluster_min_segments)
    if len(rows) < effective_min:
        log.info(
            "voice clustering: only %d segment(s), below min_segments=%d; skipping",
            len(rows), effective_min,
        )
        return 0

    ids = [r[0] for r in rows]
    embeddings = np.stack([r[1] for r in rows])

    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=settings.voice_cluster_distance_threshold,
        metric="cosine",
        linkage="average",
    ).fit_predict(embeddings)

    clusters_created = 0
    for label in sorted(set(labels)):
        member_indices = [i for i, assigned in enumerate(labels) if assigned == label]
        member_ids = [ids[i] for i in member_indices]
        centroid = embeddings[member_indices].mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-8)

        cluster_id = repository.create_voice_cluster(case_id, centroid, len(member_ids))
        repository.assign_voice_segments_to_cluster(member_ids, cluster_id)
        clusters_created += 1

    log.info(
        "voice clustering: %d segment(s) -> %d cluster(s)", len(rows), clusters_created
    )
    return clusters_created
