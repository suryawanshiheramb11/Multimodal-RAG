"""DBSCAN clustering of face embeddings into face_cluster rows.

scikit-learn's DBSCAN is used directly rather than hand-rolled: it already
implements cosine-metric neighbourhood search efficiently, which is exactly
what grouping face embeddings needs.
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
