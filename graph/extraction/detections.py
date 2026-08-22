"""Turning phase 2's YOLO detections into entities, at no extra model cost.

The object-detection metadata already sits on each node from enrichment; this
just reads it back rather than re-running YOLO.
"""
from __future__ import annotations

from ..config import DETECTION_ENTITY_TYPES
from .entities import ExtractedEntity


def entities_from_detections(node_metadata: dict) -> list[ExtractedEntity]:
    """Map notable COCO labels in a node's stored detections to entities."""
    labels = (node_metadata.get("detections") or {}).get("labels") or {}
    return [
        ExtractedEntity(
            entity_type=DETECTION_ENTITY_TYPES[label], name=label, source="object_detection"
        )
        for label in labels
        if label in DETECTION_ENTITY_TYPES
    ]
