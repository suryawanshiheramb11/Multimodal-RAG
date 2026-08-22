"""Combining a node's several text sources into one searchable field.

Transcript, caption, and OCR each describe the same moment from a different
angle. They are labelled rather than concatenated blindly so that a later
reader — human or LLM — can tell what was *said* from what was *seen*.
"""
from __future__ import annotations

import logging

from ..registry import ModelRegistry
from .base import EnrichmentResult

log = logging.getLogger(__name__)


def fuse_text(
    transcript: str | None = None,
    caption: str | None = None,
    ocr: str | None = None,
    page_text: str | None = None,
) -> str | None:
    """Join the available text sources under explicit headings."""
    parts = [
        ("Transcript", transcript),
        ("Visual description", caption),
        ("On-screen text", ocr),
        ("Document text", page_text),
    ]
    sections = [
        f"{label}: {value.strip()}"
        for label, value in parts
        if value and value.strip()
    ]
    return "\n\n".join(sections) if sections else None


def build_text_embedding(registry: ModelRegistry, result: EnrichmentResult) -> None:
    """Embed the fused text with MiniLM, recording why if it cannot be done."""
    if not result.text_content:
        result.note_skip("text_embedding", "node has no text to embed")
        return

    encoder = registry.text_encoder
    if not encoder.available:
        result.note_skip(
            "text_embedding", encoder.unavailable_reason or "text encoder unavailable"
        )
        return

    embedding = encoder.embed(result.text_content)
    if embedding is None:
        result.note_skip("text_embedding", "encoder produced no vector")
        return
    result.text_embedding = embedding


def build_clip_text_embedding(registry: ModelRegistry, text: str | None) -> None:
    """CLIP text vector, for text-to-image retrieval.

    Kept separate from the image vector: both live in CLIP's joint 512-d space,
    but only one can occupy the node's clip_embedding column, and for a node
    that has frames the image vector is the more faithful representation.
    """
    if not text:
        return None
    clip = registry.clip
    if not clip.available:
        return None
    return clip.embed_text(text)
