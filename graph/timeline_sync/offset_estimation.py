"""Offset estimation: compute source-to-source time offsets with confidence.

Collect all anchor pairs from audio, visual, and identity methods.
Estimate offset as the median of (time_B - time_A) across all pairs.
Confidence is the fraction of pairs that agree within ±1 second.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OffsetEstimate:
    """An estimate of the time offset between two sources."""

    offset_seconds: float
    confidence: float  # 0.0-1.0: fraction of anchors within ±1 sec of median
    anchor_count: int
    method_counts: dict[str, int]  # counts by method
    residuals: list[float]  # (offset - median) for each anchor


def estimate_offset(
    audio_anchors: list,  # AudioAnchor objects
    visual_anchors: list,  # VisualAnchor objects
    identity_anchors: list,  # IdentityAnchor objects
    confidence_window_sec: float = 1.0,
) -> OffsetEstimate | None:
    """Estimate time offset between sources from all available anchors.

    Args:
        audio_anchors: from audio_fingerprinting
        visual_anchors: from visual_matching
        identity_anchors: from identity_matching
        confidence_window_sec: tolerance for "agreeing" with median (±1 sec default)

    Returns:
        OffsetEstimate or None if insufficient anchors.
    """
    # Collect all offsets: time_B - time_A
    offsets = []
    method_counts = {"audio": 0, "visual": 0, "identity": 0}

    for anchor in audio_anchors:
        offsets.append(anchor.time_b - anchor.time_a)
        method_counts["audio"] += 1

    for anchor in visual_anchors:
        offsets.append(anchor.time_b - anchor.time_a)
        method_counts["visual"] += 1

    for anchor in identity_anchors:
        offsets.append(anchor.time_b - anchor.time_a)
        method_counts["identity"] += 1

    if not offsets:
        log.warning("offset_estimation: no anchors provided")
        return None

    offsets_array = np.array(offsets)
    median_offset = float(np.median(offsets_array))

    # Compute confidence: fraction within ±confidence_window_sec
    residuals = offsets_array - median_offset
    within_window = np.abs(residuals) <= confidence_window_sec
    confidence = float(np.mean(within_window))

    log.info(
        "offset_estimation: offset=%.2f±%.2f sec, confidence=%.1f%% (%d anchors: "
        "audio=%d, visual=%d, identity=%d)",
        median_offset,
        np.std(offsets_array),
        confidence * 100,
        len(offsets),
        method_counts["audio"],
        method_counts["visual"],
        method_counts["identity"],
    )

    return OffsetEstimate(
        offset_seconds=median_offset,
        confidence=confidence,
        anchor_count=len(offsets),
        method_counts=method_counts,
        residuals=residuals.tolist(),
    )
