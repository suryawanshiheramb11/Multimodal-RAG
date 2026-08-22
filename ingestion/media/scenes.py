"""Segmentation of a video into semantic units.

PySceneDetect's content detector splits on visual discontinuity. When it finds
nothing — a static CCTV shot, a screen recording, a single unbroken take — the
video still has to be chunked for downstream embedding, so we fall back to
fixed windows.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..errors import ResourceLimitError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeSegment:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class SceneSegmenter:
    """Splits a video into segments, with a fixed-window fallback."""

    def __init__(
        self, threshold: float, fallback_window_sec: float, max_duration_sec: float
    ) -> None:
        self._threshold = threshold
        self._fallback_window_sec = fallback_window_sec
        self._max_duration_sec = max_duration_sec

    def segment(self, path: Path, duration_sec: float) -> tuple[list[TimeSegment], str]:
        """Return (segments, strategy) where strategy is 'scene' or 'fixed'."""
        if duration_sec > self._max_duration_sec:
            raise ResourceLimitError(
                f"{path.name} runs {duration_sec:.0f}s, over the "
                f"{self._max_duration_sec:.0f}s limit"
            )

        boundaries = self._detect(path)
        if boundaries:
            return self._to_segments(boundaries), "scene"
        return self.fixed_windows(duration_sec), "fixed"

    def _detect(self, path: Path) -> list[tuple[float, float]]:
        """Run content-aware detection, returning [] if it finds nothing.

        Detection failure is not fatal — the caller falls back to fixed
        windows — so every decoder error is swallowed here by design and
        recorded at warning level.
        """
        try:
            from scenedetect import SceneManager, open_video
            from scenedetect.detectors import ContentDetector

            video = open_video(str(path))
            manager = SceneManager()
            manager.add_detector(ContentDetector(threshold=self._threshold))
            manager.detect_scenes(video=video, show_progress=False)
            return [
                (start.seconds, end.seconds) for start, end in manager.get_scene_list()
            ]
        except Exception as exc:  # noqa: BLE001 - fallback path must be total
            log.warning("scene detection failed for %s (%s); using fixed windows",
                        path.name, exc)
            return []

    def fixed_windows(self, duration_sec: float) -> list[TimeSegment]:
        if duration_sec <= 0:
            return [TimeSegment(index=0, start=0.0, end=self._fallback_window_sec)]

        segments: list[TimeSegment] = []
        start = 0.0
        while start < duration_sec:
            end = min(start + self._fallback_window_sec, duration_sec)
            segments.append(TimeSegment(index=len(segments), start=start, end=end))
            start = end
        return segments

    @staticmethod
    def _to_segments(boundaries: list[tuple[float, float]]) -> list[TimeSegment]:
        return [
            TimeSegment(index=i, start=start, end=end)
            for i, (start, end) in enumerate(boundaries)
        ]
