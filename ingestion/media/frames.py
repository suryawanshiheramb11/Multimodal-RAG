"""Frame sampling: one image per interval, per segment.

Decoding is a single forward pass with PyAV. The previous version seeked with
OpenCV's CAP_PROP_POS_FRAMES for every sample, which is both slow (each seek
re-decodes from the preceding keyframe) and inaccurate on streams with
B-frames, where OpenCV routinely lands on a different frame than requested.
A sequential pass reads each packet once and carries exact presentation
timestamps.
"""
from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import av

from ..errors import MediaProcessingError
from ..security import harden_file
from .scenes import TimeSegment

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampledFrame:
    segment_index: int
    timestamp: float
    path: Path


class FrameSampler:
    """Extracts frames at fixed intervals within each segment."""

    def __init__(self, interval_sec: float, max_frames: int, jpeg_quality: int = 90) -> None:
        self._interval_sec = interval_sec
        self._max_frames = max_frames
        self._jpeg_quality = jpeg_quality

    def sample(
        self, path: Path, segments: list[TimeSegment], out_dir: Path
    ) -> dict[int, list[SampledFrame]]:
        """Return frames grouped by segment index.

        The frame budget is global to the file, so a video with thousands of
        short scenes cannot multiply its way past the limit.
        """
        targets = list(
            itertools.islice(self._iter_targets(segments), self._max_frames)
        )
        if not targets:
            return {}

        by_segment: dict[int, list[SampledFrame]] = defaultdict(list)
        try:
            with av.open(str(path)) as container:
                if not container.streams.video:
                    raise MediaProcessingError(f"no video stream in {path.name}")
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"

                cursor = 0
                for frame in container.decode(stream):
                    if cursor >= len(targets):
                        break
                    timestamp = frame.time
                    if timestamp is None:
                        continue

                    written: Path | None = None
                    # One decoded frame can satisfy several targets when the
                    # stream is sparser than the sample interval; encode once
                    # and point every satisfied target at the same file.
                    while cursor < len(targets) and targets[cursor][1] <= timestamp:
                        segment_index, requested = targets[cursor]
                        if written is None:
                            written = self._write(frame, out_dir, segment_index, requested)
                        by_segment[segment_index].append(
                            SampledFrame(segment_index, timestamp, written)
                        )
                        cursor += 1
        except av.FFmpegError as exc:
            raise MediaProcessingError(f"frame extraction failed for {path.name}: {exc}") from exc

        if len(targets) >= self._max_frames:
            log.warning("%s hit the %d frame cap", path.name, self._max_frames)
        return dict(by_segment)

    def _iter_targets(self, segments: list[TimeSegment]):
        """Yield (segment_index, timestamp) in ascending time order."""
        for segment in segments:
            timestamp = segment.start
            while timestamp < segment.end:
                yield segment.index, timestamp
                timestamp += self._interval_sec

    def _write(self, frame, out_dir: Path, segment_index: int, requested: float) -> Path:
        target = out_dir / f"seg{segment_index:04d}_t{requested:09.3f}.jpg"
        frame.to_image().save(target, format="JPEG", quality=self._jpeg_quality)
        harden_file(target)
        return target
