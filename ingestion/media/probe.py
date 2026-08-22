"""Container inspection via PyAV.

PyAV binds libav* in-process, so probing costs no subprocess and reports the
container's own timebase. The previous implementation multiplied OpenCV's
CAP_PROP_FPS by CAP_PROP_FRAME_COUNT, which is wrong for variable-frame-rate
recordings — exactly what phone footage and screen captures are.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import av

from ..errors import MediaProcessingError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoInfo:
    duration_sec: float
    width: int
    height: int
    has_audio: bool
    codec: str | None


def probe_video(path: Path) -> VideoInfo:
    """Read stream metadata without decoding the whole file."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise MediaProcessingError(f"no video stream in {path.name}")
            stream = container.streams.video[0]

            duration = _duration_seconds(container, stream)
            return VideoInfo(
                duration_sec=duration,
                width=stream.codec_context.width or 0,
                height=stream.codec_context.height or 0,
                has_audio=bool(container.streams.audio),
                codec=stream.codec_context.name,
            )
    except av.FFmpegError as exc:
        raise MediaProcessingError(f"cannot open {path.name}: {exc}") from exc


def _duration_seconds(container, stream) -> float:
    """Prefer the stream's own duration, fall back to the container's."""
    if stream.duration is not None and stream.time_base:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration) / av.time_base
    log.warning("no duration reported for %s", container.name)
    return 0.0
