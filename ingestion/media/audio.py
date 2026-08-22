"""Audio normalisation to the format the ASR stage expects (16 kHz mono WAV).

ffmpeg is invoked as a subprocess rather than through PyAV: resampling and
downmixing through libswresample by hand is a lot of code to own when the CLI
does it in one flag each. The call is hardened rather than avoided.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ..errors import MediaProcessingError
from ..security import assert_safe_external_path, harden_file

log = logging.getLogger(__name__)

#: Truncated ffmpeg stderr kept on failure; enough to diagnose, bounded so a
#: pathological file cannot flood the logs.
_ERROR_TAIL_CHARS = 2000


class AudioExtractor:
    """Extracts or normalises an audio track to 16 kHz mono WAV."""

    def __init__(self, sample_rate_hz: int, timeout_sec: int, ffmpeg: str = "ffmpeg") -> None:
        self._sample_rate_hz = sample_rate_hz
        self._timeout_sec = timeout_sec
        self._ffmpeg = ffmpeg

    def extract(self, source: Path, destination: Path) -> Path:
        """Write normalised audio from `source` to `destination`.

        Raises MediaProcessingError when the source carries no audio track,
        which is a normal condition for silent video, not a crash.
        """
        source_arg = assert_safe_external_path(source)
        destination_arg = assert_safe_external_path(destination)

        command = [
            self._ffmpeg,
            "-nostdin",              # never block waiting on stdin
            "-loglevel", "error",    # bounded stderr
            "-y",
            "-i", source_arg,
            "-vn",                   # ignore any video stream
            "-map_metadata", "-1",   # drop source metadata from the derivative
            "-ac", "1",
            "-ar", str(self._sample_rate_hz),
            "-f", "wav",
            destination_arg,
        ]

        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_sec,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise MediaProcessingError(
                f"{self._ffmpeg} not found on PATH; install ffmpeg to ingest media"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise MediaProcessingError(
                f"audio extraction timed out after {self._timeout_sec}s for {source.name}"
            ) from exc

        if result.returncode != 0:
            stderr = (result.stderr or "")[-_ERROR_TAIL_CHARS:].strip()
            raise MediaProcessingError(
                f"ffmpeg failed on {source.name} (exit {result.returncode}): {stderr}"
            )

        if not destination.exists() or destination.stat().st_size == 0:
            raise MediaProcessingError(f"no audio track produced for {source.name}")

        harden_file(destination)
        return destination
