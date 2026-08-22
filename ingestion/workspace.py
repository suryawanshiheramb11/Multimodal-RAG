"""Output locations for derived media (extracted audio, frames, page renders).

Derived filenames are built from the source file's SHA256 prefix rather than
its original name. That kills filename-injection and traversal through crafted
evidence names in one move, keeps names collision-free across a case, and makes
every artefact traceable back to the exact bytes it came from.
"""
from __future__ import annotations

from pathlib import Path

from .models import ScannedFile
from .security import ensure_private_dir

#: Enough SHA256 to be collision-free within a case, short enough to read.
_KEY_LENGTH = 16


class Workspace:
    """Allocates private output directories for derived artefacts."""

    def __init__(self, audio_dir: Path, frames_dir: Path, pages_dir: Path) -> None:
        self._audio_dir = audio_dir
        self._frames_dir = frames_dir
        self._pages_dir = pages_dir

    def prepare(self) -> None:
        for directory in (self._audio_dir, self._frames_dir, self._pages_dir):
            ensure_private_dir(directory)

    @staticmethod
    def key(source: ScannedFile) -> str:
        return source.sha256[:_KEY_LENGTH]

    def audio_path(self, source: ScannedFile) -> Path:
        ensure_private_dir(self._audio_dir)
        return self._audio_dir / f"{self.key(source)}.wav"

    def frames_dir(self, source: ScannedFile) -> Path:
        return ensure_private_dir(self._frames_dir / self.key(source))

    def pages_dir(self, source: ScannedFile) -> Path:
        return ensure_private_dir(self._pages_dir / self.key(source))
