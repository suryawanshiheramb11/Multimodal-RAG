"""Safety primitives applied to every untrusted file the pipeline touches.

Evidence folders are attacker-influenced input: a case can contain a symlink
pointing at /etc, a .jpg that is really a 4GB zip bomb, or a filename crafted
to escape an output directory. Everything here exists to make those cases
boring failures instead of incidents.
"""
from __future__ import annotations

import hashlib
import logging
import os
import stat
from pathlib import Path

import puremagic

from .errors import ResourceLimitError, SecurityError
from .models import MediaType

log = logging.getLogger(__name__)

#: Directory mode for derived artefacts: owner-only. Evidence derivatives can
#: contain the same sensitive content as the originals.
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600

#: Extension -> media type. Used as a hint only; content sniffing decides.
EXTENSION_TYPES: dict[str, MediaType] = {
    ".mp4": MediaType.VIDEO, ".mov": MediaType.VIDEO, ".avi": MediaType.VIDEO,
    ".mkv": MediaType.VIDEO, ".webm": MediaType.VIDEO, ".m4v": MediaType.VIDEO,
    ".wav": MediaType.AUDIO, ".mp3": MediaType.AUDIO, ".m4a": MediaType.AUDIO,
    ".flac": MediaType.AUDIO, ".aac": MediaType.AUDIO, ".ogg": MediaType.AUDIO,
    ".jpg": MediaType.IMAGE, ".jpeg": MediaType.IMAGE, ".png": MediaType.IMAGE,
    ".bmp": MediaType.IMAGE, ".tiff": MediaType.IMAGE, ".tif": MediaType.IMAGE,
    ".webp": MediaType.IMAGE, ".heic": MediaType.IMAGE,
    ".pdf": MediaType.PDF,
    ".doc": MediaType.DOC, ".docx": MediaType.DOC, ".rtf": MediaType.DOC,
    ".odt": MediaType.DOC, ".txt": MediaType.DOC,
}

#: MIME prefix -> media type, for content-sniffed results.
_MIME_PREFIX_TYPES: tuple[tuple[str, MediaType], ...] = (
    ("video/", MediaType.VIDEO),
    ("audio/", MediaType.AUDIO),
    ("image/", MediaType.IMAGE),
)

_MIME_EXACT_TYPES: dict[str, MediaType] = {
    "application/pdf": MediaType.PDF,
    "application/msword": MediaType.DOC,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": MediaType.DOC,
    "application/rtf": MediaType.DOC,
    "text/rtf": MediaType.DOC,
    "application/vnd.oasis.opendocument.text": MediaType.DOC,
    # Some containers sniff as generic streams; treat as video and let the
    # decoder be the final arbiter.
    "application/octet-stream": MediaType.VIDEO,
}


def resolve_within(base: Path, candidate: Path) -> Path:
    """Resolve `candidate` and prove it stays inside `base`.

    Uses fully resolved (symlink-followed) paths on both sides, so a symlink
    inside the case folder that points outside it is rejected here rather than
    silently read.
    """
    base_resolved = base.resolve(strict=False)
    target = candidate if candidate.is_absolute() else base_resolved / candidate
    target_resolved = target.resolve(strict=False)
    if base_resolved != target_resolved and base_resolved not in target_resolved.parents:
        raise SecurityError(
            f"path escapes the case folder: {candidate} -> {target_resolved}"
        )
    return target_resolved


def assert_regular_file(path: Path) -> None:
    """Reject anything that is not a plain file.

    Symlinks, FIFOs, devices, and sockets can block forever on read or point
    somewhere they should not; none of them are evidence.
    """
    try:
        info = path.lstat()
    except OSError as exc:
        raise SecurityError(f"cannot stat {path}: {exc}") from exc

    if stat.S_ISLNK(info.st_mode):
        raise SecurityError(f"refusing to ingest symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise SecurityError(f"not a regular file: {path}")


def assert_within_size_limit(path: Path, max_bytes: int) -> int:
    size = path.stat().st_size
    if size > max_bytes:
        raise ResourceLimitError(
            f"{path.name} is {size} bytes, over the {max_bytes} byte limit"
        )
    return size


def sha256_file(path: Path) -> str:
    """Hash a file with the stdlib's buffered digest helper.

    hashlib.file_digest (3.11+) reads through a reusable buffer and releases
    the GIL around the read, which replaces the hand-rolled chunk loop this
    used to have.
    """
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sniff_mime(path: Path) -> str | None:
    """Best-effort content-based MIME detection. None when undetectable."""
    try:
        return puremagic.from_file(str(path), mime=True) or None
    except (puremagic.PureError, ValueError, OSError):
        return None


def media_type_from_mime(mime: str | None) -> MediaType | None:
    if not mime:
        return None
    if mime in _MIME_EXACT_TYPES:
        return _MIME_EXACT_TYPES[mime]
    for prefix, media_type in _MIME_PREFIX_TYPES:
        if mime.startswith(prefix):
            return media_type
    if mime.startswith("text/"):
        return MediaType.DOC
    return None


def classify(
    path: Path, declared: MediaType | None
) -> tuple[MediaType | None, str | None, bool]:
    """Decide a file's media type from its declaration, content, and extension.

    Returns ``(media_type, detected_mime, mismatch)``.

    Content sniffing takes precedence over both the config declaration and the
    extension: a file's bytes are harder to fake than its name, and handing a
    zip archive to the video decoder because it was called ``.mp4`` is exactly
    the confusion an attacker wants.

    A mismatch is reported when the content disagrees with *either* the config
    declaration or the extension. Checking the extension as well matters: an
    undeclared file dropped into the case folder has no declaration to
    contradict, and masquerading by extension alone is the more common trick.
    In a forensic context a mislabelled exhibit is itself evidence, so the
    conflict is recorded rather than quietly resolved.
    """
    detected_mime = sniff_mime(path)
    detected_type = media_type_from_mime(detected_mime)
    extension_type = EXTENSION_TYPES.get(path.suffix.lower())

    resolved = detected_type or declared or extension_type
    expected = declared or extension_type
    mismatch = bool(expected and detected_type and expected != detected_type)

    if mismatch:
        claimed_by = "config" if declared else "extension"
        log.warning(
            "type mismatch for %s: %s says %s, content sniffs as %s (%s)",
            path.name, claimed_by, expected.value, detected_type.value, detected_mime,
        )
    return resolved, detected_mime, mismatch


def ensure_private_dir(path: Path) -> Path:
    """Create a directory readable only by the owner.

    mkdir's mode is masked by the process umask, so the mode is set explicitly
    afterwards to guarantee 0700 regardless of environment.
    """
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, PRIVATE_DIR_MODE)
    except OSError as exc:  # e.g. a mounted volume that ignores chmod
        log.debug("could not tighten permissions on %s: %s", path, exc)
    return path


def harden_file(path: Path) -> None:
    """Restrict a derived artefact to owner-only access."""
    try:
        os.chmod(path, PRIVATE_FILE_MODE)
    except OSError as exc:
        log.debug("could not tighten permissions on %s: %s", path, exc)


def assert_safe_external_path(path: Path) -> str:
    """Return a path string safe to hand to an external command.

    A file named ``-i`` or ``--strict`` is parsed as an option by nearly every
    CLI tool, so only absolute paths — which can never be mistaken for a flag —
    are allowed through to ffmpeg.
    """
    if not path.is_absolute():
        raise SecurityError(f"external tools require an absolute path, got: {path}")
    return str(path)
