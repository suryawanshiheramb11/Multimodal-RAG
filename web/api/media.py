"""Resolving an evidence node to a viewable file, safely.

The API never accepts a filesystem path from the client. A request names a
node id; this module looks up that node's stored path and proves it resolves
inside one of the configured roots before anything is served. That keeps the
ingestion security model intact at the HTTP boundary: `..%2F..%2Fetc%2Fpasswd`
is not a path this code can ever be handed, and a symlink planted in the case
folder is rejected by the same `resolve_within` the scanner uses.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from psycopg2.extras import RealDictCursor

from ingestion.errors import SecurityError
from ingestion.security import resolve_within

log = logging.getLogger(__name__)

#: Extension -> MIME, for the handful of types this pipeline actually produces.
#: An unrecognised extension is refused rather than served as octet-stream:
#: everything reachable here was written by a processor we control, so an
#: unknown type means something unexpected is on disk.
_CONTENT_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
    ".pdf": "application/pdf",
}


@dataclass(frozen=True)
class ResolvedMedia:
    path: Path
    content_type: str


class MediaResolver:
    """Maps node/file ids to on-disk artefacts inside the allowed roots."""

    def __init__(self, roots: list[Path]) -> None:
        # Resolved once: every later containment check compares against these.
        self._roots = [root.resolve(strict=False) for root in roots]

    def _within_any_root(self, candidate: Path) -> Path:
        for root in self._roots:
            try:
                return resolve_within(root, candidate)
            except SecurityError:
                continue
        raise SecurityError(f"path is outside every served root: {candidate}")

    def _serve(self, raw_path: str | None) -> ResolvedMedia | None:
        if not raw_path:
            return None
        try:
            resolved = self._within_any_root(Path(raw_path))
        except SecurityError as exc:
            log.warning("refusing to serve %s: %s", raw_path, exc)
            return None

        if not resolved.is_file():
            return None

        content_type = _CONTENT_TYPES.get(resolved.suffix.lower())
        if content_type is None:
            log.warning("refusing to serve unknown media type: %s", resolved.name)
            return None
        return ResolvedMedia(path=resolved, content_type=content_type)

    # -- node thumbnails ----------------------------------------------------

    @staticmethod
    def thumbnail_path(node: dict) -> str | None:
        """The best still image representing a node.

        Each node type stores its image somewhere different: a page and a
        standalone image point at `file_path`, while a video segment's
        `file_path` is its extracted audio and the frames live in metadata —
        preferring the representative frame enrichment already chose keeps the
        thumbnail consistent with the frame the CLIP vector came from.
        """
        node_type = node.get("node_type")
        metadata = node.get("metadata") or {}

        if node_type in ("image", "page"):
            return node.get("file_path")

        if node_type == "scene_segment":
            representative = metadata.get("representative_frame")
            if representative:
                return representative
            frames = metadata.get("frames") or []
            for frame in frames:
                if isinstance(frame, dict) and frame.get("path"):
                    return frame["path"]
        return None  # audio_track and anything else has no still to show

    def node_thumbnail(self, conn, node_id: str) -> ResolvedMedia | None:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT node_type, file_path, metadata FROM evidence_node WHERE id = %s::uuid",
                (node_id,),
            )
            node = cur.fetchone()
        if node is None:
            return None
        return self._serve(self.thumbnail_path(dict(node)))

    # -- original files -----------------------------------------------------

    def source_file(self, conn, file_id: str) -> ResolvedMedia | None:
        """The original uploaded file, for in-page playback of video/audio."""
        with conn.cursor() as cur:
            cur.execute("SELECT file_path FROM source_file WHERE id = %s::uuid", (file_id,))
            row = cur.fetchone()
        return self._serve(row[0]) if row else None

    def node_media(self, conn, node_id: str) -> ResolvedMedia | None:
        """A node's own artefact — the extracted audio clip for a segment or
        audio track, the render for a page, the original for an image."""
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT node_type, file_path, metadata FROM evidence_node WHERE id = %s::uuid",
                (node_id,),
            )
            node = cur.fetchone()
        return self._serve(node["file_path"]) if node else None
