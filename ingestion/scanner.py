"""Scans the case folder, hashes each file, and registers it in source_file."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import psycopg2.extras

from .config import Config

EXT_TYPE_MAP = {
    ".mp4": "video", ".mov": "video", ".avi": "video", ".mkv": "video",
    ".wav": "audio", ".mp3": "audio", ".m4a": "audio", ".flac": "audio",
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".bmp": "image", ".tiff": "image",
    ".pdf": "pdf",
    ".doc": "doc", ".docx": "doc",
}


@dataclass
class FileObject:
    id: str
    case_id: str
    path: Path
    file_name: str
    file_type: str
    sha256: str
    size_bytes: int
    author: str | None = None
    created_date: str | None = None
    metadata: dict | None = None


def _sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _infer_type(path: Path) -> str | None:
    return EXT_TYPE_MAP.get(path.suffix.lower())


def _parse_date(value):
    if not value:
        return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return value


def scan_case_folder(conn, cfg: Config, case_id: str) -> list[FileObject]:
    """Walk cfg.case_folder, hash + classify each file, upsert into source_file,
    and return the resulting FileObject list for downstream processing."""
    case_folder = cfg.case_folder
    if not case_folder.exists():
        raise FileNotFoundError(f"case_folder not found: {case_folder}")

    results: list[FileObject] = []
    all_paths = sorted(p for p in case_folder.rglob("*") if p.is_file())

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        for path in all_paths:
            rel_path = str(path.relative_to(case_folder))
            declared = cfg.file_metadata(rel_path)
            file_type = declared.get("type") or _infer_type(path)
            if file_type is None:
                continue  # unrecognized file type, skip

            sha256 = _sha256_of(path)
            size_bytes = path.stat().st_size
            author = declared.get("author")
            created_date = _parse_date(declared.get("created_date"))

            cur.execute(
                """
                INSERT INTO source_file
                    (case_id, file_path, file_name, file_type, sha256,
                     size_bytes, author, created_date, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (case_id, sha256) DO UPDATE
                    SET file_path = EXCLUDED.file_path,
                        file_name = EXCLUDED.file_name,
                        file_type = EXCLUDED.file_type
                RETURNING id
                """,
                (
                    case_id, str(path), path.name, file_type, sha256,
                    size_bytes, author, created_date,
                    psycopg2.extras.Json(declared),
                ),
            )
            row_id = cur.fetchone()["id"]

            results.append(
                FileObject(
                    id=row_id,
                    case_id=case_id,
                    path=path,
                    file_name=path.name,
                    file_type=file_type,
                    sha256=sha256,
                    size_bytes=size_bytes,
                    author=author,
                    created_date=declared.get("created_date"),
                    metadata=declared,
                )
            )

    conn.commit()
    return results
