"""Domain types shared across the pipeline.

These are deliberately free of database and filesystem concerns so that
processors can be unit-tested without Postgres or real media files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path


class MediaType(StrEnum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    PDF = "pdf"
    DOC = "doc"


class NodeType(StrEnum):
    """Kinds of evidence_node rows this phase produces.

    Later phases add transcript_chunk, ocr_block, detection, etc.
    """

    SCENE_SEGMENT = "scene_segment"
    FRAME = "frame"
    AUDIO_TRACK = "audio_track"
    PAGE = "page"
    IMAGE = "image"


@dataclass(frozen=True)
class ScannedFile:
    """One evidence file discovered on disk, hashed but not yet processed.

    `declared_type` comes from config.yaml, `detected_type` from content
    sniffing. When they disagree the scanner records it: for forensic work a
    mislabelled file is itself a finding, not just a nuisance.
    """

    path: Path
    file_name: str
    media_type: MediaType
    sha256: str
    size_bytes: int
    declared_type: str | None = None
    detected_mime: str | None = None
    type_mismatch: bool = False
    author: str | None = None
    created_date: datetime | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceNodeDraft:
    """A unit of extracted evidence, before it is persisted.

    Processors return these; only the repository layer knows they become rows.
    Embedding columns stay NULL here — they are filled by the enrichment phase.
    """

    node_type: NodeType
    start_time: float | None = None
    end_time: float | None = None
    page_number: int | None = None
    text_content: str | None = None
    file_path: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class FileReport:
    """Per-file outcome, aggregated into the run report.

    'unchanged' is distinct from 'skipped': a skipped file produced nothing,
    while an unchanged one already has nodes from an earlier run that were
    deliberately left in place (along with their enrichment).
    """

    file_name: str
    media_type: str
    status: str  # 'ok' | 'unchanged' | 'skipped' | 'failed'
    node_count: int = 0
    detail: str | None = None


@dataclass
class IngestionReport:
    """Summary of one pipeline run."""

    case_id: str
    case_number: str
    files: list[FileReport] = field(default_factory=list)

    @property
    def total_nodes(self) -> int:
        return sum(f.node_count for f in self.files)

    @property
    def ok(self) -> list[FileReport]:
        return [f for f in self.files if f.status == "ok"]

    @property
    def failed(self) -> list[FileReport]:
        return [f for f in self.files if f.status == "failed"]

    @property
    def skipped(self) -> list[FileReport]:
        return [f for f in self.files if f.status == "skipped"]

    @property
    def unchanged(self) -> list[FileReport]:
        """Files left untouched because their bytes matched the last ingest."""
        return [f for f in self.files if f.status == "unchanged"]
