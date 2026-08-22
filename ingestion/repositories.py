"""Persistence layer.

All SQL lives here. Every statement uses a fixed column list with bound
parameters — no identifier or value is ever interpolated into a query string,
so evidence filenames and config metadata cannot reach the parser.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import RealDictCursor, execute_values

from .models import EvidenceNodeDraft, ScannedFile

log = logging.getLogger(__name__)


class CaseRepository:
    def __init__(self, conn: PgConnection) -> None:
        self._conn = conn

    def get_or_create(self, case_number: str, title: str | None, description: str | None) -> str:
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO "case" (case_number, title, description)
                VALUES (%s, %s, %s)
                ON CONFLICT (case_number) DO UPDATE
                    SET title = EXCLUDED.title,
                        description = EXCLUDED.description
                RETURNING id
                """,
                (case_number, title, description),
            )
            case_id = cur.fetchone()["id"]
        self._conn.commit()
        return str(case_id)


@dataclass(frozen=True)
class RegisteredSource:
    """A source file after registration, with its integrity verdict."""

    id: str
    is_new: bool
    #: True when this path was ingested before under a different hash — the
    #: file changed on disk between runs, which is a tampering signal worth
    #: surfacing rather than silently inserting a second row.
    content_changed: bool
    previous_sha256: str | None = None


class SourceFileRepository:
    def __init__(self, conn: PgConnection) -> None:
        self._conn = conn

    def register(self, case_id: str, source: ScannedFile) -> RegisteredSource:
        previous = self._previous_hash_for_path(case_id, str(source.path))
        content_changed = previous is not None and previous != source.sha256
        if content_changed:
            log.warning(
                "content of %s changed since the last ingest (%s -> %s)",
                source.file_name, previous[:12], source.sha256[:12],
            )

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO source_file (
                    case_id, file_path, file_name, file_type, sha256,
                    hash_algorithm, size_bytes, author, created_date,
                    declared_type, detected_mime, type_mismatch, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (case_id, sha256) DO UPDATE
                    SET file_path     = EXCLUDED.file_path,
                        file_name     = EXCLUDED.file_name,
                        file_type     = EXCLUDED.file_type,
                        size_bytes    = EXCLUDED.size_bytes,
                        author        = EXCLUDED.author,
                        created_date  = EXCLUDED.created_date,
                        declared_type = EXCLUDED.declared_type,
                        detected_mime = EXCLUDED.detected_mime,
                        type_mismatch = EXCLUDED.type_mismatch,
                        metadata      = EXCLUDED.metadata
                RETURNING id, (xmax = 0) AS inserted
                """,
                (
                    case_id,
                    str(source.path),
                    source.file_name,
                    source.media_type.value,
                    source.sha256,
                    "sha256",
                    source.size_bytes,
                    source.author,
                    source.created_date,
                    source.declared_type,
                    source.detected_mime,
                    source.type_mismatch,
                    json.dumps(source.metadata),
                ),
            )
            row = cur.fetchone()
        self._conn.commit()

        return RegisteredSource(
            id=str(row["id"]),
            is_new=bool(row["inserted"]),
            content_changed=content_changed,
            previous_sha256=previous if content_changed else None,
        )

    def _previous_hash_for_path(self, case_id: str, file_path: str) -> str | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT sha256 FROM source_file
                WHERE case_id = %s AND file_path = %s
                ORDER BY registered_at DESC
                LIMIT 1
                """,
                (case_id, file_path),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def count_for_case(self, case_id: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM source_file WHERE case_id = %s", (case_id,))
            return cur.fetchone()[0]


class EvidenceNodeRepository:
    def __init__(self, conn: PgConnection) -> None:
        self._conn = conn

    def replace_for_source(self, source_file_id: str, drafts: list[EvidenceNodeDraft]) -> int:
        """Write a source file's nodes, replacing any from a previous run.

        Re-ingesting the same case must be idempotent: without the delete, a
        second run would double every node and quietly corrupt downstream
        counts. Delete and insert share one transaction so a failure mid-write
        cannot leave the file with no nodes at all.
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM evidence_node WHERE source_file_id = %s",
                    (source_file_id,),
                )
                if drafts:
                    execute_values(
                        cur,
                        """
                        INSERT INTO evidence_node (
                            source_file_id, node_type, start_time, end_time,
                            page_number, text_content, file_path, metadata
                        ) VALUES %s
                        """,
                        [self._row(source_file_id, draft) for draft in drafts],
                    )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return len(drafts)

    @staticmethod
    def _row(source_file_id: str, draft: EvidenceNodeDraft) -> tuple:
        return (
            source_file_id,
            draft.node_type.value,
            draft.start_time,
            draft.end_time,
            draft.page_number,
            draft.text_content,
            draft.file_path,
            json.dumps(draft.metadata or {}),
        )

    def count_for_case(self, case_id: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FROM evidence_node n
                JOIN source_file f ON f.id = n.source_file_id
                WHERE f.case_id = %s
                """,
                (case_id,),
            )
            return cur.fetchone()[0]
