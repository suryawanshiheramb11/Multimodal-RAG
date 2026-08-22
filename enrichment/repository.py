"""Persistence for the enrichment phase.

Follows the project rule that all SQL lives in a repository: fixed column
lists, bound parameters, no interpolation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import Json, RealDictCursor

from .analyzers.base import EnrichmentResult, PendingNode

log = logging.getLogger(__name__)


class EnrichmentRepository:
    def __init__(self, conn: PgConnection) -> None:
        self._conn = conn

    def fetch_nodes(
        self, case_id: str, *, only_pending: bool = True, node_types: list[str] | None = None
    ) -> list[PendingNode]:
        """Load nodes for a case, joined to their source file.

        `only_pending` skips nodes already enriched, so a re-run resumes rather
        than repeating hours of model work.
        """
        clauses = ["f.case_id = %s"]
        params: list = [case_id]

        if only_pending:
            clauses.append("n.enriched_at IS NULL")
        if node_types:
            clauses.append("n.node_type = ANY(%s)")
            params.append(node_types)

        query = f"""
            SELECT n.id, n.node_type, n.source_file_id, n.start_time, n.end_time,
                   n.page_number, n.text_content, n.file_path, n.metadata,
                   f.file_path AS source_path, f.file_type AS source_type
            FROM evidence_node n
            JOIN source_file f ON f.id = n.source_file_id
            WHERE {" AND ".join(clauses)}
            ORDER BY n.source_file_id, n.page_number NULLS LAST, n.start_time NULLS LAST
        """  # noqa: S608 - clauses are literals chosen above, values stay bound

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [
            PendingNode(
                id=str(row["id"]),
                node_type=row["node_type"],
                source_file_id=str(row["source_file_id"]),
                source_path=Path(row["source_path"]),
                source_type=row["source_type"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                page_number=row["page_number"],
                text_content=row["text_content"],
                file_path=row["file_path"],
                metadata=row["metadata"] or {},
            )
            for row in rows
        ]

    def save(self, node: PendingNode, result: EnrichmentResult) -> None:
        """Write extracted features back onto the node.

        Metadata is merged rather than replaced so the ingestion phase's record
        (frame paths, codec, dimensions) survives enrichment.
        """
        merged = {**result.metadata}
        if result.skipped:
            merged["enrichment_skipped"] = result.skipped

        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evidence_node
                SET text_content    = COALESCE(%s, text_content),
                    text_embedding  = %s,
                    clip_embedding  = %s,
                    audio_embedding = %s,
                    metadata        = COALESCE(metadata, '{}'::jsonb) || %s::jsonb,
                    enriched_at     = now(),
                    enrichment_error = NULL
                WHERE id = %s
                """,
                (
                    result.text_content,
                    _vector(result.text_embedding),
                    _vector(result.clip_embedding),
                    _vector(result.audio_embedding),
                    json.dumps(merged, default=str),
                    node.id,
                ),
            )
        self._conn.commit()

    def mark_failed(self, node: PendingNode, error: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evidence_node
                SET enrichment_error = %s, enriched_at = now()
                WHERE id = %s
                """,
                (error[:2000], node.id),
            )
        self._conn.commit()

    def record_run(self, case_id: str, availability: dict[str, str], settings: dict) -> None:
        """Store which models were live for this run.

        Without it, a NULL embedding six weeks later is indistinguishable from
        a model that was quietly missing on the day.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO enrichment_run (case_id, model_availability, settings)
                VALUES (%s, %s, %s)
                """,
                (case_id, Json(availability), Json(settings)),
            )
        self._conn.commit()

    def coverage(self, case_id: str) -> dict:
        """Per-node-type counts of what actually got populated."""
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT n.node_type,
                       count(*)                                      AS total,
                       count(n.enriched_at)                          AS enriched,
                       count(n.text_content)                         AS with_text,
                       count(n.text_embedding)                       AS with_text_vec,
                       count(n.clip_embedding)                       AS with_clip_vec,
                       count(n.audio_embedding)                      AS with_audio_vec,
                       count(n.enrichment_error)                     AS failed
                FROM evidence_node n
                JOIN source_file f ON f.id = n.source_file_id
                WHERE f.case_id = %s
                GROUP BY n.node_type
                ORDER BY n.node_type
                """,
                (case_id,),
            )
            return {"by_type": [dict(r) for r in cur.fetchall()]}


def _vector(embedding: np.ndarray | None):
    """Adapt a numpy vector for pgvector, passing NULL straight through."""
    if embedding is None:
        return None
    return np.asarray(embedding, dtype=np.float32)
