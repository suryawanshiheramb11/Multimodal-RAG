"""All SQL for graph construction.

Follows the project rule that SQL lives only in repositories, with fixed
column lists and bound parameters. The similarity pass in particular is done
as one SQL statement using pgvector's `<=>` operator rather than pulling every
vector into Python: the database can compute and filter cosine distance for a
few hundred nodes far faster than a numpy loop, and it never has to move the
vectors over the wire at all.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import Json, RealDictCursor, execute_values

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextNode:
    id: str
    text_content: str
    metadata: dict


@dataclass(frozen=True)
class TimeWindow:
    id: str
    node_type: str
    source_file_id: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class FrameRef:
    """One image worth running face detection on, with its owning node."""

    evidence_node_id: str
    frame_path: Path


class GraphRepository:
    def __init__(self, conn: PgConnection) -> None:
        self._conn = conn

    # -- source data for extraction ----------------------------------------

    def fetch_text_nodes(self, case_id: str) -> list[TextNode]:
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT n.id, n.text_content, n.metadata
                FROM evidence_node n
                JOIN source_file f ON f.id = n.source_file_id
                WHERE f.case_id = %s AND n.text_content IS NOT NULL AND n.text_content <> ''
                """,
                (case_id,),
            )
            return [TextNode(str(r["id"]), r["text_content"], r["metadata"] or {}) for r in cur]

    def fetch_time_windows(self, case_id: str) -> list[TimeWindow]:
        """Nodes with a defined [start_time, end_time), for ALIGNS_WITH."""
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT n.id, n.node_type, n.source_file_id, n.start_time, n.end_time
                FROM evidence_node n
                JOIN source_file f ON f.id = n.source_file_id
                WHERE f.case_id = %s AND n.start_time IS NOT NULL AND n.end_time IS NOT NULL
                """,
                (case_id,),
            )
            return [
                TimeWindow(str(r["id"]), r["node_type"], str(r["source_file_id"]),
                           r["start_time"], r["end_time"])
                for r in cur
            ]

    def fetch_frames_for_faces(self, case_id: str) -> list[FrameRef]:
        """Every frame worth scanning for faces: video frames + standalone images."""
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT n.id, n.node_type, n.file_path, n.metadata
                FROM evidence_node n
                JOIN source_file f ON f.id = n.source_file_id
                WHERE f.case_id = %s AND n.node_type IN ('scene_segment', 'image')
                """,
                (case_id,),
            )
            rows = cur.fetchall()

        refs: list[FrameRef] = []
        for row in rows:
            if row["node_type"] == "image" and row["file_path"]:
                refs.append(FrameRef(str(row["id"]), Path(row["file_path"])))
                continue
            for frame in (row["metadata"] or {}).get("frames") or []:
                path = frame.get("path") if isinstance(frame, dict) else None
                if path:
                    refs.append(FrameRef(str(row["id"]), Path(path)))
        return refs

    # -- entities & mentions --------------------------------------------------

    def upsert_entity(
        self, case_id: str, entity_type: str, canonical_name: str,
        normalized_name: str, embedding: np.ndarray | None,
    ) -> str:
        """Insert or find the canonical entity, bumping its mention count.

        ON CONFLICT targets the (case, type, normalized_name) unique index,
        which is what makes "knife" mentioned in five nodes collapse to one row.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entity (case_id, entity_type, canonical_name, normalized_name,
                                     embedding, mention_count)
                VALUES (%s, %s, %s, %s, %s, 1)
                ON CONFLICT (case_id, entity_type, normalized_name) DO UPDATE
                    SET mention_count = entity.mention_count + 1
                RETURNING id
                """,
                (case_id, entity_type, canonical_name, normalized_name, embedding),
            )
            return str(cur.fetchone()[0])

    def add_mention(
        self, entity_id: str, evidence_node_id: str, mention_text: str,
        source: str, confidence: float | None = None,
    ) -> bool:
        """Insert a MENTIONS edge. Returns False if it already existed."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mention (entity_id, evidence_node_id, mention_text, source, confidence)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (entity_id, evidence_node_id) DO NOTHING
                RETURNING id
                """,
                (entity_id, evidence_node_id, mention_text, source, confidence),
            )
            return cur.fetchone() is not None

    def commit(self) -> None:
        self._conn.commit()

    def entities_mentioning_text(self, case_id: str, name: str) -> list[dict]:
        """Test/query helper: nodes that mention an entity by (fuzzy) name."""
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT e.canonical_name, e.entity_type, e.mention_count,
                       array_agg(DISTINCT m.evidence_node_id) AS node_ids
                FROM entity e
                JOIN mention m ON m.entity_id = e.id
                WHERE e.case_id = %s AND e.normalized_name LIKE %s
                GROUP BY e.id
                """,
                (case_id, f"%{name.lower()}%"),
            )
            return [dict(r) for r in cur.fetchall()]

    # -- temporal alignment ---------------------------------------------------

    def insert_alignment(
        self, case_id: str, node_a_id: str, node_b_id: str,
        alignment_type: str, score: float | None, metadata: dict | None = None,
    ) -> bool:
        """A node pair is stored once, in a canonical (lower id, higher id)
        order, so ALIGNS_WITH(a, b) and ALIGNS_WITH(b, a) are the same row."""
        first, second = sorted((node_a_id, node_b_id))
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO source_alignment
                    (case_id, node_a_id, node_b_id, alignment_type, score, metadata)
                SELECT %s, %s, %s, %s, %s, %s
                WHERE NOT EXISTS (
                    SELECT 1 FROM source_alignment
                    WHERE node_a_id = %s AND node_b_id = %s AND alignment_type = %s
                )
                RETURNING id
                """,
                (case_id, first, second, alignment_type, score, Json(metadata or {}),
                 first, second, alignment_type),
            )
            return cur.fetchone() is not None

    # -- similarity edges, computed entirely in SQL ----------------------------

    def insert_similarity_edges(
        self, case_id: str, threshold: float, max_nodes: int
    ) -> int:
        """SIMILAR_TO edges between every pair of image-bearing nodes whose CLIP
        cosine similarity exceeds `threshold`.

        `<=>` is pgvector's cosine *distance* (1 - cosine similarity), so the
        filter is `distance < 1 - threshold`. Capped by `max_nodes` so a huge
        case doesn't silently attempt an O(n^2) join.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FROM evidence_node n
                JOIN source_file f ON f.id = n.source_file_id
                WHERE f.case_id = %s AND n.clip_embedding IS NOT NULL
                """,
                (case_id,),
            )
            candidate_count = cur.fetchone()[0]
            if candidate_count > max_nodes:
                raise ValueError(
                    f"{candidate_count} image-bearing nodes exceeds the "
                    f"{max_nodes}-node similarity cap; raise max_nodes_for_similarity "
                    "or restrict the case"
                )

            cur.execute(
                """
                INSERT INTO source_alignment (case_id, node_a_id, node_b_id, alignment_type, score)
                SELECT %(case_id)s, a.id, b.id, 'SIMILAR_TO',
                       1 - (a.clip_embedding <=> b.clip_embedding)
                FROM evidence_node a
                JOIN evidence_node b ON b.id > a.id
                JOIN source_file fa ON fa.id = a.source_file_id
                JOIN source_file fb ON fb.id = b.source_file_id
                WHERE fa.case_id = %(case_id)s AND fb.case_id = %(case_id)s
                  AND a.clip_embedding IS NOT NULL AND b.clip_embedding IS NOT NULL
                  AND 1 - (a.clip_embedding <=> b.clip_embedding) > %(threshold)s
                  AND NOT EXISTS (
                      SELECT 1 FROM source_alignment sa
                      WHERE sa.node_a_id = a.id AND sa.node_b_id = b.id
                        AND sa.alignment_type = 'SIMILAR_TO'
                  )
                """,
                {"case_id": case_id, "threshold": threshold},
            )
            inserted = cur.rowcount
        self._conn.commit()
        return inserted

    # -- faces ------------------------------------------------------------------

    def insert_face_detections(self, case_id: str, rows: list[dict]) -> list[str]:
        """Bulk-insert detected faces, returning their new ids in order."""
        if not rows:
            return []
        with self._conn.cursor() as cur:
            ids = execute_values(
                cur,
                """
                INSERT INTO face_detection
                    (case_id, evidence_node_id, frame_path, bbox, confidence, embedding)
                VALUES %s
                RETURNING id
                """,
                [
                    (case_id, r["evidence_node_id"], str(r["frame_path"]),
                     Json(r["bbox"]), r["confidence"], r["embedding"])
                    for r in rows
                ],
                fetch=True,
            )
        self._conn.commit()
        return [str(row[0]) for row in ids]

    def fetch_face_embeddings(self, case_id: str) -> list[tuple[str, np.ndarray]]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, embedding FROM face_detection WHERE case_id = %s",
                (case_id,),
            )
            return [(str(row[0]), np.asarray(row[1], dtype=np.float32)) for row in cur.fetchall()]

    def create_face_cluster(
        self, case_id: str, representative_embedding: np.ndarray, face_count: int
    ) -> str:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_cluster (case_id, representative_embedding, face_count)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (case_id, representative_embedding, face_count),
            )
            return str(cur.fetchone()[0])

    def assign_faces_to_cluster(
        self, face_detection_ids: list[str], cluster_id: str | None
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE face_detection SET face_cluster_id = %s WHERE id = ANY(%s)",
                (cluster_id, face_detection_ids),
            )
        self._conn.commit()

    def clear_face_clusters(self, case_id: str) -> None:
        """Drop prior clustering so a re-run starts clean rather than piling
        duplicate clusters on top of old ones."""
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE face_detection SET face_cluster_id = NULL WHERE case_id = %s",
                (case_id,),
            )
            cur.execute("DELETE FROM face_cluster WHERE case_id = %s", (case_id,))
        self._conn.commit()

    # -- reporting --------------------------------------------------------------

    def graph_summary(self, case_id: str) -> dict:
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT count(*) AS n FROM entity WHERE case_id = %s", (case_id,))
            entities = cur.fetchone()["n"]

            cur.execute(
                """
                SELECT count(*) AS n FROM mention m
                JOIN entity e ON e.id = m.entity_id
                WHERE e.case_id = %s
                """,
                (case_id,),
            )
            mentions = cur.fetchone()["n"]

            cur.execute(
                """
                SELECT alignment_type, count(*) AS n FROM source_alignment
                WHERE case_id = %s GROUP BY alignment_type
                """,
                (case_id,),
            )
            alignments = {r["alignment_type"]: r["n"] for r in cur.fetchall()}

            cur.execute(
                "SELECT count(*) AS n FROM face_detection WHERE case_id = %s", (case_id,)
            )
            faces = cur.fetchone()["n"]

            cur.execute(
                "SELECT count(*) AS n FROM face_cluster WHERE case_id = %s", (case_id,)
            )
            face_clusters = cur.fetchone()["n"]

        return {
            "entities": entities,
            "mentions": mentions,
            "alignments": alignments,
            "faces_detected": faces,
            "face_clusters": face_clusters,
        }
