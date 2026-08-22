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


def _to_ndarray(value) -> np.ndarray:
    """Normalise a fetched vector column to a plain float32 ndarray.

    pgvector-python's psycopg2 cast (registered by ingestion.db.connect) hands
    back its own `Vector` wrapper rather than a bare ndarray, so a direct
    `np.asarray()` fails; `.to_numpy()` is the documented way to unwrap it.
    """
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value, dtype=np.float32)


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


@dataclass(frozen=True)
class ClipFrameNode:
    """A node with a stored CLIP *image* embedding — a page render, a video
    segment's representative frame, or a standalone image — as a candidate
    for cross-modal matching (REFERENCES, DESCRIBES)."""

    id: str
    node_type: str
    source_file_id: str
    embedding: np.ndarray
    page_number: int | None
    #: The timestamp of the specific frame the embedding was computed from,
    #: recovered from metadata.frames[] by matching representative_frame.
    #: None for node types with no meaningful timestamp (e.g. 'image', 'page').
    frame_timestamp: float | None
    frame_path: str | None


@dataclass(frozen=True)
class TranscriptNode:
    """A text-bearing audio/video node, as a candidate for DESCRIBES linking."""

    id: str
    node_type: str
    source_file_id: str
    text_content: str


@dataclass(frozen=True)
class ClaimNode:
    """A node whose text is a candidate for claim extraction."""

    id: str
    node_type: str
    text_content: str


@dataclass(frozen=True)
class ClaimRecord:
    """A node's extracted claim, with the vector used to pre-filter pairs."""

    id: str
    node_type: str
    claim: str
    text_embedding: np.ndarray | None


@dataclass(frozen=True)
class EventCandidate:
    """A timestamped node considered for timeline event grouping."""

    id: str
    node_type: str
    source_file_id: str
    start_time: float
    end_time: float
    text_content: str | None
    text_embedding: np.ndarray | None


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

    # -- cross-modal candidates -------------------------------------------------

    def fetch_clip_frame_nodes(self, case_id: str, node_types: list[str]) -> list[ClipFrameNode]:
        """Every node in `node_types` with a stored CLIP image embedding.

        Used as the candidate pool on the "image" side of REFERENCES and
        DESCRIBES: pages and standalone images embed their own file directly,
        a scene_segment's clip_embedding is its representative frame, whose
        timestamp is recovered here from metadata.frames[] rather than
        re-decoded from the video.
        """
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT n.id, n.node_type, n.source_file_id, n.page_number, n.clip_embedding,
                       n.file_path,
                       n.metadata->>'representative_frame' AS representative_frame,
                       (
                           SELECT (f->>'timestamp')::float
                           FROM jsonb_array_elements(n.metadata->'frames') f
                           WHERE f->>'path' = n.metadata->>'representative_frame'
                           LIMIT 1
                       ) AS frame_timestamp
                FROM evidence_node n
                JOIN source_file f ON f.id = n.source_file_id
                WHERE f.case_id = %s AND n.node_type = ANY(%s) AND n.clip_embedding IS NOT NULL
                """,
                (case_id, node_types),
            )
            return [
                ClipFrameNode(
                    id=str(r["id"]), node_type=r["node_type"], source_file_id=str(r["source_file_id"]),
                    embedding=_to_ndarray(r["clip_embedding"]), page_number=r["page_number"],
                    frame_timestamp=r["frame_timestamp"],
                    frame_path=r["representative_frame"] or r["file_path"],
                )
                for r in cur.fetchall()
            ]

    def fetch_transcript_nodes(self, case_id: str) -> list[TranscriptNode]:
        """Audio/video nodes with text — candidates for the "spoken" side of
        DESCRIBES. Restricted to audio-bearing node types: a pdf page's text
        is document text, not a transcript, and isn't what step 4 means by it.
        """
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT n.id, n.node_type, n.source_file_id, n.text_content
                FROM evidence_node n
                JOIN source_file f ON f.id = n.source_file_id
                WHERE f.case_id = %s AND n.node_type IN ('scene_segment', 'audio_track')
                  AND n.text_content IS NOT NULL AND n.text_content <> ''
                """,
                (case_id,),
            )
            return [
                TranscriptNode(str(r["id"]), r["node_type"], str(r["source_file_id"]), r["text_content"])
                for r in cur.fetchall()
            ]

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
        """Test/query helper: nodes that mention an entity by (fuzzy) name.

        Cast to text before aggregating: psycopg2 (without register_uuid())
        cannot parse a uuid[] array back into a Python list and hands back the
        raw "{...}" literal instead, whereas text[] is always parsed.
        """
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT e.canonical_name, e.entity_type, e.mention_count,
                       array_agg(DISTINCT m.evidence_node_id::text) AS node_ids
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

    def insert_document_video_references(
        self, case_id: str, threshold: float, max_nodes: int
    ) -> int:
        """REFERENCES edges: a pdf page's rendered image and a video segment's
        representative frame that are visually the same thing.

        Both sides already have their CLIP image vector stored, so — like
        SIMILAR_TO — this is one SQL join rather than a Python loop. The
        matching frame's timestamp is recovered from the segment's
        metadata.frames[] in the same subquery `fetch_clip_frame_nodes` uses.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FROM evidence_node n
                JOIN source_file f ON f.id = n.source_file_id
                WHERE f.case_id = %s AND n.node_type IN ('page', 'scene_segment')
                  AND n.clip_embedding IS NOT NULL
                """,
                (case_id,),
            )
            candidate_count = cur.fetchone()[0]
            if candidate_count > max_nodes:
                raise ValueError(
                    f"{candidate_count} page/scene_segment nodes exceeds the "
                    f"{max_nodes}-node cross-modal cap; raise max_nodes_for_cross_modal "
                    "or restrict the case"
                )

            cur.execute(
                """
                INSERT INTO source_alignment (case_id, node_a_id, node_b_id, alignment_type, score, metadata)
                SELECT %(case_id)s, p.id, s.id, 'REFERENCES',
                       1 - (p.clip_embedding <=> s.clip_embedding),
                       jsonb_build_object(
                           'cosine_similarity', round((1 - (p.clip_embedding <=> s.clip_embedding))::numeric, 4),
                           'page_number', p.page_number,
                           'matched_frame_path', s.metadata->>'representative_frame',
                           'matched_frame_timestamp', (
                               SELECT (fr->>'timestamp')::float
                               FROM jsonb_array_elements(s.metadata->'frames') fr
                               WHERE fr->>'path' = s.metadata->>'representative_frame'
                               LIMIT 1
                           )
                       )
                FROM evidence_node p
                JOIN evidence_node s ON true
                JOIN source_file fp ON fp.id = p.source_file_id
                JOIN source_file fs ON fs.id = s.source_file_id
                WHERE fp.case_id = %(case_id)s AND fs.case_id = %(case_id)s
                  AND p.node_type = 'page' AND s.node_type = 'scene_segment'
                  AND p.clip_embedding IS NOT NULL AND s.clip_embedding IS NOT NULL
                  AND 1 - (p.clip_embedding <=> s.clip_embedding) > %(threshold)s
                  AND NOT EXISTS (
                      SELECT 1 FROM source_alignment sa
                      WHERE sa.node_a_id = p.id AND sa.node_b_id = s.id
                        AND sa.alignment_type = 'REFERENCES'
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
            return [(str(row[0]), _to_ndarray(row[1])) for row in cur.fetchall()]

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
                # ANY(%s) infers text[] from a Python list of strings; an
                # explicit cast is required to compare against a uuid column.
                "UPDATE face_detection SET face_cluster_id = %s WHERE id = ANY(%s::uuid[])",
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

    # -- timeline events ----------------------------------------------------------

    def fetch_event_candidates(self, case_id: str) -> list[EventCandidate]:
        """Every timestamped, text-bearing node — the pool timeline grouping
        draws from. Untimed nodes (a standalone pdf page) have no [start, end)
        window to group by and are left out."""
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT n.id, n.node_type, n.source_file_id, n.start_time, n.end_time,
                       n.text_content, n.text_embedding
                FROM evidence_node n
                JOIN source_file f ON f.id = n.source_file_id
                WHERE f.case_id = %s AND n.start_time IS NOT NULL AND n.end_time IS NOT NULL
                """,
                (case_id,),
            )
            return [
                EventCandidate(
                    id=str(r["id"]), node_type=r["node_type"], source_file_id=str(r["source_file_id"]),
                    start_time=r["start_time"], end_time=r["end_time"], text_content=r["text_content"],
                    text_embedding=_to_ndarray(r["text_embedding"]) if r["text_embedding"] is not None else None,
                )
                for r in cur.fetchall()
            ]

    def fetch_entities_by_node(self, case_id: str) -> dict[str, set[str]]:
        """node_id -> the set of entity ids it mentions, for "share an entity"
        grouping."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.evidence_node_id, m.entity_id
                FROM mention m
                JOIN entity e ON e.id = m.entity_id
                WHERE e.case_id = %s
                """,
                (case_id,),
            )
            by_node: dict[str, set[str]] = {}
            for node_id, entity_id in cur.fetchall():
                by_node.setdefault(str(node_id), set()).add(str(entity_id))
            return by_node

    def clear_timeline_events(self, case_id: str) -> None:
        """Drop prior events so a re-run regroups cleanly rather than piling
        duplicate events on top of old ones (ON DELETE CASCADE also removes
        their timeline_event_link rows)."""
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM timeline_event WHERE case_id = %s", (case_id,))
        self._conn.commit()

    def insert_timeline_event(
        self, case_id: str, description: str | None, start_time: float, end_time: float,
        node_ids: list[str], metadata: dict | None = None,
    ) -> str:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO timeline_event (case_id, description, start_time, end_time, node_ids, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (case_id, description, start_time, end_time, Json(node_ids), Json(metadata or {})),
            )
            return str(cur.fetchone()[0])

    def link_node_to_event(self, timeline_event_id: str, evidence_node_id: str) -> bool:
        """Insert one SAME_EVENT edge. Returns False if it already existed."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO timeline_event_link (timeline_event_id, evidence_node_id)
                VALUES (%s, %s)
                ON CONFLICT (timeline_event_id, evidence_node_id) DO NOTHING
                RETURNING id
                """,
                (timeline_event_id, evidence_node_id),
            )
            return cur.fetchone() is not None

    # -- claims -------------------------------------------------------------------

    def fetch_claim_candidates(
        self, case_id: str, node_types: list[str], only_pending: bool = True
    ) -> list[ClaimNode]:
        """Text-bearing nodes of the given types awaiting claim extraction.

        `only_pending` skips nodes already attempted (claim_extracted_at set),
        so a re-run costs nothing for work already done — the same resume
        contract phase 2 gives via `enriched_at`.
        """
        clauses = [
            "f.case_id = %s",
            "n.node_type = ANY(%s)",
            "n.text_content IS NOT NULL",
            "n.text_content <> ''",
        ]
        params: list = [case_id, node_types]
        if only_pending:
            clauses.append("n.claim_extracted_at IS NULL")

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT n.id, n.node_type, n.text_content
                FROM evidence_node n
                JOIN source_file f ON f.id = n.source_file_id
                WHERE {' AND '.join(clauses)}
                """,  # noqa: S608 - clauses are literals; every value is bound
                params,
            )
            return [ClaimNode(str(r["id"]), r["node_type"], r["text_content"]) for r in cur.fetchall()]

    def store_claim(self, node_id: str, claim: str | None) -> None:
        """Record the extracted claim (or that there wasn't one) on the node."""
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE evidence_node SET claim = %s, claim_extracted_at = now() WHERE id = %s",
                (claim, node_id),
            )

    def fetch_claims(self, case_id: str) -> dict[str, ClaimRecord]:
        """Every node in the case that has a claim, keyed by node id."""
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT n.id, n.node_type, n.claim, n.text_embedding
                FROM evidence_node n
                JOIN source_file f ON f.id = n.source_file_id
                WHERE f.case_id = %s AND n.claim IS NOT NULL AND n.claim <> ''
                """,
                (case_id,),
            )
            return {
                str(r["id"]): ClaimRecord(
                    id=str(r["id"]), node_type=r["node_type"], claim=r["claim"],
                    text_embedding=(
                        _to_ndarray(r["text_embedding"]) if r["text_embedding"] is not None else None
                    ),
                )
                for r in cur.fetchall()
            }

    # -- contradiction candidate sources -------------------------------------------

    def fetch_entity_node_groups(self, case_id: str) -> list[tuple[str, str, list[str]]]:
        """(entity_id, canonical_name, node_ids) for entities mentioned by two
        or more nodes — a single-node entity has nothing to compare against."""
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT e.id, e.canonical_name,
                       array_agg(DISTINCT m.evidence_node_id::text) AS node_ids
                FROM entity e
                JOIN mention m ON m.entity_id = e.id
                WHERE e.case_id = %s
                GROUP BY e.id
                HAVING count(DISTINCT m.evidence_node_id) >= 2
                """,
                (case_id,),
            )
            return [(str(r["id"]), r["canonical_name"], list(r["node_ids"])) for r in cur.fetchall()]

    def fetch_alignment_pairs(
        self, case_id: str, alignment_types: list[str]
    ) -> list[tuple[str, str, str]]:
        """(node_a_id, node_b_id, alignment_type) for the given edge types."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT node_a_id, node_b_id, alignment_type FROM source_alignment
                WHERE case_id = %s AND alignment_type = ANY(%s)
                """,
                (case_id, alignment_types),
            )
            return [(str(a), str(b), t) for a, b, t in cur.fetchall()]

    def fetch_event_node_groups(self, case_id: str) -> list[tuple[str, list[str]]]:
        """(timeline_event_id, node_ids) for events grouping two or more nodes."""
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT te.id, array_agg(DISTINCT tel.evidence_node_id::text) AS node_ids
                FROM timeline_event te
                JOIN timeline_event_link tel ON tel.timeline_event_id = te.id
                WHERE te.case_id = %s
                GROUP BY te.id
                HAVING count(DISTINCT tel.evidence_node_id) >= 2
                """,
                (case_id,),
            )
            return [(str(r["id"]), list(r["node_ids"])) for r in cur.fetchall()]

    # -- claim relationships --------------------------------------------------------

    def clear_claim_relationships(self, case_id: str) -> None:
        """Drop prior CONTRADICTS/CORROBORATES edges so a re-run re-judges
        rather than leaving stale verdicts from an earlier set of claims.

        Scoped by relationship_type: identity-level rows in this table belong
        to a different phase and must survive.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM relationship
                WHERE case_id = %s AND relationship_type IN ('CONTRADICTS', 'CORROBORATES')
                """,
                (case_id,),
            )
        self._conn.commit()

    def insert_claim_relationship(
        self, case_id: str, subject_node_id: str, object_node_id: str,
        relationship_type: str, confidence: float | None, explanation: str | None,
        metadata: dict | None = None,
    ) -> bool:
        """Store one CONTRADICTS/CORROBORATES edge between two nodes.

        Stored in canonical (lower id, higher id) order — contradiction is
        symmetric, so A-contradicts-B and B-contradicts-A are one finding, and
        the unique index enforces that. Returns False if it already existed.
        """
        first, second = sorted((subject_node_id, object_node_id))
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO relationship
                    (case_id, subject_node_id, object_node_id, relationship_type,
                     confidence, explanation, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                -- uq_relationship_node_pair is a *partial* index, so its
                -- predicate has to be restated here: Postgres will not infer a
                -- partial index from the column list alone.
                ON CONFLICT (case_id, subject_node_id, object_node_id, relationship_type)
                    WHERE subject_node_id IS NOT NULL AND object_node_id IS NOT NULL
                    DO NOTHING
                RETURNING id
                """,
                (case_id, first, second, relationship_type, confidence, explanation,
                 Json(metadata or {})),
            )
            return cur.fetchone() is not None

    # -- evidence pack --------------------------------------------------------------

    def fetch_evidence_pack(
        self, case_id: str, entity_name: str | None = None,
        start_time: float | None = None, end_time: float | None = None,
    ) -> list[dict]:
        """Evidence nodes matching an entity and/or time window, each carrying
        the CONTRADICTS/CORROBORATES edges it participates in.

        This is what lets the final pack *highlight* disagreements: a caller
        gets the nodes and their conflicts in one read rather than fetching
        evidence and then discovering, separately, that two pieces of it
        disagree. Filters combine — passing both narrows to nodes that satisfy
        each; passing neither returns the whole case.
        """
        clauses = ["f.case_id = %(case_id)s"]
        params: dict = {"case_id": case_id}

        if entity_name:
            clauses.append(
                """EXISTS (
                    SELECT 1 FROM mention m
                    JOIN entity e ON e.id = m.entity_id
                    WHERE m.evidence_node_id = n.id AND e.case_id = %(case_id)s
                      AND e.normalized_name LIKE %(entity_pattern)s
                )"""
            )
            params["entity_pattern"] = f"%{entity_name.lower()}%"

        # Half-open overlap against the requested window, so a node is included
        # when any part of it falls inside — an untimed node (a pdf page) has
        # no window and is excluded only when a window was actually asked for.
        if start_time is not None:
            clauses.append("n.end_time IS NOT NULL AND n.end_time >= %(start_time)s")
            params["start_time"] = start_time
        if end_time is not None:
            clauses.append("n.start_time IS NOT NULL AND n.start_time <= %(end_time)s")
            params["end_time"] = end_time

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT n.id, n.node_type, n.start_time, n.end_time, n.page_number,
                       n.claim, n.text_content, f.file_name
                FROM evidence_node n
                JOIN source_file f ON f.id = n.source_file_id
                WHERE {' AND '.join(clauses)}
                ORDER BY n.start_time NULLS LAST, n.page_number NULLS LAST
                """,  # noqa: S608 - clauses are literals; every value is bound
                params,
            )
            nodes = [
                {
                    "node_id": str(r["id"]), "node_type": r["node_type"],
                    "start_time": r["start_time"], "end_time": r["end_time"],
                    "page_number": r["page_number"], "claim": r["claim"],
                    "text_content": r["text_content"], "file_name": r["file_name"],
                    "relations": [],
                }
                for r in cur.fetchall()
            ]
            if not nodes:
                return []

            by_id = {node["node_id"]: node for node in nodes}
            cur.execute(
                """
                SELECT r.subject_node_id, r.object_node_id, r.relationship_type,
                       r.confidence, r.explanation,
                       sn.node_type AS subject_node_type, on_.node_type AS object_node_type
                FROM relationship r
                JOIN evidence_node sn ON sn.id = r.subject_node_id
                JOIN evidence_node on_ ON on_.id = r.object_node_id
                WHERE r.case_id = %s
                  AND r.relationship_type IN ('CONTRADICTS', 'CORROBORATES')
                  AND (r.subject_node_id = ANY(%s::uuid[]) OR r.object_node_id = ANY(%s::uuid[]))
                ORDER BY r.relationship_type, r.confidence DESC NULLS LAST
                """,
                (case_id, list(by_id), list(by_id)),
            )
            for row in cur.fetchall():
                subject, obj = str(row["subject_node_id"]), str(row["object_node_id"])
                # A pair where both ends are in the result set is attached to
                # each end, so either node shows the disagreement.
                for near, far, far_type in (
                    (subject, obj, row["object_node_type"]),
                    (obj, subject, row["subject_node_type"]),
                ):
                    if near in by_id:
                        by_id[near]["relations"].append({
                            "relationship_type": row["relationship_type"],
                            "other_node_id": far,
                            "other_node_type": far_type,
                            "confidence": row["confidence"],
                            "explanation": row["explanation"],
                        })

        return nodes

    # -- edge introspection -------------------------------------------------------

    def fetch_edges_for_node(self, case_id: str, node_id: str) -> list[dict]:
        """Every edge touching one node, across both edge tables — the query a
        reviewer (or a test) runs to answer "what is this node connected to."
        """
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT alignment_type,
                       CASE WHEN node_a_id = %(node_id)s THEN node_b_id ELSE node_a_id END AS other_node_id,
                       score, metadata
                FROM source_alignment
                WHERE case_id = %(case_id)s AND (node_a_id = %(node_id)s OR node_b_id = %(node_id)s)
                """,
                {"case_id": case_id, "node_id": node_id},
            )
            edges = [
                {
                    "alignment_type": r["alignment_type"], "other_node_id": str(r["other_node_id"]),
                    "score": r["score"], "metadata": r["metadata"] or {},
                }
                for r in cur.fetchall()
            ]

            cur.execute(
                """
                SELECT te.id AS timeline_event_id, te.description
                FROM timeline_event_link tel
                JOIN timeline_event te ON te.id = tel.timeline_event_id
                WHERE te.case_id = %s AND tel.evidence_node_id = %s
                """,
                (case_id, node_id),
            )
            edges.extend(
                {
                    "alignment_type": "SAME_EVENT", "other_node_id": str(r["timeline_event_id"]),
                    "score": None, "metadata": {"description": r["description"]},
                }
                for r in cur.fetchall()
            )

            cur.execute(
                """
                SELECT relationship_type,
                       CASE WHEN subject_node_id = %(node_id)s THEN object_node_id
                            ELSE subject_node_id END AS other_node_id,
                       confidence, explanation
                FROM relationship
                WHERE case_id = %(case_id)s
                  AND relationship_type IN ('CONTRADICTS', 'CORROBORATES')
                  AND (subject_node_id = %(node_id)s OR object_node_id = %(node_id)s)
                """,
                {"case_id": case_id, "node_id": node_id},
            )
            edges.extend(
                {
                    "alignment_type": r["relationship_type"],
                    "other_node_id": str(r["other_node_id"]),
                    "score": r["confidence"],
                    "metadata": {"explanation": r["explanation"]},
                }
                for r in cur.fetchall()
            )
        return edges

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

            cur.execute(
                "SELECT count(*) AS n FROM timeline_event WHERE case_id = %s", (case_id,)
            )
            timeline_events = cur.fetchone()["n"]

            cur.execute(
                """
                SELECT relationship_type, count(*) AS n FROM relationship
                WHERE case_id = %s AND relationship_type IN ('CONTRADICTS', 'CORROBORATES')
                GROUP BY relationship_type
                """,
                (case_id,),
            )
            relationships = {r["relationship_type"]: r["n"] for r in cur.fetchall()}

            cur.execute(
                """
                SELECT count(n.claim) AS n FROM evidence_node n
                JOIN source_file f ON f.id = n.source_file_id
                WHERE f.case_id = %s
                """,
                (case_id,),
            )
            claims = cur.fetchone()["n"]

        return {
            "entities": entities,
            "mentions": mentions,
            "alignments": alignments,
            "faces_detected": faces,
            "face_clusters": face_clusters,
            "timeline_events": timeline_events,
            "claims": claims,
            "relationships": relationships,
        }

    def audio_embedding_coverage(self, case_id: str) -> dict:
        """How many audio-bearing nodes actually have an AST embedding stored.

        Step 3 of phase 4 ("cross-source alignment prep") does not estimate
        offsets yet — that needs a later phase's clock-alignment work — it
        only needs to confirm the embeddings a future offset estimator will
        consume are actually present and indexed (the HNSW index in the
        schema handles "indexed"; this handles "present").
        """
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT n.node_type,
                       count(*) AS total,
                       count(n.audio_embedding) AS with_embedding
                FROM evidence_node n
                JOIN source_file f ON f.id = n.source_file_id
                WHERE f.case_id = %s AND n.node_type IN ('scene_segment', 'audio_track')
                GROUP BY n.node_type
                """,
                (case_id,),
            )
            return {r["node_type"]: {"total": r["total"], "with_embedding": r["with_embedding"]}
                    for r in cur.fetchall()}

    # ========================================================================
    # Phase 6: Multi-source timeline synchronization
    # ========================================================================

    def fetch_audio_segments_by_source(self, source_file_id: str) -> list[tuple[float, float, np.ndarray]]:
        """Fetch all audio segments (with AST embeddings) for one source.

        Returns: list of (start_time, end_time, embedding) tuples for scene_segment and audio_track nodes.
        """
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT n.start_time, n.end_time, n.audio_embedding
                FROM evidence_node n
                WHERE n.source_file_id = %s
                  AND n.node_type IN ('scene_segment', 'audio_track')
                  AND n.audio_embedding IS NOT NULL
                ORDER BY n.start_time
                """,
                (source_file_id,),
            )
            return [
                (r["start_time"], r["end_time"], _to_ndarray(r["audio_embedding"]))
                for r in cur.fetchall()
            ]

    def fetch_video_frames_by_source(self, source_file_id: str) -> list[tuple[float, np.ndarray]]:
        """Fetch all video frames (with CLIP embeddings) for one source.

        Returns: list of (timestamp, embedding) tuples for scene_segment nodes with representative_frame.
        """
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT n.start_time,
                       (n.metadata->>'representative_frame_timestamp')::float AS frame_timestamp,
                       n.clip_embedding
                FROM evidence_node n
                WHERE n.source_file_id = %s
                  AND n.node_type = 'scene_segment'
                  AND n.clip_embedding IS NOT NULL
                  AND n.metadata->>'representative_frame_timestamp' IS NOT NULL
                ORDER BY n.start_time
                """,
                (source_file_id,),
            )
            rows = cur.fetchall()
            # Use frame_timestamp if available; fall back to start_time
            return [
                (r["frame_timestamp"] or r["start_time"], _to_ndarray(r["clip_embedding"]))
                for r in rows
            ]

    def fetch_face_appearances_by_source(self, source_file_id: str) -> dict[str, list[float]]:
        """Fetch face cluster appearances in one source.

        Returns: dict mapping face_cluster_id -> list of timestamps.
        """
        appearances: dict[str, list[float]] = {}
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT f.face_cluster_id, n.start_time
                FROM face_detection f
                JOIN evidence_node n ON n.id = f.evidence_node_id
                WHERE n.source_file_id = %s
                  AND f.face_cluster_id IS NOT NULL
                ORDER BY f.face_cluster_id, n.start_time
                """,
                (source_file_id,),
            )
            for r in cur.fetchall():
                cluster_id = str(r["face_cluster_id"])
                appearances.setdefault(cluster_id, []).append(r["start_time"])
        return appearances

    def insert_source_offset(
        self,
        case_id: str,
        source_a_id: str,
        source_b_id: str,
        offset_seconds: float,
        confidence: float,
        method: str,
        anchor_count: int,
        metadata: dict | None = None,
    ) -> None:
        """Store or replace a source-to-source offset estimate.

        offset_seconds = time_in_source_b - time_in_source_a. Negative if B started before A.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO source_offset (case_id, source_a_id, source_b_id, offset_seconds,
                                          confidence, method, anchor_count, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (case_id, source_a_id, source_b_id) DO UPDATE
                SET offset_seconds = %s, confidence = %s, method = %s, anchor_count = %s,
                    metadata = %s
                """,
                (
                    case_id, source_a_id, source_b_id, offset_seconds, confidence, method,
                    anchor_count, Json(metadata or {}),
                    offset_seconds, confidence, method, anchor_count, Json(metadata or {}),
                ),
            )
        self._conn.commit()

    def update_evidence_case_time(self, case_id: str, reference_source_id: str) -> int:
        """Compute case_time for all evidence nodes based on source offsets.

        case_time = start_time + offset_from_reference. Reference source gets offset 0.
        Returns the number of nodes updated.
        """
        # First, zero out the reference source
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evidence_node SET case_time = start_time
                WHERE source_file_id = %s
                  AND source_file_id = (SELECT id FROM source_file WHERE id = %s)
                """,
                (reference_source_id, reference_source_id),
            )
            reference_updated = cur.rowcount

            # For other sources, apply their offset to the reference
            cur.execute(
                """
                UPDATE evidence_node n SET case_time = n.start_time + so.offset_seconds
                FROM source_offset so, source_file sf
                WHERE n.source_file_id = so.source_b_id
                  AND so.case_id = %s
                  AND so.source_a_id = %s
                  AND sf.id = so.source_b_id
                """,
                (case_id, reference_source_id),
            )
            other_updated = cur.rowcount
        self._conn.commit()
        return reference_updated + other_updated

    def query_unified_timeline(self, case_id: str, limit: int = 1000) -> list[dict]:
        """Query evidence sorted by case_time across all sources.

        Returns: list of dicts with node_id, node_type, case_time, source_file_name, text snippet.
        """
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT n.id, n.node_type, n.case_time, sf.file_name,
                       left(n.text_content, 100) AS text_snippet
                FROM evidence_node n
                JOIN source_file sf ON sf.id = n.source_file_id
                WHERE sf.case_id = %s AND n.case_time IS NOT NULL
                ORDER BY n.case_time, sf.file_name
                LIMIT %s
                """,
                (case_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]
