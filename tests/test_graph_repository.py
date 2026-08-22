"""Integration tests for graph SQL against a real Postgres.

Skipped automatically when no database is reachable, matching
test_repositories.py's pattern for phase 1.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest
from psycopg2.extras import Json

from graph.repository import GraphRepository
from ingestion.config import DatabaseSettings
from ingestion.db import apply_schema, connect
from ingestion.errors import IngestionError
from ingestion.models import MediaType, ScannedFile
from ingestion.repositories import CaseRepository, SourceFileRepository


@pytest.fixture(scope="module")
def conn():
    settings = DatabaseSettings()
    try:
        with connect(settings) as connection:
            apply_schema(connection)
            yield connection
    except IngestionError as exc:
        pytest.skip(f"no database available: {exc}")


@pytest.fixture
def case_id(conn):
    number = f"GRAPH-TEST-{uuid.uuid4()}"
    created = CaseRepository(conn).get_or_create(number, "temp", "graph integration test")
    yield created
    with conn.cursor() as cur:
        cur.execute('DELETE FROM "case" WHERE id = %s', (created,))
    conn.commit()


@pytest.fixture
def source_file_id(conn, case_id):
    source = ScannedFile(
        path=Path(f"/evidence/{uuid.uuid4()}.mp4"),
        file_name="clip.mp4",
        media_type=MediaType.VIDEO,
        sha256=uuid.uuid4().hex * 2,
        size_bytes=100,
    )
    return SourceFileRepository(conn).register(case_id, source).id


def make_node(conn, source_file_id, node_type="scene_segment", clip_embedding=None,
              start=None, end=None, text_content=None, page_number=None,
              text_embedding=None, audio_embedding=None, metadata=None):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO evidence_node
                (source_file_id, node_type, clip_embedding, start_time, end_time,
                 text_content, page_number, text_embedding, audio_embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (source_file_id, node_type, clip_embedding, start, end,
             text_content, page_number, text_embedding, audio_embedding, Json(metadata or {})),
        )
        node_id = str(cur.fetchone()[0])
    conn.commit()
    return node_id


def _unit(vector: list[float], dim: int = 512) -> np.ndarray:
    """A unit vector padded to `dim`, matching the clip_embedding/face
    embedding column width — pgvector rejects a mismatched dimension outright."""
    padded = list(vector) + [0.0] * (dim - len(vector))
    array = np.asarray(padded, dtype=np.float32)
    return array / np.linalg.norm(array)


class TestEntityDedup:
    def test_same_normalized_name_collapses_to_one_row(self, conn, case_id):
        repo = GraphRepository(conn)

        first = repo.upsert_entity(case_id, "weapon", "Knife", "knife", None)
        second = repo.upsert_entity(case_id, "weapon", "knife", "knife", None)
        repo.commit()

        assert first == second
        with conn.cursor() as cur:
            cur.execute("SELECT mention_count FROM entity WHERE id = %s", (first,))
            assert cur.fetchone()[0] == 2

    def test_different_types_stay_separate(self, conn, case_id):
        repo = GraphRepository(conn)
        a = repo.upsert_entity(case_id, "person", "Jordan", "jordan", None)
        b = repo.upsert_entity(case_id, "location", "Jordan", "jordan", None)
        repo.commit()
        assert a != b


class TestMentions:
    def test_duplicate_mention_is_not_inserted_twice(self, conn, case_id, source_file_id):
        repo = GraphRepository(conn)
        node_id = make_node(conn, source_file_id)
        entity_id = repo.upsert_entity(case_id, "weapon", "knife", "knife", None)

        first = repo.add_mention(entity_id, node_id, "knife", "llm_extraction")
        second = repo.add_mention(entity_id, node_id, "knife", "llm_extraction")
        repo.commit()

        assert first is True
        assert second is False

    def test_entities_mentioning_text_finds_the_node(self, conn, case_id, source_file_id):
        repo = GraphRepository(conn)
        node_id = make_node(conn, source_file_id)
        entity_id = repo.upsert_entity(case_id, "weapon", "hunting knife", "hunting knife", None)
        repo.add_mention(entity_id, node_id, "hunting knife", "llm_extraction")
        repo.commit()

        results = repo.entities_mentioning_text(case_id, "knife")

        assert len(results) == 1
        assert results[0]["canonical_name"] == "hunting knife"
        assert node_id in [str(n) for n in results[0]["node_ids"]]


class TestSimilarityEdges:
    def test_similar_nodes_get_an_edge_dissimilar_do_not(self, conn, case_id, source_file_id):
        close_a = make_node(conn, source_file_id, clip_embedding=_unit([1.0, 0.0, 0.0]))
        close_b = make_node(conn, source_file_id, clip_embedding=_unit([0.999, 0.001, 0.0]))
        far = make_node(conn, source_file_id, clip_embedding=_unit([0.0, 1.0, 0.0]))

        repo = GraphRepository(conn)
        inserted = repo.insert_similarity_edges(case_id, threshold=0.9, max_nodes=100)

        assert inserted == 1
        with conn.cursor() as cur:
            cur.execute(
                "SELECT node_a_id, node_b_id FROM source_alignment "
                "WHERE case_id = %s AND alignment_type = 'SIMILAR_TO'",
                (case_id,),
            )
            pairs = {tuple(sorted((str(a), str(b)))) for a, b in cur.fetchall()}
        assert tuple(sorted((close_a, close_b))) in pairs
        assert not any(far in pair for pair in pairs)

    def test_rerun_does_not_duplicate_edges(self, conn, case_id, source_file_id):
        make_node(conn, source_file_id, clip_embedding=_unit([1.0, 0.0, 0.0]))
        make_node(conn, source_file_id, clip_embedding=_unit([0.999, 0.001, 0.0]))

        repo = GraphRepository(conn)
        repo.insert_similarity_edges(case_id, threshold=0.9, max_nodes=100)
        second_run = repo.insert_similarity_edges(case_id, threshold=0.9, max_nodes=100)

        assert second_run == 0

    def test_exceeding_the_node_cap_raises(self, conn, case_id, source_file_id):
        for _ in range(3):
            make_node(conn, source_file_id, clip_embedding=_unit([1.0, 0.0, 0.0]))

        repo = GraphRepository(conn)
        with pytest.raises(ValueError, match="exceeds"):
            repo.insert_similarity_edges(case_id, threshold=0.9, max_nodes=2)


class TestAlignmentEdges:
    def test_alignment_stored_in_canonical_order(self, conn, case_id, source_file_id):
        a = make_node(conn, source_file_id, start=0.0, end=5.0)
        b = make_node(conn, source_file_id, "audio_track", start=0.0, end=30.0)

        repo = GraphRepository(conn)
        # Insert with arguments in reverse id order; the row must still be
        # stored once, not twice, regardless of call order.
        higher, lower = sorted((a, b), reverse=True)
        created = repo.insert_alignment(case_id, higher, lower, "ALIGNS_WITH", 5.0)
        duplicate = repo.insert_alignment(case_id, lower, higher, "ALIGNS_WITH", 5.0)
        repo.commit()

        assert created is True
        assert duplicate is False


class TestFaceDetectionAndClustering:
    def test_insert_and_fetch_round_trips_embeddings(self, conn, case_id, source_file_id):
        node_id = make_node(conn, source_file_id, "image")
        repo = GraphRepository(conn)

        ids = repo.insert_face_detections(case_id, [{
            "evidence_node_id": node_id,
            "frame_path": "/data/frames/f1.jpg",
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "confidence": 0.95,
            "embedding": _unit([1.0, 0.0] + [0.0] * 510),
        }])

        assert len(ids) == 1
        rows = repo.fetch_face_embeddings(case_id)
        assert len(rows) == 1
        assert rows[0][1].shape == (512,)

    def test_clear_face_clusters_removes_prior_assignment(self, conn, case_id, source_file_id):
        node_id = make_node(conn, source_file_id, "image")
        repo = GraphRepository(conn)
        face_ids = repo.insert_face_detections(case_id, [{
            "evidence_node_id": node_id, "frame_path": "/f.jpg",
            "bbox": [0, 0, 1, 1], "confidence": 0.9,
            "embedding": _unit([1.0] + [0.0] * 511),
        }])
        cluster_id = repo.create_face_cluster(case_id, _unit([1.0] + [0.0] * 511), 1)
        repo.assign_faces_to_cluster(face_ids, cluster_id)

        repo.clear_face_clusters(case_id)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT face_cluster_id FROM face_detection WHERE id = %s", (face_ids[0],)
            )
            assert cur.fetchone()[0] is None
            cur.execute("SELECT count(*) FROM face_cluster WHERE case_id = %s", (case_id,))
            assert cur.fetchone()[0] == 0


class TestGraphSummary:
    def test_summary_counts_match_what_was_inserted(self, conn, case_id, source_file_id):
        node_id = make_node(conn, source_file_id)
        repo = GraphRepository(conn)
        entity_id = repo.upsert_entity(case_id, "weapon", "knife", "knife", None)
        repo.add_mention(entity_id, node_id, "knife", "llm_extraction")
        repo.commit()

        summary = repo.graph_summary(case_id)

        assert summary["entities"] == 1
        assert summary["mentions"] == 1


# --------------------------------------------------------------------------
# Phase 4: document-to-video linking, timeline events, audio-prep,
# and node edge traversal
# --------------------------------------------------------------------------

@pytest.fixture
def pdf_source_file_id(conn, case_id):
    source = ScannedFile(
        path=Path(f"/evidence/{uuid.uuid4()}.pdf"),
        file_name="doc.pdf",
        media_type=MediaType.PDF,
        sha256=uuid.uuid4().hex * 2,
        size_bytes=100,
    )
    return SourceFileRepository(conn).register(case_id, source).id


class TestDocumentVideoReferences:
    def test_similar_page_and_segment_get_a_references_edge(
        self, conn, case_id, source_file_id, pdf_source_file_id
    ):
        page_id = make_node(
            conn, pdf_source_file_id, "page", clip_embedding=_unit([1.0, 0.0, 0.0]), page_number=3,
        )
        segment_id = make_node(
            conn, source_file_id, "scene_segment", clip_embedding=_unit([0.999, 0.001, 0.0]),
            start=0.0, end=10.0,
            metadata={
                "representative_frame": "/frames/f2.jpg",
                "frames": [
                    {"timestamp": 1.0, "path": "/frames/f1.jpg"},
                    {"timestamp": 4.25, "path": "/frames/f2.jpg"},
                ],
            },
        )
        unrelated_id = make_node(
            conn, source_file_id, "scene_segment", clip_embedding=_unit([0.0, 1.0, 0.0]),
            start=20.0, end=25.0,
        )

        repo = GraphRepository(conn)
        created = repo.insert_document_video_references(case_id, threshold=0.9, max_nodes=100)

        assert created == 1
        with conn.cursor() as cur:
            cur.execute(
                "SELECT node_a_id, node_b_id, metadata FROM source_alignment "
                "WHERE case_id = %s AND alignment_type = 'REFERENCES'",
                (case_id,),
            )
            rows = cur.fetchall()
        assert len(rows) == 1
        a, b, metadata = rows[0]
        assert {str(a), str(b)} == {page_id, segment_id}
        assert unrelated_id not in (str(a), str(b))
        assert metadata["page_number"] == 3
        assert metadata["matched_frame_timestamp"] == 4.25
        assert metadata["matched_frame_path"] == "/frames/f2.jpg"

    def test_rerun_does_not_duplicate_references(self, conn, case_id, source_file_id, pdf_source_file_id):
        make_node(conn, pdf_source_file_id, "page", clip_embedding=_unit([1.0, 0.0, 0.0]))
        make_node(conn, source_file_id, "scene_segment", clip_embedding=_unit([0.999, 0.001, 0.0]))

        repo = GraphRepository(conn)
        repo.insert_document_video_references(case_id, threshold=0.9, max_nodes=100)
        second_run = repo.insert_document_video_references(case_id, threshold=0.9, max_nodes=100)

        assert second_run == 0


class TestTimelineEventStorage:
    def test_insert_event_and_link_round_trips(self, conn, case_id, source_file_id):
        node_id = make_node(conn, source_file_id, start=0.0, end=5.0)
        repo = GraphRepository(conn)

        event_id = repo.insert_timeline_event(
            case_id, "a knife incident", start_time=0.0, end_time=5.0, node_ids=[node_id],
        )
        linked = repo.link_node_to_event(event_id, node_id)
        duplicate = repo.link_node_to_event(event_id, node_id)

        assert linked is True
        assert duplicate is False
        summary = repo.graph_summary(case_id)
        assert summary["timeline_events"] == 1

    def test_clear_timeline_events_removes_events_and_links(self, conn, case_id, source_file_id):
        node_id = make_node(conn, source_file_id, start=0.0, end=5.0)
        repo = GraphRepository(conn)
        event_id = repo.insert_timeline_event(case_id, "d", 0.0, 5.0, [node_id])
        repo.link_node_to_event(event_id, node_id)

        repo.clear_timeline_events(case_id)

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM timeline_event WHERE case_id = %s", (case_id,))
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT count(*) FROM timeline_event_link WHERE timeline_event_id = %s", (event_id,))
            assert cur.fetchone()[0] == 0

    def test_fetch_event_candidates_and_entities_by_node(self, conn, case_id, source_file_id):
        node_id = make_node(
            conn, source_file_id, start=0.0, end=5.0, text_content="he pulled a knife",
        )
        repo = GraphRepository(conn)
        entity_id = repo.upsert_entity(case_id, "weapon", "knife", "knife", None)
        repo.add_mention(entity_id, node_id, "knife", "llm_extraction")
        repo.commit()

        candidates = repo.fetch_event_candidates(case_id)
        entities_by_node = repo.fetch_entities_by_node(case_id)

        assert any(c.id == node_id for c in candidates)
        assert entities_by_node[node_id] == {entity_id}


class TestAudioEmbeddingCoverage:
    def test_reports_present_and_missing_embeddings(self, conn, case_id, source_file_id):
        make_node(conn, source_file_id, "scene_segment", start=0.0, end=5.0,
                  audio_embedding=_unit([1.0] + [0.0] * 767, dim=768))
        make_node(conn, source_file_id, "scene_segment", start=10.0, end=15.0)

        coverage = GraphRepository(conn).audio_embedding_coverage(case_id)

        assert coverage["scene_segment"]["total"] == 2
        assert coverage["scene_segment"]["with_embedding"] == 1


class TestNodeEdgeTraversal:
    def test_transcript_node_exposes_aligns_with_describes_and_references(
        self, conn, case_id, source_file_id, pdf_source_file_id
    ):
        """The phase 4 acceptance check: a transcript-bearing node (a
        scene_segment) should be reachable via all three new/existing edge
        kinds at once — temporal (ALIGNS_WITH), semantic-visual (REFERENCES),
        and spoken-description-to-frame (DESCRIBES)."""
        transcript_node = make_node(
            conn, source_file_id, "scene_segment", clip_embedding=_unit([1.0, 0.0, 0.0]),
            start=0.0, end=5.0, text_content="he pulled out a knife",
        )
        audio_track = make_node(
            conn, source_file_id, "audio_track", start=0.0, end=30.0,
        )
        page = make_node(
            conn, pdf_source_file_id, "page", clip_embedding=_unit([0.999, 0.001, 0.0]), page_number=1,
        )
        frame = make_node(
            conn, source_file_id, "scene_segment", clip_embedding=_unit([0.0, 1.0, 0.0]),
            start=100.0, end=105.0,
        )

        repo = GraphRepository(conn)
        repo.insert_alignment(case_id, transcript_node, audio_track, "ALIGNS_WITH", 5.0)
        repo.insert_document_video_references(case_id, threshold=0.9, max_nodes=100)
        repo.insert_alignment(
            case_id, transcript_node, frame, "DESCRIBES", 0.42,
            metadata={"cosine_similarity": 0.42, "frame_path": "/frames/x.jpg"},
        )
        repo.commit()

        edges = repo.fetch_edges_for_node(case_id, transcript_node)
        edge_types = {e["alignment_type"] for e in edges}

        assert edge_types == {"ALIGNS_WITH", "REFERENCES", "DESCRIBES"}
        references = next(e for e in edges if e["alignment_type"] == "REFERENCES")
        assert references["other_node_id"] == page


# --------------------------------------------------------------------------
# Phase 5: claims, contradiction edges, and the evidence pack query
# --------------------------------------------------------------------------

class TestClaimStorage:
    def test_store_and_fetch_round_trips_a_claim(self, conn, case_id, source_file_id):
        node_id = make_node(conn, source_file_id, text_content="he pulled a knife",
                            text_embedding=_unit([1.0, 0.0, 0.0], dim=384))
        repo = GraphRepository(conn)

        repo.store_claim(node_id, "The suspect held a knife.")
        repo.commit()

        claims = repo.fetch_claims(case_id)
        assert claims[node_id].claim == "The suspect held a knife."
        assert claims[node_id].text_embedding.shape == (384,)

    def test_a_node_marked_attempted_is_not_re_offered(self, conn, case_id, source_file_id):
        """The resume contract: claim_extracted_at, not claim, is what makes a
        re-run skip a node — including one that yielded no claim."""
        node_id = make_node(conn, source_file_id, text_content="page header")
        repo = GraphRepository(conn)

        pending_before = repo.fetch_claim_candidates(case_id, ["scene_segment"])
        repo.store_claim(node_id, None)
        repo.commit()
        pending_after = repo.fetch_claim_candidates(case_id, ["scene_segment"])

        assert node_id in [n.id for n in pending_before]
        assert node_id not in [n.id for n in pending_after]
        assert repo.fetch_claims(case_id) == {}  # no claim, so nothing to compare

    def test_only_pending_false_re_offers_everything(self, conn, case_id, source_file_id):
        node_id = make_node(conn, source_file_id, text_content="he pulled a knife")
        repo = GraphRepository(conn)
        repo.store_claim(node_id, "The suspect held a knife.")
        repo.commit()

        forced = repo.fetch_claim_candidates(case_id, ["scene_segment"], only_pending=False)
        assert node_id in [n.id for n in forced]


class TestCandidateSourceQueries:
    def test_entity_groups_need_two_or_more_nodes(self, conn, case_id, source_file_id):
        shared_a = make_node(conn, source_file_id)
        shared_b = make_node(conn, source_file_id)
        lonely = make_node(conn, source_file_id)
        repo = GraphRepository(conn)

        knife = repo.upsert_entity(case_id, "weapon", "knife", "knife", None)
        repo.add_mention(knife, shared_a, "knife", "llm_extraction")
        repo.add_mention(knife, shared_b, "knife", "llm_extraction")
        solo = repo.upsert_entity(case_id, "vehicle", "van", "van", None)
        repo.add_mention(solo, lonely, "van", "llm_extraction")
        repo.commit()

        groups = repo.fetch_entity_node_groups(case_id)

        assert len(groups) == 1
        _entity_id, name, node_ids = groups[0]
        assert name == "knife"
        assert set(node_ids) == {shared_a, shared_b}

    def test_alignment_pairs_are_filtered_by_type(self, conn, case_id, source_file_id):
        a = make_node(conn, source_file_id)
        b = make_node(conn, source_file_id, "audio_track")
        c = make_node(conn, source_file_id, "image")
        repo = GraphRepository(conn)
        repo.insert_alignment(case_id, a, b, "ALIGNS_WITH", 1.0)
        repo.insert_alignment(case_id, a, c, "SIMILAR_TO", 0.9)
        repo.commit()

        pairs = repo.fetch_alignment_pairs(case_id, ["ALIGNS_WITH", "DESCRIBES"])

        assert len(pairs) == 1
        assert {pairs[0][0], pairs[0][1]} == {a, b}

    def test_event_groups_need_two_or_more_nodes(self, conn, case_id, source_file_id):
        a = make_node(conn, source_file_id, start=0.0, end=5.0)
        b = make_node(conn, source_file_id, start=1.0, end=6.0)
        lonely = make_node(conn, source_file_id, start=100.0, end=105.0)
        repo = GraphRepository(conn)

        paired = repo.insert_timeline_event(case_id, "an event", 0.0, 6.0, [a, b])
        repo.link_node_to_event(paired, a)
        repo.link_node_to_event(paired, b)
        solo = repo.insert_timeline_event(case_id, "solo", 100.0, 105.0, [lonely])
        repo.link_node_to_event(solo, lonely)
        repo.commit()

        groups = repo.fetch_event_node_groups(case_id)

        assert len(groups) == 1
        assert set(groups[0][1]) == {a, b}


class TestClaimRelationships:
    def test_edge_is_stored_once_in_canonical_order(self, conn, case_id, source_file_id):
        a = make_node(conn, source_file_id)
        b = make_node(conn, source_file_id, "page")
        repo = GraphRepository(conn)

        higher, lower = sorted((a, b), reverse=True)
        created = repo.insert_claim_relationship(
            case_id, higher, lower, "CONTRADICTS", 0.9, "they disagree"
        )
        duplicate = repo.insert_claim_relationship(
            case_id, lower, higher, "CONTRADICTS", 0.9, "they disagree"
        )
        repo.commit()

        assert created is True
        assert duplicate is False

    def test_upsert_survives_the_partial_index_predicate(self, conn, case_id, source_file_id):
        """Regression: uq_relationship_node_pair is a *partial* unique index, and
        Postgres refuses to infer one from a column list alone — the ON CONFLICT
        clause must restate the index predicate or every insert raises
        InvalidColumnReference. Re-inserting is the cheapest way to prove the
        arbiter still resolves."""
        a = make_node(conn, source_file_id)
        b = make_node(conn, source_file_id, "page")
        repo = GraphRepository(conn)

        assert repo.insert_claim_relationship(case_id, a, b, "CORROBORATES", 0.7, "agree") is True
        assert repo.insert_claim_relationship(case_id, a, b, "CORROBORATES", 0.7, "agree") is False
        # A different verdict between the same pair is a different row: the
        # index keys on relationship_type too.
        assert repo.insert_claim_relationship(case_id, a, b, "CONTRADICTS", 0.7, "clash") is True
        repo.commit()

    def test_identity_rows_are_outside_the_partial_index(self, conn, case_id):
        """The predicate excludes NULL node columns, so identity-level rows —
        which phase 6 will write — stay unconstrained by phase 5's uniqueness."""
        with conn.cursor() as cur:
            for _ in range(3):
                cur.execute(
                    """
                    INSERT INTO relationship (case_id, relationship_type, confidence)
                    VALUES (%s, 'co_occurs_with', 0.5)
                    """,
                    (case_id,),
                )
            cur.execute(
                "SELECT count(*) FROM relationship WHERE case_id = %s AND subject_node_id IS NULL",
                (case_id,),
            )
            assert cur.fetchone()[0] == 3
        conn.commit()

    def test_clear_removes_claim_edges_but_spares_identity_edges(
        self, conn, case_id, source_file_id
    ):
        """`relationship` is shared with identity-level edges from another
        phase; clearing phase 5's verdicts must not delete those."""
        a = make_node(conn, source_file_id)
        b = make_node(conn, source_file_id, "page")
        repo = GraphRepository(conn)
        repo.insert_claim_relationship(case_id, a, b, "CONTRADICTS", 0.9, "x")
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO relationship (case_id, relationship_type, confidence)
                VALUES (%s, 'co_occurs_with', 0.5)
                """,
                (case_id,),
            )
        conn.commit()

        repo.clear_claim_relationships(case_id)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT relationship_type FROM relationship WHERE case_id = %s", (case_id,)
            )
            remaining = [r[0] for r in cur.fetchall()]
        assert remaining == ["co_occurs_with"]


class TestWitnessContradictsVideo:
    """The phase 5 acceptance scenario, end to end through the SQL layer: a PDF
    witness statement denying a weapon and a video segment showing a knife,
    linked by the shared 'knife' entity, must produce a retrievable
    CONTRADICTS edge between exactly those two nodes."""

    def _scenario(self, conn, case_id, source_file_id, pdf_source_file_id):
        page = make_node(
            conn, pdf_source_file_id, "page", page_number=2,
            text_content="Document text: The witness stated there was no weapon at the scene.",
            text_embedding=_unit([1.0, 0.2, 0.0], dim=384),
        )
        segment = make_node(
            conn, source_file_id, "scene_segment", start=12.0, end=18.0,
            text_content="Visual description: a man holding a knife.",
            text_embedding=_unit([0.9, 0.3, 0.0], dim=384),
        )
        repo = GraphRepository(conn)
        repo.store_claim(page, "The witness states no weapon was present.")
        repo.store_claim(segment, "The suspect is holding a knife.")

        knife = repo.upsert_entity(case_id, "weapon", "knife", "knife", None)
        repo.add_mention(knife, page, "weapon", "llm_extraction")
        repo.add_mention(knife, segment, "knife", "object_detection")
        repo.commit()
        return repo, page, segment

    def test_contradiction_is_detected_and_retrievable(
        self, conn, case_id, source_file_id, pdf_source_file_id
    ):
        from graph.config import GraphSettings
        from graph.contradictions import detect_contradictions
        from graph.extraction.claims import ClaimVerdict

        repo, page, segment = self._scenario(conn, case_id, source_file_id, pdf_source_file_id)

        class Judge:
            available = True
            unavailable_reason = None

            def compare(self, claim_a, claim_b):
                return ClaimVerdict(
                    "contradicts", 0.93,
                    "One states no weapon was present; the other shows a knife.",
                )

        report = detect_contradictions(repo, Judge(), case_id, GraphSettings())

        assert report.contradicts == 1

        # Retrievable by node...
        edges = repo.fetch_edges_for_node(case_id, page)
        contradiction = next(e for e in edges if e["alignment_type"] == "CONTRADICTS")
        assert contradiction["other_node_id"] == segment
        assert contradiction["score"] == 0.93

        # ...and by the evidence-pack query, filtered to the entity.
        pack = repo.fetch_evidence_pack(case_id, entity_name="knife")
        by_node = {n["node_id"]: n for n in pack}
        assert set(by_node) == {page, segment}
        assert by_node[page]["claim"] == "The witness states no weapon was present."
        page_relations = by_node[page]["relations"]
        assert len(page_relations) == 1
        assert page_relations[0]["relationship_type"] == "CONTRADICTS"
        assert page_relations[0]["other_node_id"] == segment
        assert page_relations[0]["other_node_type"] == "scene_segment"
        assert "no weapon" in page_relations[0]["explanation"]
        # The same disagreement is visible from the other end of the pair.
        assert by_node[segment]["relations"][0]["other_node_id"] == page


class TestEvidencePackFilters:
    def test_time_window_selects_overlapping_nodes(self, conn, case_id, source_file_id):
        inside = make_node(conn, source_file_id, start=10.0, end=20.0)
        overlapping = make_node(conn, source_file_id, start=18.0, end=40.0)
        outside = make_node(conn, source_file_id, start=100.0, end=110.0)

        pack = GraphRepository(conn).fetch_evidence_pack(
            case_id, start_time=5.0, end_time=25.0
        )

        node_ids = {n["node_id"] for n in pack}
        assert inside in node_ids
        assert overlapping in node_ids
        assert outside not in node_ids

    def test_entity_filter_excludes_unmentioned_nodes(self, conn, case_id, source_file_id):
        mentioned = make_node(conn, source_file_id)
        other = make_node(conn, source_file_id)
        repo = GraphRepository(conn)
        knife = repo.upsert_entity(case_id, "weapon", "knife", "knife", None)
        repo.add_mention(knife, mentioned, "knife", "llm_extraction")
        repo.commit()

        pack = repo.fetch_evidence_pack(case_id, entity_name="knife")

        node_ids = {n["node_id"] for n in pack}
        assert node_ids == {mentioned}
        assert other not in node_ids

    def test_no_filters_returns_the_whole_case(self, conn, case_id, source_file_id):
        a = make_node(conn, source_file_id, start=0.0, end=5.0)
        b = make_node(conn, source_file_id, "page", page_number=1)

        pack = GraphRepository(conn).fetch_evidence_pack(case_id)

        assert {n["node_id"] for n in pack} == {a, b}
