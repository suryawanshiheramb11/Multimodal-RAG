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


# --------------------------------------------------------------------------
# Phase 7: voice segments, voice clusters, identities, and IDENTITY_LINK
# --------------------------------------------------------------------------

def _voice(vector: list[float]) -> np.ndarray:
    """A unit vector at voice_segment's real width (256, WeSpeaker ResNet34 —
    see the phase 7 schema migration for how that number was determined)."""
    return _unit(vector, dim=256)


class TestVoiceSegmentStorage:
    def test_insert_and_fetch_round_trips_embeddings(self, conn, case_id, source_file_id):
        repo = GraphRepository(conn)

        ids = repo.insert_voice_segments(case_id, [{
            "source_file_id": source_file_id, "start_time": 0.0, "end_time": 5.0,
            "speaker_label": "SPEAKER_00", "embedding": _voice([1.0, 0.0]),
        }])

        assert len(ids) == 1
        rows = repo.fetch_voice_embeddings(case_id)
        assert len(rows) == 1
        assert rows[0][0] == ids[0]
        assert rows[0][1].shape == (256,)

    def test_fetch_audio_sources_deduplicates_by_source_file(self, conn, case_id, source_file_id):
        """Every scene_segment node from one video shares its extracted audio
        path; diarization must run once per source, not once per segment."""
        make_node(conn, source_file_id, "scene_segment", start=0.0, end=5.0)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE evidence_node SET file_path = %s WHERE source_file_id = %s",
                ("/data/audio/shared.wav", source_file_id),
            )
        make_node(conn, source_file_id, "scene_segment", start=5.0, end=10.0)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE evidence_node SET file_path = %s WHERE source_file_id = %s AND start_time = 5.0",
                ("/data/audio/shared.wav", source_file_id),
            )
        conn.commit()

        sources = GraphRepository(conn).fetch_audio_sources_for_diarization(case_id)

        assert len([s for s in sources if s.source_file_id == source_file_id]) == 1


class TestVoiceClusterStorage:
    def test_assign_and_clear_round_trips(self, conn, case_id, source_file_id):
        repo = GraphRepository(conn)
        ids = repo.insert_voice_segments(case_id, [{
            "source_file_id": source_file_id, "start_time": 0.0, "end_time": 5.0,
            "speaker_label": "SPEAKER_00", "embedding": _voice([1.0, 0.0]),
        }])
        cluster_id = repo.create_voice_cluster(case_id, _voice([1.0, 0.0]), 1)
        repo.assign_voice_segments_to_cluster(ids, cluster_id)

        with conn.cursor() as cur:
            cur.execute("SELECT voice_cluster_id FROM voice_segment WHERE id = %s", (ids[0],))
            assert str(cur.fetchone()[0]) == cluster_id

        repo.clear_voice_clusters(case_id)

        with conn.cursor() as cur:
            cur.execute("SELECT voice_cluster_id FROM voice_segment WHERE id = %s", (ids[0],))
            assert cur.fetchone()[0] is None
            cur.execute("SELECT count(*) FROM voice_cluster WHERE case_id = %s", (case_id,))
            assert cur.fetchone()[0] == 0


class TestIdentityLinkEdge:
    def test_upsert_survives_the_partial_index_predicate(self, conn, case_id, source_file_id):
        """Regression, same class of bug phase 5's CONTRADICTS edge hit:
        uq_relationship_identity_link is a *partial* unique index, and
        Postgres refuses to infer one from a column list alone — ON CONFLICT
        must restate the index predicate or every insert raises
        InvalidColumnReference."""
        node_id = make_node(conn, source_file_id)
        repo = GraphRepository(conn)
        identity_id = repo.create_identity(case_id, "Jordan")

        created = repo.insert_identity_link(case_id, node_id, identity_id, "face", 0.9)
        duplicate = repo.insert_identity_link(case_id, node_id, identity_id, "face", 0.9)

        assert created is True
        assert duplicate is False

    def test_clearing_the_identity_cascades_the_edge_away(self, conn, case_id, source_file_id):
        node_id = make_node(conn, source_file_id)
        repo = GraphRepository(conn)
        identity_id = repo.create_identity(case_id, None)
        repo.insert_identity_link(case_id, node_id, identity_id, "voice", 0.8)

        repo.clear_identities(case_id)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM relationship WHERE case_id = %s AND relationship_type = 'IDENTITY_LINK'",
                (case_id,),
            )
            assert cur.fetchone()[0] == 0

    def test_clearing_identities_unlinks_but_does_not_delete_clusters(self, conn, case_id, source_file_id):
        repo = GraphRepository(conn)
        node_id = make_node(conn, source_file_id, "image")
        face_ids = repo.insert_face_detections(case_id, [{
            "evidence_node_id": node_id, "frame_path": "/f.jpg",
            "bbox": [0, 0, 1, 1], "confidence": 0.9, "embedding": _unit([1.0]),
        }])
        face_cluster_id = repo.create_face_cluster(case_id, _unit([1.0]), 1)
        repo.assign_faces_to_cluster(face_ids, face_cluster_id)
        identity_id = repo.create_identity(case_id, None)
        repo.link_face_cluster_to_identity(face_cluster_id, identity_id)

        repo.clear_identities(case_id)

        with conn.cursor() as cur:
            cur.execute("SELECT identity_id FROM face_cluster WHERE id = %s", (face_cluster_id,))
            assert cur.fetchone()[0] is None  # unlinked...
            cur.execute("SELECT count(*) FROM face_cluster WHERE id = %s", (face_cluster_id,))
            assert cur.fetchone()[0] == 1  # ...but the cluster itself survives


class TestIdentityFusionEndToEnd:
    """The phase 7 acceptance scenario, through the real SQL layer: a person
    visible on camera and speaking at the same time must have their face
    cluster and voice cluster fused into one identity, with IDENTITY_LINK
    edges reaching both a face-matched node and a voice-matched node."""

    def test_speaking_face_merges_into_one_identity_with_both_kinds_of_evidence(
        self, conn, case_id, source_file_id
    ):
        from graph.clustering import cluster_faces, cluster_voices
        from graph.config import GraphSettings
        from graph.identity_fusion import build_identities

        # A video segment where the person's face is visible for 5 seconds...
        node_face = make_node(conn, source_file_id, "scene_segment", start=0.0, end=5.0)
        # ...and a second node, from the same recording, carrying the words
        # they said during that same span — deliberately a *different* node,
        # so "both visual and audio evidence" is demonstrated across two
        # distinct pieces of evidence rather than one node wearing both hats.
        node_voice = make_node(
            conn, source_file_id, "scene_segment", start=0.0, end=5.0,
            text_content="Transcript: interview subject speaking throughout.",
        )

        repo = GraphRepository(conn)
        repo.insert_face_detections(case_id, [{
            "evidence_node_id": node_face, "frame_path": "/frames/f1.jpg",
            "bbox": [1, 2, 3, 4], "confidence": 0.95, "embedding": _unit([1.0, 0.0, 0.0]),
        }])
        # Two turns from the same speaker (near-identical embeddings) so
        # AgglomerativeClustering — which needs at least 2 samples — has
        # something to cluster; a real diarization would split this span
        # into turns at pauses even for one continuous speaker.
        repo.insert_voice_segments(case_id, [
            {"source_file_id": source_file_id, "start_time": 0.0, "end_time": 2.5,
             "speaker_label": "SPEAKER_00", "embedding": _voice([1.0, 0.0])},
            {"source_file_id": source_file_id, "start_time": 2.5, "end_time": 5.0,
             "speaker_label": "SPEAKER_00", "embedding": _voice([0.99, 0.01])},
        ])

        settings = GraphSettings(
            identity_min_windows=3, identity_overlap_threshold=0.6,
            face_cluster_min_samples=1,
        )
        assert cluster_faces(repo, case_id, settings) == 1
        assert cluster_voices(repo, case_id, settings) == 1

        report = build_identities(repo, None, case_id, settings)

        assert report.identities_created == 1
        identities = repo.fetch_identities(case_id)
        assert len(identities) == 1
        identity_id = str(identities[0]["id"])

        evidence = repo.fetch_identity_evidence(case_id, identity_id)
        via_by_node = {row["node_id"]: row["via"] for row in evidence}
        assert via_by_node[node_face] == "face"
        assert via_by_node[node_voice] == "voice"
        assert {row["via"] for row in evidence} == {"face", "voice"}

    def test_weakly_overlapping_face_and_voice_are_not_fused(self, conn, case_id, source_file_id):
        from graph.clustering import cluster_faces, cluster_voices
        from graph.config import GraphSettings
        from graph.identity_fusion import build_identities

        node_face = make_node(conn, source_file_id, "scene_segment", start=0.0, end=10.0)

        repo = GraphRepository(conn)
        repo.insert_face_detections(case_id, [{
            "evidence_node_id": node_face, "frame_path": "/frames/f1.jpg",
            "bbox": [1, 2, 3, 4], "confidence": 0.95, "embedding": _unit([1.0, 0.0, 0.0]),
        }])
        # A different person's voice, active for only one second of the ten
        # the face is visible — a passer-by overheard, not the person on camera.
        repo.insert_voice_segments(case_id, [
            {"source_file_id": source_file_id, "start_time": 8.0, "end_time": 8.5,
             "speaker_label": "SPEAKER_00", "embedding": _voice([0.0, 1.0])},
            {"source_file_id": source_file_id, "start_time": 8.5, "end_time": 9.0,
             "speaker_label": "SPEAKER_00", "embedding": _voice([0.01, 0.99])},
        ])

        settings = GraphSettings(
            identity_min_windows=1, identity_overlap_threshold=0.6,
            face_cluster_min_samples=1,
        )
        cluster_faces(repo, case_id, settings)
        cluster_voices(repo, case_id, settings)

        report = build_identities(repo, None, case_id, settings)

        assert report.identities_created == 0
        assert repo.fetch_identities(case_id) == []


# --------------------------------------------------------------------------
# Question answering repository methods
# --------------------------------------------------------------------------

class TestQARepositoryMethods:
    def test_fetch_nodes_by_ids_returns_requested_nodes_only(self, conn, case_id, source_file_id):
        wanted = make_node(conn, source_file_id, text_content="he pulled a knife")
        other = make_node(conn, source_file_id, text_content="unrelated")

        rows = GraphRepository(conn).fetch_nodes_by_ids([wanted])

        assert {str(r["id"]) for r in rows} == {wanted}
        assert other not in [str(r["id"]) for r in rows]

    def test_fetch_nodes_by_ids_with_empty_list_makes_no_query(self, conn):
        assert GraphRepository(conn).fetch_nodes_by_ids([]) == []

    def test_entity_time_bounds_spans_every_mentioning_node(self, conn, case_id, source_file_id):
        early = make_node(conn, source_file_id, start=5.0, end=10.0)
        late = make_node(conn, source_file_id, start=100.0, end=110.0)
        repo = GraphRepository(conn)
        knife = repo.upsert_entity(case_id, "weapon", "knife", "knife", None)
        repo.add_mention(knife, early, "knife", "llm_extraction")
        repo.add_mention(knife, late, "knife", "llm_extraction")
        repo.commit()

        bounds = repo.fetch_entity_time_bounds(case_id, "knife")

        assert bounds == (5.0, 110.0)

    def test_entity_time_bounds_is_none_for_an_unmentioned_entity(self, conn, case_id):
        assert GraphRepository(conn).fetch_entity_time_bounds(case_id, "nonexistent") is None

    def test_fetch_relations_about_filters_by_subject_text(self, conn, case_id, source_file_id):
        a = make_node(conn, source_file_id, text_content="no weapon was present")
        b = make_node(conn, source_file_id, "page", text_content="a knife is visible")
        c = make_node(conn, source_file_id, text_content="unrelated topic entirely")
        d = make_node(conn, source_file_id, "page", text_content="also unrelated")
        repo = GraphRepository(conn)
        repo.insert_claim_relationship(case_id, a, b, "CONTRADICTS", 0.9, "weapon dispute")
        repo.insert_claim_relationship(case_id, c, d, "CORROBORATES", 0.8, "agreement on something else")
        repo.commit()

        relations = repo.fetch_relations_about(case_id, "weapon", ["CONTRADICTS", "CORROBORATES"])

        assert len(relations) == 1
        assert relations[0]["relationship_type"] == "CONTRADICTS"

    def test_fetch_relations_about_with_no_subject_returns_everything(self, conn, case_id, source_file_id):
        a = make_node(conn, source_file_id)
        b = make_node(conn, source_file_id, "page")
        repo = GraphRepository(conn)
        repo.insert_claim_relationship(case_id, a, b, "CONTRADICTS", 0.9, "x")
        repo.commit()

        relations = repo.fetch_relations_about(case_id, None, ["CONTRADICTS", "CORROBORATES"])

        assert len(relations) == 1

    def test_co_mentioned_entities_share_at_least_one_node(self, conn, case_id, source_file_id):
        shared_node = make_node(conn, source_file_id)
        solo_node = make_node(conn, source_file_id)
        repo = GraphRepository(conn)
        knife = repo.upsert_entity(case_id, "weapon", "knife", "knife", None)
        john = repo.upsert_entity(case_id, "person", "John", "john", None)
        van = repo.upsert_entity(case_id, "vehicle", "van", "van", None)
        repo.add_mention(knife, shared_node, "knife", "llm_extraction")
        repo.add_mention(john, shared_node, "John", "llm_extraction")
        repo.add_mention(van, solo_node, "van", "llm_extraction")
        repo.commit()

        co_mentioned = repo.fetch_co_mentioned_entities(case_id, "knife")

        names = {r["canonical_name"] for r in co_mentioned}
        assert names == {"John"}

    def test_identities_matching_is_case_insensitive_substring(self, conn, case_id):
        repo = GraphRepository(conn)
        repo.create_identity(case_id, "John Smith")

        assert len(repo.fetch_identities_matching(case_id, "john")) == 1
        assert len(repo.fetch_identities_matching(case_id, "SMITH")) == 1
        assert len(repo.fetch_identities_matching(case_id, "nobody")) == 0

    def test_identities_for_nodes_reads_identity_link_edges(self, conn, case_id, source_file_id):
        node_id = make_node(conn, source_file_id)
        repo = GraphRepository(conn)
        identity_id = repo.create_identity(case_id, "Jordan")
        repo.insert_identity_link(case_id, node_id, identity_id, "face", 0.8)

        identities = repo.fetch_identities_for_nodes(case_id, [node_id])

        assert len(identities) == 1
        assert identities[0]["display_name"] == "Jordan"
        assert identities[0]["via"] == "face"

    def test_search_nodes_by_text_ranks_by_cosine_similarity(self, conn, case_id, source_file_id):
        close = make_node(
            conn, source_file_id, text_content="a knife on the table",
            text_embedding=_unit([1.0, 0.0, 0.0], dim=384),
        )
        far = make_node(
            conn, source_file_id, text_content="completely unrelated",
            text_embedding=_unit([0.0, 1.0, 0.0], dim=384),
        )

        results = GraphRepository(conn).search_nodes_by_text(
            case_id, _unit([0.99, 0.01, 0.0], dim=384), limit=5
        )

        by_id = {str(r["id"]): r["score"] for r in results}
        assert by_id[close] > by_id[far]

    def test_search_nodes_by_keyword_matches_a_literal_substring(
        self, conn, case_id, source_file_id
    ):
        has_code = make_node(conn, source_file_id, text_content="PNR# YRKE2C booked by IBOOK")
        no_code = make_node(conn, source_file_id, text_content="completely unrelated text")

        results = GraphRepository(conn).search_nodes_by_keyword(case_id, ["yrke2c"], limit=5)

        ids = {str(r["id"]) for r in results}
        assert has_code in ids
        assert no_code not in ids

    def test_search_nodes_by_keyword_ranks_more_matched_terms_first(
        self, conn, case_id, source_file_id
    ):
        both = make_node(conn, source_file_id, text_content="alpha bravo charlie")
        one = make_node(conn, source_file_id, text_content="alpha only")

        results = GraphRepository(conn).search_nodes_by_keyword(
            case_id, ["alpha", "bravo"], limit=5
        )

        ids = [str(r["id"]) for r in results]
        assert ids.index(both) < ids.index(one)

    def test_search_nodes_by_keyword_with_no_matches_returns_empty(
        self, conn, case_id, source_file_id
    ):
        make_node(conn, source_file_id, text_content="nothing relevant here")

        results = GraphRepository(conn).search_nodes_by_keyword(case_id, ["zzznomatch"], limit=5)

        assert results == []


class TestAnswerQuestionEndToEnd:
    """The QA orchestrator against real Postgres: retrieval must be correct
    SQL, not just correct against fakes."""

    def test_entity_question_finds_a_real_mention(self, conn, case_id, source_file_id):
        from graph.config import GraphSettings
        from graph.qa import answer_question

        node_id = make_node(
            conn, source_file_id, start=4.0, end=8.0, text_content="A knife is on the table."
        )
        repo = GraphRepository(conn)
        knife = repo.upsert_entity(case_id, "weapon", "knife", "knife", None)
        repo.add_mention(knife, node_id, "knife", "llm_extraction")
        repo.commit()

        class StubTextEncoder:
            available = True
            def embed(self, text):
                return None

        answer = answer_question(
            repo, StubTextEncoder(), None, case_id, "Tell me about the knife", GraphSettings()
        )

        assert answer.intent == "entity"
        assert node_id in answer.source_node_ids
        assert "4.0" in answer.text or "knife" in answer.text.lower()


class TestTimelineSyncRepositoryMethods:
    def test_fetch_source_ids_for_case(self, conn, case_id, source_file_id):
        repo = GraphRepository(conn)
        ids = repo.fetch_source_ids_for_case(case_id)
        assert source_file_id in ids

    def test_fetch_source_offsets(self, conn, case_id, source_file_id):
        source_b = ScannedFile(
            path=Path(f"/evidence/{uuid.uuid4()}.mp4"),
            file_name="clip2.mp4",
            media_type=MediaType.VIDEO,
            sha256=uuid.uuid4().hex * 2,
            size_bytes=100,
        )
        source_b_id = SourceFileRepository(conn).register(case_id, source_b).id
        repo = GraphRepository(conn)
        repo.insert_source_offset(
            case_id=case_id,
            source_a_id=source_file_id,
            source_b_id=source_b_id,
            offset_seconds=2.5,
            confidence=0.95,
            method="audio(5)",
            anchor_count=5,
            metadata={"test": True},
        )
        offsets = repo.fetch_source_offsets(case_id)
        assert len(offsets) == 1
        assert str(offsets[0]["source_a_id"]) == str(source_file_id)
        assert str(offsets[0]["source_b_id"]) == str(source_b_id)
        assert offsets[0]["offset_seconds"] == 2.5
        assert offsets[0]["confidence"] == 0.95

    def test_query_unified_timeline_full(self, conn, case_id, source_file_id):
        make_node(
            conn,
            source_file_id,
            start=10.0,
            end=15.0,
            text_content="Transcript: hello world\n\nVisual description: a room",
            metadata={
                "transcript": {
                    "text": "hello world",
                    "segments": [{"start": 10.0, "end": 15.0, "text": "hello world"}],
                },
                "caption": "a room",
                "ocr": {"text": "EXIT"},
            },
        )
        repo = GraphRepository(conn)
        rows = repo.query_unified_timeline_full(case_id)
        assert len(rows) >= 1
        found = [r for r in rows if r["start_time"] == 10.0][0]
        assert found["transcript_text"] == "hello world"
        assert found["caption"] == "a room"
        assert found["ocr_text"] == "EXIT"


class TestGraphOverview:
    """Everything the knowledge-graph view renders, in one call."""

    def test_entities_and_identities_are_both_returned(self, conn, case_id):
        repo = GraphRepository(conn)
        repo.upsert_entity(case_id, "weapon", "Knife", "knife", None)
        repo.create_identity(case_id, "Jordan")
        repo.commit()

        overview = repo.fetch_graph_overview(case_id)

        assert [e["canonical_name"] for e in overview["entities"]] == ["Knife"]
        assert [i["display_name"] for i in overview["identities"]] == ["Jordan"]

    def test_co_occurrence_edge_between_entities_sharing_a_node(
        self, conn, case_id, source_file_id
    ):
        node_id = make_node(conn, source_file_id)
        repo = GraphRepository(conn)
        knife = repo.upsert_entity(case_id, "weapon", "knife", "knife", None)
        jordan = repo.upsert_entity(case_id, "person", "Jordan", "jordan", None)
        repo.add_mention(knife, node_id, "knife", "llm_extraction")
        repo.add_mention(jordan, node_id, "Jordan", "llm_extraction")
        repo.commit()

        overview = repo.fetch_graph_overview(case_id)

        assert len(overview["co_occurrences"]) == 1
        edge = overview["co_occurrences"][0]
        assert {str(edge["source"]), str(edge["target"])} == {knife, jordan}
        assert edge["weight"] == 1

    def test_identity_link_edge_via_shared_node(self, conn, case_id, source_file_id):
        node_id = make_node(conn, source_file_id)
        repo = GraphRepository(conn)
        jordan_entity = repo.upsert_entity(case_id, "person", "Jordan", "jordan", None)
        repo.add_mention(jordan_entity, node_id, "Jordan", "llm_extraction")
        identity_id = repo.create_identity(case_id, "Jordan")
        repo.insert_identity_link(case_id, node_id, identity_id, "face", 0.9)
        repo.commit()

        overview = repo.fetch_graph_overview(case_id)

        assert len(overview["identity_links"]) == 1
        link = overview["identity_links"][0]
        assert str(link["identity_id"]) == identity_id
        assert str(link["entity_id"]) == jordan_entity

    def test_contradiction_edge_pulls_in_both_evidence_nodes(
        self, conn, case_id, source_file_id
    ):
        node_a = make_node(conn, source_file_id, text_content="no weapon was seen")
        node_b = make_node(conn, source_file_id, page_number=1, node_type="page",
                            text_content="a knife is visible")
        repo = GraphRepository(conn)
        repo.insert_claim_relationship(
            case_id, node_a, node_b, "CONTRADICTS", 0.8, "one denies, one shows a weapon"
        )
        repo.commit()

        overview = repo.fetch_graph_overview(case_id)

        assert len(overview["claim_links"]) == 1
        assert overview["claim_links"][0]["relationship_type"] == "CONTRADICTS"
        node_ids = {str(n["id"]) for n in overview["claim_nodes"]}
        assert node_ids == {node_a, node_b}

    def test_entities_with_no_edges_still_return_empty_lists(self, conn, case_id):
        repo = GraphRepository(conn)
        repo.upsert_entity(case_id, "weapon", "Knife", "knife", None)
        repo.commit()

        overview = repo.fetch_graph_overview(case_id)

        assert overview["co_occurrences"] == []
        assert overview["identity_links"] == []
        assert overview["claim_links"] == []
        assert overview["claim_nodes"] == []
        assert overview["events"] == []
        assert overview["event_links"] == []

    def test_timeline_event_and_its_linked_nodes_are_returned(
        self, conn, case_id, source_file_id
    ):
        node_id = make_node(conn, source_file_id, start=10.0, end=12.0)
        repo = GraphRepository(conn)
        event_id = repo.insert_timeline_event(
            case_id, "a car arrives", 10.0, 12.0, [node_id]
        )
        repo.link_node_to_event(event_id, node_id)
        repo.commit()

        overview = repo.fetch_graph_overview(case_id)

        assert [e["description"] for e in overview["events"]] == ["a car arrives"]
        assert overview["event_links"] == [{"event_id": event_id, "evidence_node_id": node_id}]
        assert [str(n["id"]) for n in overview["claim_nodes"]] == [node_id]

