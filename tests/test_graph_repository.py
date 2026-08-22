"""Integration tests for graph SQL against a real Postgres.

Skipped automatically when no database is reachable, matching
test_repositories.py's pattern for phase 1.
"""
from __future__ import annotations

import uuid

import numpy as np
import pytest

from graph.repository import GraphRepository
from ingestion.config import DatabaseSettings
from ingestion.db import apply_schema, connect
from ingestion.errors import IngestionError
from ingestion.repositories import CaseRepository, SourceFileRepository
from ingestion.models import MediaType, ScannedFile
from pathlib import Path


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
              start=None, end=None):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO evidence_node (source_file_id, node_type, clip_embedding, start_time, end_time)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (source_file_id, node_type, clip_embedding, start, end),
        )
        node_id = str(cur.fetchone()[0])
    conn.commit()
    return node_id


def _unit(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
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
