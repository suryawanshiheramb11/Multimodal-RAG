"""Tests for web API endpoints including timeline, sync status, and file transcript."""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from psycopg2.extras import Json

from ingestion.config import DatabaseSettings
from ingestion.db import apply_schema, connect
from ingestion.errors import IngestionError
from ingestion.models import MediaType, ScannedFile
from ingestion.repositories import CaseRepository, SourceFileRepository
from web.api.main import app


@pytest.fixture(scope="module")
def client():
    settings = DatabaseSettings()
    try:
        with connect(settings) as connection:
            apply_schema(connection)
    except IngestionError as exc:
        pytest.skip(f"no database available: {exc}")
    yield TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def db_conn():
    settings = DatabaseSettings()
    with connect(settings) as connection:
        yield connection


@pytest.fixture
def case_id(db_conn):
    number = f"API-TEST-{uuid.uuid4()}"
    created = CaseRepository(db_conn).get_or_create(number, "temp", "api integration test")
    yield created
    with db_conn.cursor() as cur:
        cur.execute('DELETE FROM "case" WHERE id = %s', (created,))
    db_conn.commit()


@pytest.fixture
def source_file_id(db_conn, case_id):
    source = ScannedFile(
        path=Path(f"/evidence/{uuid.uuid4()}.mp4"),
        file_name="test_video.mp4",
        media_type=MediaType.VIDEO,
        sha256=uuid.uuid4().hex * 2,
        size_bytes=100,
    )
    return SourceFileRepository(db_conn).register(case_id, source).id


def test_file_transcript_empty(client, source_file_id):
    response = client.get(f"/api/files/{source_file_id}/transcript")
    assert response.status_code == 200
    data = response.json()
    assert data["file_id"] == str(source_file_id)
    assert data["file_name"] == "test_video.mp4"
    assert data["segments"] == []
    assert data["full_text"] is None


def test_file_transcript_with_segments(client, db_conn, source_file_id):
    # Insert evidence nodes with transcripts
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO evidence_node (source_file_id, node_type, start_time, end_time, metadata)
            VALUES (%s, 'scene_segment', 0.0, 5.0, %s),
                   (%s, 'scene_segment', 5.0, 10.0, %s)
            """,
            (
                source_file_id,
                Json({
                    "transcript": {
                        "text": "Hello, world!",
                        "language": "en",
                        "segments": [
                            {"start": 0.0, "end": 2.5, "text": "Hello,"},
                            {"start": 2.5, "end": 5.0, "text": "world!"},
                        ],
                    }
                }),
                source_file_id,
                Json({
                    "transcript": {
                        "text": "How are you doing today?",
                        "language": "en",
                        "segments": [
                            {"start": 5.0, "end": 7.5, "text": "How are you"},
                            {"start": 7.5, "end": 10.0, "text": "doing today?"},
                        ],
                    }
                }),
            ),
        )
    db_conn.commit()

    response = client.get(f"/api/files/{source_file_id}/transcript")
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "en"
    assert len(data["segments"]) == 4
    assert data["full_text"] == "Hello, world! How are you doing today?"
    assert data["segments"][0]["start"] == 0.0
    assert data["segments"][3]["text"] == "doing today?"


def test_collection_timeline(client, db_conn, case_id, source_file_id):
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO evidence_node (source_file_id, node_type, start_time, end_time, case_time, text_content, metadata)
            VALUES (%s, 'scene_segment', 0.0, 5.0, 2.5, 'hello', %s)
            """,
            (
                source_file_id,
                Json({
                    "transcript": {"text": "hello", "segments": []},
                    "caption": "a person outdoors",
                    "ocr": {"text": "STOP"},
                }),
            ),
        )
    db_conn.commit()

    response = client.get(f"/api/collections/{case_id}/timeline")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 1
    item = data["timeline"][0]
    assert item["caption"] == "a person outdoors"
    assert item["ocr_text"] == "STOP"
    assert item["case_time"] == 2.5
    assert item["display_time"] == 2.5


def test_sync_status(client, db_conn, case_id, source_file_id):
    response = client.get(f"/api/collections/{case_id}/sync-status")
    assert response.status_code == 200
    data = response.json()
    assert data["synced"] is False
    assert data["offsets"] == []


def test_collection_graph_empty(client, case_id):
    response = client.get(f"/api/collections/{case_id}/graph")
    assert response.status_code == 200
    assert response.json() == {"nodes": [], "edges": []}


def test_build_graph_unknown_collection_is_404(client):
    response = client.post(f"/api/collections/{uuid.uuid4()}/graph/build")
    assert response.status_code == 404


def test_build_graph_queues_a_job(client, case_id):
    response = client.post(f"/api/collections/{case_id}/graph/build")
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"]

    job = client.get(f"/api/jobs/{body['job_id']}").json()
    assert job["case_id"] == case_id


def test_collection_graph_shapes_entities_identities_and_events(
    client, db_conn, case_id, source_file_id
):
    from graph.repository import GraphRepository

    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO evidence_node (source_file_id, node_type, start_time, end_time)
            VALUES (%s, 'scene_segment', 0.0, 5.0) RETURNING id
            """,
            (source_file_id,),
        )
        node_id = str(cur.fetchone()[0])

    repo = GraphRepository(db_conn)
    knife = repo.upsert_entity(case_id, "weapon", "Knife", "knife", None)
    repo.add_mention(knife, node_id, "knife", "llm_extraction")
    repo.create_identity(case_id, "Jordan")
    event_id = repo.insert_timeline_event(case_id, "something happens", 0.0, 5.0, [node_id])
    repo.link_node_to_event(event_id, node_id)
    repo.commit()

    response = client.get(f"/api/collections/{case_id}/graph")
    assert response.status_code == 200
    data = response.json()

    types = {n["type"] for n in data["nodes"]}
    assert types == {"weapon", "identity", "event", "evidence"}
    labels = {n["label"] for n in data["nodes"]}
    assert {"Knife", "Jordan", "something happens"} <= labels

    edge_types = {e["type"] for e in data["edges"]}
    assert "belongs_to" in edge_types
