"""Integration tests against a real Postgres.

Skipped automatically when no database is reachable, so the unit suite still
runs on a machine without Postgres.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from ingestion.config import DatabaseSettings
from ingestion.db import apply_schema, connect
from ingestion.errors import IngestionError
from ingestion.models import EvidenceNodeDraft, MediaType, NodeType, ScannedFile
from ingestion.repositories import (
    CaseRepository,
    EvidenceNodeRepository,
    SourceFileRepository,
)


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
    """A throwaway case, removed afterwards by ON DELETE CASCADE."""
    number = f"TEST-{uuid.uuid4()}"
    created = CaseRepository(conn).get_or_create(number, "temp", "integration test")
    yield created
    with conn.cursor() as cur:
        cur.execute('DELETE FROM "case" WHERE id = %s', (created,))
    conn.commit()


def make_source(name="clip.mp4", sha=None, path=None) -> ScannedFile:
    return ScannedFile(
        path=Path(path or f"/evidence/{name}"),
        file_name=name,
        media_type=MediaType.VIDEO,
        sha256=sha or uuid.uuid4().hex * 2,
        size_bytes=1234,
        declared_type="video",
        detected_mime="video/mp4",
    )


class TestSqlInjectionResistance:
    def test_hostile_filename_is_stored_as_data(self, conn, case_id):
        """A filename crafted as SQL must round-trip as a literal string."""
        hostile = "'; DROP TABLE source_file; --"
        source = make_source(name=hostile, path=f"/evidence/{hostile}")

        registered = SourceFileRepository(conn).register(case_id, source)

        with conn.cursor() as cur:
            cur.execute("SELECT file_name FROM source_file WHERE id = %s", (registered.id,))
            stored = cur.fetchone()[0]
            # The table is obviously still there if this query succeeds.
            cur.execute("SELECT count(*) FROM source_file")
            assert cur.fetchone()[0] >= 1

        assert stored == hostile

    def test_hostile_metadata_is_stored_as_json(self, conn, case_id):
        source = make_source()
        source.metadata.update({"note": "'); DELETE FROM \"case\"; --"})

        registered = SourceFileRepository(conn).register(case_id, source)

        with conn.cursor() as cur:
            cur.execute("SELECT metadata->>'note' FROM source_file WHERE id = %s",
                        (registered.id,))
            assert cur.fetchone()[0] == "'); DELETE FROM \"case\"; --"

    def test_hostile_text_content_is_stored_as_data(self, conn, case_id):
        registered = SourceFileRepository(conn).register(case_id, make_source())
        payload = "'; UPDATE evidence_node SET text_content = 'pwned'; --"

        EvidenceNodeRepository(conn).replace_for_source(
            registered.id,
            [EvidenceNodeDraft(node_type=NodeType.PAGE, page_number=1, text_content=payload)],
        )

        with conn.cursor() as cur:
            cur.execute(
                "SELECT text_content FROM evidence_node WHERE source_file_id = %s",
                (registered.id,),
            )
            assert cur.fetchone()[0] == payload


class TestIdempotency:
    def test_reingesting_replaces_rather_than_duplicates(self, conn, case_id):
        registered = SourceFileRepository(conn).register(case_id, make_source())
        repo = EvidenceNodeRepository(conn)
        drafts = [
            EvidenceNodeDraft(node_type=NodeType.SCENE_SEGMENT, start_time=0.0, end_time=5.0),
            EvidenceNodeDraft(node_type=NodeType.SCENE_SEGMENT, start_time=5.0, end_time=10.0),
        ]

        repo.replace_for_source(registered.id, drafts)
        repo.replace_for_source(registered.id, drafts)
        repo.replace_for_source(registered.id, drafts)

        assert repo.count_for_case(case_id) == 2

    def test_same_file_registers_once(self, conn, case_id):
        repo = SourceFileRepository(conn)
        source = make_source(sha="a" * 64)

        first = repo.register(case_id, source)
        second = repo.register(case_id, source)

        assert first.id == second.id
        assert first.is_new is True
        assert second.is_new is False
        assert repo.count_for_case(case_id) == 1


class TestTamperDetection:
    def test_changed_content_at_the_same_path_is_flagged(self, conn, case_id):
        repo = SourceFileRepository(conn)
        path = "/evidence/interview.mp4"

        repo.register(case_id, make_source(sha="a" * 64, path=path))
        second = repo.register(case_id, make_source(sha="b" * 64, path=path))

        assert second.content_changed is True
        assert second.previous_sha256 == "a" * 64

    def test_unchanged_content_is_not_flagged(self, conn, case_id):
        repo = SourceFileRepository(conn)
        path = "/evidence/steady.mp4"

        repo.register(case_id, make_source(sha="c" * 64, path=path))
        second = repo.register(case_id, make_source(sha="c" * 64, path=path))

        assert second.content_changed is False


class TestNodeStats:
    """`stats_for_source` is what lets the pipeline decide whether re-extracting
    a file would throw away enrichment it cannot regenerate."""

    def test_counts_total_and_enriched_separately(self, conn, case_id):
        registered = SourceFileRepository(conn).register(case_id, make_source())
        repo = EvidenceNodeRepository(conn)
        repo.replace_for_source(registered.id, [
            EvidenceNodeDraft(node_type=NodeType.SCENE_SEGMENT, start_time=0.0, end_time=5.0),
            EvidenceNodeDraft(node_type=NodeType.SCENE_SEGMENT, start_time=5.0, end_time=10.0),
        ])

        before = repo.stats_for_source(registered.id)
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evidence_node SET enriched_at = now()
                WHERE id = (SELECT id FROM evidence_node WHERE source_file_id = %s LIMIT 1)
                """,
                (registered.id,),
            )
        conn.commit()
        after = repo.stats_for_source(registered.id)

        assert (before.total, before.enriched) == (2, 0)
        assert (after.total, after.enriched) == (2, 1)

    def test_a_source_with_no_nodes_reports_zero(self, conn, case_id):
        registered = SourceFileRepository(conn).register(case_id, make_source())
        stats = EvidenceNodeRepository(conn).stats_for_source(registered.id)

        assert (stats.total, stats.enriched) == (0, 0)

    def test_replace_for_source_really_does_discard_enrichment(self, conn, case_id):
        """The behaviour the pipeline now guards against, pinned so nobody
        'optimises' the guard away believing the delete was harmless."""
        registered = SourceFileRepository(conn).register(case_id, make_source())
        repo = EvidenceNodeRepository(conn)
        drafts = [EvidenceNodeDraft(node_type=NodeType.SCENE_SEGMENT, start_time=0.0, end_time=5.0)]
        repo.replace_for_source(registered.id, drafts)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE evidence_node SET enriched_at = now() WHERE source_file_id = %s",
                (registered.id,),
            )
        conn.commit()
        assert repo.stats_for_source(registered.id).enriched == 1

        repo.replace_for_source(registered.id, drafts)

        assert repo.stats_for_source(registered.id).enriched == 0

    def test_case_wide_stats_aggregate_across_sources(self, conn, case_id):
        sources = SourceFileRepository(conn)
        repo = EvidenceNodeRepository(conn)
        first = sources.register(case_id, make_source(sha="d" * 64, path="/evidence/a.mp4"))
        second = sources.register(case_id, make_source(sha="e" * 64, path="/evidence/b.mp4"))
        draft = [EvidenceNodeDraft(node_type=NodeType.SCENE_SEGMENT, start_time=0.0, end_time=1.0)]
        repo.replace_for_source(first.id, draft)
        repo.replace_for_source(second.id, draft)

        stats = repo.stats_for_case(case_id)

        assert stats.total == 2
        assert stats.enriched == 0
