"""Tests for web/api/search.py's SemanticSearch: score floors, the relative
gate, the hybrid merge, and the keyword fallback.

Runs against real Postgres (skipped cleanly without one, matching
test_graph_repository.py's pattern) but replaces the two encoders with stubs
that return a fixed vector for a fixed query string, so a test controls
exactly what cosine similarity a query produces without needing a real model
or a network call.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest
from psycopg2.extras import Json

from ingestion.config import DatabaseSettings
from ingestion.db import apply_schema, connect
from ingestion.errors import IngestionError
from ingestion.models import MediaType, ScannedFile
from ingestion.repositories import CaseRepository, SourceFileRepository
from web.api.search import SemanticSearch, _extract_keywords, _fold_in_keyword_hits


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
    number = f"SEARCH-TEST-{uuid.uuid4()}"
    created = CaseRepository(conn).get_or_create(number, "temp", "search test")
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


def make_node(
    conn, source_file_id, node_type="scene_segment", clip_embedding=None,
    start=None, end=None, text_content=None, page_number=None, text_embedding=None,
):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO evidence_node
                (source_file_id, node_type, clip_embedding, start_time, end_time,
                 text_content, page_number, text_embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (source_file_id, node_type, clip_embedding, start, end,
             text_content, page_number, text_embedding, Json({})),
        )
        node_id = str(cur.fetchone()[0])
    conn.commit()
    return node_id


def _unit(vector: list[float], dim: int) -> np.ndarray:
    """A unit vector padded to `dim`, matching the embedding column width —
    pgvector rejects a mismatched dimension outright."""
    padded = list(vector) + [0.0] * (dim - len(vector))
    array = np.asarray(padded, dtype=np.float32)
    return array / np.linalg.norm(array)


class StubEncoder:
    """Replaces `_models.clip` / `_models.text_encoder`: returns a fixed
    vector for a fixed query string rather than actually running a model, so
    a test controls the resulting cosine score exactly."""

    def __init__(self, vector: np.ndarray | None, available: bool = True):
        self._vector = vector
        self.available = available

    def embed_text(self, query):
        return self._vector

    def embed(self, query):
        return self._vector


@pytest.fixture
def searcher():
    """A real SemanticSearch with its (otherwise heavy, network-touching)
    encoders swapped for stubs immediately after construction — the
    constructor itself only builds lazy wrappers, so nothing loads before
    the swap."""
    instance = SemanticSearch.__new__(SemanticSearch)
    instance._settings = None
    instance._models = type("Registry", (), {})()
    return instance


def _set_encoders(searcher, *, clip=None, text=None):
    searcher._models.clip = clip or StubEncoder(None, available=False)
    searcher._models.text_encoder = text or StubEncoder(None, available=False)


class TestScoreFloor:
    def test_a_match_below_the_text_floor_is_dropped(self, conn, searcher, case_id, source_file_id):
        # A visual query vector nearly orthogonal to the stored embedding
        # scores well under 0.25 — below `_MIN_SCORE["text"]`.
        make_node(
            conn, source_file_id, text_content="unrelated",
            text_embedding=_unit([0.0, 1.0, 0.0], dim=384),
        )
        _set_encoders(searcher, text=StubEncoder(_unit([1.0, 0.02, 0.0], dim=384)))

        hits = searcher.search(conn, "irrelevant query", mode="text", case_id=case_id)

        assert hits == []

    def test_a_match_above_the_floor_is_kept(self, conn, searcher, case_id, source_file_id):
        embedding = _unit([1.0, 0.0, 0.0], dim=384)
        node_id = make_node(
            conn, source_file_id, text_content="a knife on the table", text_embedding=embedding,
        )
        _set_encoders(searcher, text=StubEncoder(embedding))

        hits = searcher.search(conn, "knife", mode="text", case_id=case_id)

        assert len(hits) == 1
        assert hits[0].node_id == node_id
        # "knife" is also a literal substring of the node's text, so the
        # keyword fallback legitimately matches it too — this test is only
        # asserting the embedding path didn't drop it.
        assert "text" in hits[0].matched_on


class TestRelativeGate:
    def test_a_weak_hit_relative_to_the_best_is_filtered(self, conn, searcher, case_id, source_file_id):
        query_vector = _unit([1.0, 0.0, 0.0], dim=384)
        strong = make_node(
            conn, source_file_id, text_content="strong match",
            text_embedding=_unit([1.0, 0.0, 0.0], dim=384),
        )
        # cosine(query, this) ~= 0.40: above the 0.25 absolute floor but
        # below the 0.60x-of-best relative gate, which is what removes it.
        make_node(
            conn, source_file_id, text_content="different content entirely",
            text_embedding=_unit([0.4, 0.9, 0.1], dim=384),
        )
        _set_encoders(searcher, text=StubEncoder(query_vector))

        hits = searcher.search(conn, "query", mode="text", case_id=case_id)

        assert [h.node_id for h in hits] == [strong]


class TestHybridMerge:
    def test_a_double_match_outranks_either_single_match(
        self, conn, searcher, case_id, source_file_id
    ):
        text_vec = _unit([1.0, 0.0, 0.0], dim=384)
        clip_vec = _unit([1.0, 0.0, 0.0], dim=512)

        both = make_node(
            conn, source_file_id, text_content="matches both",
            text_embedding=text_vec, clip_embedding=clip_vec,
        )
        text_only = make_node(
            conn, source_file_id, text_content="matches text only", text_embedding=text_vec,
        )

        _set_encoders(
            searcher,
            clip=StubEncoder(clip_vec),
            text=StubEncoder(text_vec),
        )

        hits = searcher.search(conn, "query", mode="hybrid", case_id=case_id)
        by_id = {h.node_id: h for h in hits}

        assert by_id[both].score > by_id[text_only].score
        assert by_id[both].matched_on == ["text", "visual"]

    def test_visual_only_mode_ignores_text_embeddings(
        self, conn, searcher, case_id, source_file_id
    ):
        clip_vec = _unit([1.0, 0.0, 0.0], dim=512)
        make_node(
            conn, source_file_id, text_content="text match only",
            text_embedding=_unit([1.0, 0.0, 0.0], dim=384),
        )
        node_id = make_node(conn, source_file_id, clip_embedding=clip_vec)
        _set_encoders(
            searcher,
            clip=StubEncoder(clip_vec),
            text=StubEncoder(_unit([1.0, 0.0, 0.0], dim=384)),
        )

        hits = searcher.search(conn, "query", mode="visual", case_id=case_id)

        assert [h.node_id for h in hits] == [node_id]


class TestKeywordFallback:
    def test_exact_code_is_found_even_when_the_embedding_score_is_below_floor(
        self, conn, searcher, case_id, source_file_id
    ):
        """The regression this guards: a page containing an exact code (a
        PNR, a case number) can score below the semantic floor while
        unrelated nodes score above it — see graph/qa.py's identical fix."""
        node_id = make_node(
            conn, source_file_id,
            text_content="E-TICKET Reserved On 05 Aug 2025 PNR# YRKE2C Booked By IBOOK",
            text_embedding=_unit([0.0, 1.0, 0.0], dim=384),
        )
        # A query vector orthogonal to the node's embedding — well under the
        # text floor on cosine similarity alone.
        _set_encoders(searcher, text=StubEncoder(_unit([1.0, 0.0, 0.0], dim=384)))

        hits = searcher.search(conn, "what is the PNR number", mode="text", case_id=case_id)

        assert [h.node_id for h in hits] == [node_id]
        assert hits[0].matched_on == ["keyword"]

    def test_visual_mode_does_not_apply_the_keyword_fallback(
        self, conn, searcher, case_id, source_file_id
    ):
        make_node(
            conn, source_file_id, text_content="PNR# YRKE2C",
            text_embedding=_unit([0.0, 1.0, 0.0], dim=384),
        )
        _set_encoders(searcher, clip=StubEncoder(_unit([1.0, 0.0, 0.0], dim=512)))

        hits = searcher.search(conn, "PNR number", mode="visual", case_id=case_id)

        assert hits == []

    def test_keyword_and_embedding_match_on_the_same_node_combine_matched_on(
        self, conn, searcher, case_id, source_file_id
    ):
        embedding = _unit([1.0, 0.0, 0.0], dim=384)
        node_id = make_node(
            conn, source_file_id, text_content="the PNR is YRKE2C", text_embedding=embedding,
        )
        _set_encoders(searcher, text=StubEncoder(embedding))

        hits = searcher.search(conn, "PNR", mode="text", case_id=case_id)

        assert len(hits) == 1
        assert hits[0].node_id == node_id
        assert hits[0].matched_on == ["keyword", "text"]

    def test_no_keyword_terms_falls_straight_through_to_semantic(
        self, conn, searcher, case_id, source_file_id
    ):
        """A query of only stopwords/short words never reaches the keyword
        query at all — matches prior (pre-fallback) behavior exactly."""
        embedding = _unit([1.0, 0.0, 0.0], dim=384)
        node_id = make_node(conn, source_file_id, text_content="a room", text_embedding=embedding)
        _set_encoders(searcher, text=StubEncoder(embedding))

        hits = searcher.search(conn, "is it", mode="text", case_id=case_id)

        assert [h.node_id for h in hits] == [node_id]
        assert hits[0].matched_on == ["text"]

    def test_hybrid_merge_preserves_the_keyword_tag_not_just_text(
        self, conn, searcher, case_id, source_file_id
    ):
        """The regression: a node found only via keyword (its text-embedding
        score was below the floor, exactly the PNR case) that also scores
        above the visual floor must keep "keyword" in matched_on through the
        hybrid merge — not get relabeled "text" just because it came out of
        the (keyword-folded) textual dict."""
        clip_vec = _unit([1.0, 0.0, 0.0], dim=512)
        node_id = make_node(
            conn, source_file_id, text_content="PNR# YRKE2C",
            text_embedding=_unit([0.0, 1.0, 0.0], dim=384),  # orthogonal: below the text floor
            clip_embedding=clip_vec,
        )
        _set_encoders(
            searcher,
            clip=StubEncoder(clip_vec),
            text=StubEncoder(_unit([1.0, 0.0, 0.0], dim=384)),
        )

        hits = searcher.search(conn, "PNR", mode="hybrid", case_id=case_id)

        assert [h.node_id for h in hits] == [node_id]
        assert hits[0].matched_on == ["keyword", "visual"]


class TestCaseScoping:
    def test_case_id_excludes_nodes_from_other_cases(self, conn, searcher, case_id, source_file_id):
        embedding = _unit([1.0, 0.0, 0.0], dim=384)
        make_node(conn, source_file_id, text_content="in scope", text_embedding=embedding)

        other_case = CaseRepository(conn).get_or_create(
            f"OTHER-{uuid.uuid4()}", "temp", "other case"
        )
        other_source = SourceFileRepository(conn).register(
            other_case,
            ScannedFile(
                path=Path(f"/evidence/{uuid.uuid4()}.mp4"), file_name="other.mp4",
                media_type=MediaType.VIDEO, sha256=uuid.uuid4().hex * 2, size_bytes=100,
            ),
        ).id
        make_node(conn, other_source, text_content="out of scope", text_embedding=embedding)

        _set_encoders(searcher, text=StubEncoder(embedding))
        hits = searcher.search(conn, "query", mode="text", case_id=case_id)

        assert len(hits) == 1
        assert hits[0].case_id == case_id


class TestEmptyQuery:
    def test_empty_query_returns_no_hits(self, conn, searcher, case_id):
        _set_encoders(searcher, text=StubEncoder(_unit([1.0, 0.0, 0.0], dim=384)))
        assert searcher.search(conn, "   ", mode="text", case_id=case_id) == []


class TestExtractKeywords:
    def test_stopwords_and_short_words_are_dropped(self):
        assert _extract_keywords("what is the PNR number") == ["pnr", "number"]

    def test_only_stopwords_yields_nothing(self):
        assert _extract_keywords("is it the") == []

    def test_capped_at_max_terms(self):
        words = " ".join(f"term{i}" for i in range(10))
        assert len(_extract_keywords(words, max_terms=3)) == 3


class TestFoldInKeywordHits:
    def test_keyword_only_hit_is_added(self):
        from web.api.search import SearchHit

        keyword_hit = SearchHit(
            node_id="n1", source_file_id="f1", file_name="a.pdf", file_type="pdf",
            case_id="c1", node_type="page", start_time=None, end_time=None,
            page_number=1, text_content="PNR YRKE2C", score=1.0, matched_on=["keyword"],
        )
        combined = _fold_in_keyword_hits({}, {"n1": keyword_hit})
        assert combined == {"n1": keyword_hit}

    def test_overlapping_hit_keeps_text_score_but_adds_the_tag(self):
        from web.api.search import SearchHit

        text_hit = SearchHit(
            node_id="n1", source_file_id="f1", file_name="a.pdf", file_type="pdf",
            case_id="c1", node_type="page", start_time=None, end_time=None,
            page_number=1, text_content="PNR YRKE2C", score=0.42, matched_on=["text"],
        )
        keyword_hit = SearchHit(
            **{**text_hit.__dict__, "score": 1.0, "matched_on": ["keyword"]}
        )
        combined = _fold_in_keyword_hits({"n1": text_hit}, {"n1": keyword_hit})
        assert combined["n1"].score == 0.42
        assert combined["n1"].matched_on == ["keyword", "text"]
