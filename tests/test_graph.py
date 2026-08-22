"""Graph construction tests — entity dedup, extraction parsing, alignment
windows, and clustering — all offline against stubs or a real Postgres
(skipped cleanly when unavailable, matching test_repositories.py's pattern).
"""
from __future__ import annotations

import uuid

import numpy as np
import pytest

from graph.alignment import _overlap_seconds, build_temporal_alignments
from graph.clustering import cluster_faces
from graph.config import GraphSettings
from graph.contradictions import (
    build_candidate_pairs,
    detect_contradictions,
    extract_claims,
    filter_pairs_by_similarity,
)
from graph.crossmodal import build_transcript_visual_links
from graph.entities import build_entities_and_mentions
from graph.extraction.claims import ClaimExtractor, ClaimVerdict, ContradictionJudge
from graph.extraction.detections import entities_from_detections
from graph.extraction.entities import EntityExtractor, ExtractedEntity, normalize_name
from graph.extraction.json_response import parse_json_object
from graph.repository import (
    ClaimRecord,
    ClipFrameNode,
    EventCandidate,
    TimeWindow,
    TranscriptNode,
)
from graph.timeline import _describe_group, build_timeline_events, group_into_events

# --------------------------------------------------------------------------
# Pure-function tests: no I/O at all
# --------------------------------------------------------------------------

class TestNormalizeName:
    def test_collapses_whitespace_and_case(self):
        assert normalize_name("  John   SMITH ") == "john smith"

    def test_same_name_different_formatting_matches(self):
        assert normalize_name("Knife") == normalize_name("  knife  ")


class TestDetectionEntities:
    def test_maps_notable_coco_labels(self):
        metadata = {"detections": {"labels": {"knife": {}, "person": {}, "dog": {}}}}
        entities = entities_from_detections(metadata)

        by_name = {e.name: e for e in entities}
        assert by_name["knife"].entity_type == "weapon"
        assert by_name["person"].entity_type == "person"
        assert "dog" not in by_name  # not a tracked entity type
        assert all(e.source == "object_detection" for e in entities)

    def test_no_detections_gives_no_entities(self):
        assert entities_from_detections({}) == []
        assert entities_from_detections({"detections": {}}) == []


class TestEntityExtractionParsing:
    def _extractor(self):
        return EntityExtractor(captioner=None, settings=GraphSettings())

    def test_parses_clean_json(self):
        extractor = self._extractor()
        response = '{"entities": [{"type": "person", "name": "John Doe"}]}'
        assert extractor._parse(response) == [
            ExtractedEntity("person", "John Doe", source="llm_extraction")
        ]

    def test_extracts_json_from_prose_wrapper(self):
        """Models sometimes ignore 'respond with only JSON' and add commentary."""
        extractor = self._extractor()
        response = 'Sure, here is the JSON:\n{"entities": [{"type": "weapon", "name": "knife"}]}\nDone.'
        result = extractor._parse(response)
        assert result == [ExtractedEntity("weapon", "knife", source="llm_extraction")]

    def test_unrecognized_type_falls_back_to_other(self):
        extractor = self._extractor()
        response = '{"entities": [{"type": "animal", "name": "dog"}]}'
        assert extractor._parse(response)[0].entity_type == "other"

    def test_malformed_json_returns_empty(self):
        extractor = self._extractor()
        assert extractor._parse("not json at all") == []

    def test_entities_missing_a_name_are_dropped(self):
        extractor = self._extractor()
        response = '{"entities": [{"type": "person", "name": ""}, {"type": "person", "name": "Alice"}]}'
        result = extractor._parse(response)
        assert len(result) == 1
        assert result[0].name == "Alice"

    def test_non_list_entities_field_returns_empty(self):
        extractor = self._extractor()
        assert extractor._parse('{"entities": "not a list"}') == []


class TestTemporalOverlap:
    def _window(self, node_type, start, end, source="src-1"):
        return TimeWindow(id=str(uuid.uuid4()), node_type=node_type,
                          source_file_id=source, start_time=start, end_time=end)

    def test_overlapping_windows_have_positive_overlap(self):
        a = self._window("scene_segment", 0.0, 5.0)
        b = self._window("audio_track", 2.0, 10.0)
        assert _overlap_seconds(a, b) == 3.0

    def test_non_overlapping_windows_are_negative(self):
        a = self._window("scene_segment", 0.0, 5.0)
        b = self._window("scene_segment", 10.0, 15.0)
        assert _overlap_seconds(a, b) < 0


class FakeAlignmentRepository:
    def __init__(self, windows):
        self._windows = windows
        self.inserted = []

    def fetch_time_windows(self, case_id):
        return self._windows

    def insert_alignment(self, case_id, a, b, alignment_type, score, metadata=None):
        pair = tuple(sorted((a, b)))
        if pair in self.inserted:
            return False
        self.inserted.append(pair)
        return True

    def commit(self):
        pass


class TestBuildTemporalAlignments:
    def test_cross_type_overlap_creates_an_edge(self):
        segment = TimeWindow("seg-1", "scene_segment", "file-1", 0.0, 5.0)
        track = TimeWindow("track-1", "audio_track", "file-1", 0.0, 30.0)
        repo = FakeAlignmentRepository([segment, track])

        created = build_temporal_alignments(repo, "case-1", GraphSettings())

        assert created == 1
        assert repo.inserted == [("seg-1", "track-1")]

    def test_same_type_overlap_is_not_aligned(self):
        """Two scene_segments from the same split are not an alignment."""
        a = TimeWindow("seg-1", "scene_segment", "file-1", 0.0, 5.0)
        b = TimeWindow("seg-2", "scene_segment", "file-1", 3.0, 8.0)
        repo = FakeAlignmentRepository([a, b])

        assert build_temporal_alignments(repo, "case-1", GraphSettings()) == 0

    def test_different_source_files_never_align(self):
        a = TimeWindow("seg-1", "scene_segment", "file-1", 0.0, 5.0)
        b = TimeWindow("track-1", "audio_track", "file-2", 0.0, 5.0)
        repo = FakeAlignmentRepository([a, b])

        assert build_temporal_alignments(repo, "case-1", GraphSettings()) == 0

    def test_min_overlap_threshold_is_respected(self):
        a = TimeWindow("seg-1", "scene_segment", "file-1", 0.0, 5.0)
        b = TimeWindow("track-1", "audio_track", "file-1", 4.9, 30.0)
        repo = FakeAlignmentRepository([a, b])
        settings = GraphSettings(min_overlap_sec=1.0)

        assert build_temporal_alignments(repo, "case-1", settings) == 0


# --------------------------------------------------------------------------
# Entity + mention orchestration, with stub extractor/encoder/repository
# --------------------------------------------------------------------------

class StubTextNode:
    def __init__(self, id, text_content, metadata=None):
        self.id = id
        self.text_content = text_content
        self.metadata = metadata or {}


class StubExtractor:
    available = True

    def extract(self, text):
        if "knife" in text:
            return [ExtractedEntity("weapon", "knife")]
        return [ExtractedEntity("person", "John Doe")]


class StubTextEncoder:
    def embed(self, text):
        return np.ones(384, dtype=np.float32)


class FakeEntityRepository:
    def __init__(self, nodes):
        self._nodes = nodes
        self.entities: dict[str, dict] = {}
        self.mentions: set[tuple[str, str]] = set()

    def fetch_text_nodes(self, case_id):
        return self._nodes

    def upsert_entity(self, case_id, entity_type, canonical_name, normalized_name, embedding):
        key = (entity_type, normalized_name)
        if key not in self.entities:
            self.entities[key] = {"id": str(uuid.uuid4()), "canonical_name": canonical_name}
        return self.entities[key]["id"]

    def add_mention(self, entity_id, node_id, mention_text, source, confidence=None):
        pair = (entity_id, node_id)
        if pair in self.mentions:
            return False
        self.mentions.add(pair)
        return True

    def commit(self):
        pass


class TestBuildEntitiesAndMentions:
    def test_same_entity_across_nodes_deduplicates(self):
        """The literal case from the spec: 'knife' mentioned twice must
        collapse into one entity row, with two MENTIONS edges."""
        nodes = [
            StubTextNode("node-1", "he pulled out a knife"),
            StubTextNode("node-2", "the knife was on the table"),
        ]
        repo = FakeEntityRepository(nodes)

        entity_count, mention_count = build_entities_and_mentions(
            repo, StubExtractor(), StubTextEncoder(), "case-1", GraphSettings()
        )

        assert entity_count == 1
        assert mention_count == 2
        assert len(repo.entities) == 1

    def test_object_detections_also_produce_entities(self):
        """A node with a detected 'knife' but text that never mentions it
        must still get a weapon entity — detections are a second, independent
        source of candidates, not a fallback only used when text matches."""
        nodes = [
            StubTextNode("node-1", "a plain scene, nothing said",
                         metadata={"detections": {"labels": {"knife": {}}}}),
        ]
        repo = FakeEntityRepository(nodes)

        entity_count, mention_count = build_entities_and_mentions(
            repo, StubExtractor(), StubTextEncoder(), "case-1", GraphSettings()
        )

        # StubExtractor's text branch ("John Doe") plus the detection-derived
        # "knife" are two distinct entities, each with one mention.
        assert entity_count == 2
        assert mention_count == 2

    def test_llm_and_detection_agreeing_on_the_same_entity_deduplicate(self):
        """When both sources name the same normalized entity on one node, it
        is still one entity — but with only one mention, since a node can only
        mention a given entity once (the mention table's unique constraint)."""
        nodes = [
            StubTextNode("node-1", "he pulled out a knife",
                         metadata={"detections": {"labels": {"knife": {}}}}),
        ]
        repo = FakeEntityRepository(nodes)

        entity_count, mention_count = build_entities_and_mentions(
            repo, StubExtractor(), StubTextEncoder(), "case-1", GraphSettings()
        )

        assert entity_count == 1
        assert mention_count == 1

    def test_disabled_extraction_still_uses_detections(self):
        nodes = [
            StubTextNode("node-1", "irrelevant text",
                         metadata={"detections": {"labels": {"person": {}}}}),
        ]
        repo = FakeEntityRepository(nodes)
        settings = GraphSettings(enable_entity_extraction=False)

        entity_count, mention_count = build_entities_and_mentions(
            repo, StubExtractor(), StubTextEncoder(), "case-1", settings
        )

        assert entity_count == 1
        assert mention_count == 1


# --------------------------------------------------------------------------
# Face clustering (DBSCAN over stub embeddings — no insightface required)
# --------------------------------------------------------------------------

class FakeFaceRepository:
    def __init__(self, embeddings: list[tuple[str, np.ndarray]]):
        self._embeddings = embeddings
        self.clusters: list[dict] = []
        self.assignments: dict[str, str | None] = {}
        self.cleared = False

    def clear_face_clusters(self, case_id):
        self.cleared = True
        self.clusters = []
        self.assignments = {}

    def fetch_face_embeddings(self, case_id):
        return self._embeddings

    def create_face_cluster(self, case_id, representative_embedding, face_count):
        cluster_id = str(uuid.uuid4())
        self.clusters.append({"id": cluster_id, "count": face_count})
        return cluster_id

    def assign_faces_to_cluster(self, face_ids, cluster_id):
        for fid in face_ids:
            self.assignments[fid] = cluster_id


def _unit(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


class TestFaceClustering:
    def test_two_tight_groups_become_two_clusters(self):
        embeddings = [
            ("f1", _unit([1.0, 0.0, 0.0])),
            ("f2", _unit([0.99, 0.01, 0.0])),
            ("f3", _unit([0.0, 1.0, 0.0])),
            ("f4", _unit([0.01, 0.99, 0.0])),
        ]
        repo = FakeFaceRepository(embeddings)
        settings = GraphSettings(face_cluster_eps=0.1, face_cluster_min_samples=2)

        count = cluster_faces(repo, "case-1", settings)

        assert count == 2
        assert repo.assignments["f1"] == repo.assignments["f2"]
        assert repo.assignments["f3"] == repo.assignments["f4"]
        assert repo.assignments["f1"] != repo.assignments["f3"]

    def test_isolated_face_is_left_unclustered(self):
        embeddings = [
            ("f1", _unit([1.0, 0.0, 0.0])),
            ("f2", _unit([0.99, 0.01, 0.0])),
            ("f3", _unit([-1.0, 0.0, 0.0])),  # far from the pair
        ]
        repo = FakeFaceRepository(embeddings)
        settings = GraphSettings(face_cluster_eps=0.1, face_cluster_min_samples=2)

        cluster_faces(repo, "case-1", settings)

        assert repo.assignments.get("f3") is None
        assert repo.assignments["f1"] == repo.assignments["f2"]

    def test_below_min_samples_skips_clustering_entirely(self):
        repo = FakeFaceRepository([("f1", _unit([1.0, 0.0, 0.0]))])
        settings = GraphSettings(face_cluster_min_samples=2)

        assert cluster_faces(repo, "case-1", settings) == 0
        assert repo.clusters == []

    def test_reclustering_clears_prior_clusters_first(self):
        repo = FakeFaceRepository([])
        cluster_faces(repo, "case-1", GraphSettings())
        assert repo.cleared is True


# --------------------------------------------------------------------------
# Phase 4: timeline event grouping (pure) + orchestration (stub repository)
# --------------------------------------------------------------------------

def _event_candidate(id, start, end, node_type="scene_segment", text=None, embedding=None):
    return EventCandidate(
        id=id, node_type=node_type, source_file_id="file-1", start_time=start, end_time=end,
        text_content=text, text_embedding=embedding,
    )


class TestGroupIntoEvents:
    def test_shared_entity_within_window_groups(self):
        a = _event_candidate("a", 0.0, 5.0)
        b = _event_candidate("b", 20.0, 25.0)
        entities = {"a": {"ent-1"}, "b": {"ent-1"}}

        groups = group_into_events([a, b], entities, GraphSettings(timeline_window_sec=30.0))

        assert len(groups) == 1
        assert {c.id for c in groups[0]} == {"a", "b"}

    def test_outside_window_never_groups_even_with_shared_entity(self):
        a = _event_candidate("a", 0.0, 5.0)
        b = _event_candidate("b", 60.0, 65.0)
        entities = {"a": {"ent-1"}, "b": {"ent-1"}}

        groups = group_into_events([a, b], entities, GraphSettings(timeline_window_sec=30.0))

        assert groups == []

    def test_no_shared_entity_and_low_text_similarity_does_not_group(self):
        a = _event_candidate("a", 0.0, 5.0, embedding=_unit([1.0, 0.0, 0.0]))
        b = _event_candidate("b", 5.0, 10.0, embedding=_unit([0.0, 1.0, 0.0]))

        groups = group_into_events([a, b], {}, GraphSettings(timeline_window_sec=30.0))

        assert groups == []

    def test_high_text_similarity_groups_without_a_shared_entity(self):
        a = _event_candidate("a", 0.0, 5.0, embedding=_unit([1.0, 0.0, 0.0]))
        b = _event_candidate("b", 5.0, 10.0, embedding=_unit([0.999, 0.001, 0.0]))

        groups = group_into_events(
            [a, b], {}, GraphSettings(timeline_window_sec=30.0, timeline_text_similarity_threshold=0.9)
        )

        assert len(groups) == 1
        assert {c.id for c in groups[0]} == {"a", "b"}

    def test_singleton_is_not_returned_as_a_group(self):
        a = _event_candidate("a", 0.0, 5.0)
        groups = group_into_events([a], {}, GraphSettings())
        assert groups == []

    def test_chain_of_overlapping_windows_transitively_groups(self):
        """a<->b share an entity, b<->c share a different one: a 'same event'
        chain should still merge into one group via b, even though a and c
        share nothing directly."""
        a = _event_candidate("a", 0.0, 1.0)
        b = _event_candidate("b", 20.0, 21.0)
        c = _event_candidate("c", 40.0, 41.0)
        entities = {"a": {"ent-1"}, "b": {"ent-1", "ent-2"}, "c": {"ent-2"}}

        groups = group_into_events([a, b, c], entities, GraphSettings(timeline_window_sec=30.0))

        assert len(groups) == 1
        assert {c.id for c in groups[0]} == {"a", "b", "c"}


class StubCaptioner:
    available = True

    def __init__(self, response="a summarized event"):
        self._response = response

    def complete(self, prompt, *, json_mode=False):
        return self._response


class TestDescribeGroup:
    def test_uses_llm_summary_when_available(self):
        group = [_event_candidate("a", 0.0, 5.0, text="he pulled a knife")]
        description = _describe_group(StubCaptioner("a threat was made"), group, GraphSettings())
        assert description == "a threat was made"

    def test_falls_back_when_captioner_unavailable(self):
        group = [_event_candidate("a", 0.0, 5.0, text="he pulled a knife")]
        description = _describe_group(None, group, GraphSettings())
        assert "1 evidence node(s)" in description

    def test_falls_back_when_llm_summary_disabled(self):
        group = [_event_candidate("a", 0.0, 5.0, text="he pulled a knife")]
        settings = GraphSettings(enable_timeline_llm_summary=False)
        description = _describe_group(StubCaptioner("unused"), group, settings)
        assert "evidence node(s)" in description


class FakeTimelineRepository:
    def __init__(self, candidates, entities_by_node=None):
        self._candidates = candidates
        self._entities_by_node = entities_by_node or {}
        self.cleared = False
        self.events: list[dict] = []
        self.links: set[tuple[str, str]] = set()

    def clear_timeline_events(self, case_id):
        self.cleared = True
        self.events = []
        self.links = set()

    def fetch_event_candidates(self, case_id):
        return self._candidates

    def fetch_entities_by_node(self, case_id):
        return self._entities_by_node

    def insert_timeline_event(self, case_id, description, start_time, end_time, node_ids, metadata=None):
        event_id = f"event-{len(self.events)}"
        self.events.append({"id": event_id, "description": description, "node_ids": node_ids})
        return event_id

    def link_node_to_event(self, timeline_event_id, evidence_node_id):
        pair = (timeline_event_id, evidence_node_id)
        if pair in self.links:
            return False
        self.links.add(pair)
        return True

    def commit(self):
        pass


class TestBuildTimelineEvents:
    def test_grouped_nodes_produce_one_event_with_links_to_each_member(self):
        a = _event_candidate("a", 0.0, 5.0, text="he pulled a knife")
        b = _event_candidate("b", 10.0, 15.0, text="a knife was seen")
        repo = FakeTimelineRepository([a, b], {"a": {"ent-1"}, "b": {"ent-1"}})

        events_created, links_created = build_timeline_events(
            repo, StubCaptioner("a knife incident"), "case-1", GraphSettings()
        )

        assert events_created == 1
        assert links_created == 2
        assert repo.events[0]["description"] == "a knife incident"
        assert repo.events[0]["node_ids"] == ["a", "b"]

    def test_ungrouped_nodes_produce_no_events(self):
        a = _event_candidate("a", 0.0, 5.0)
        b = _event_candidate("b", 500.0, 505.0)
        repo = FakeTimelineRepository([a, b])

        events_created, links_created = build_timeline_events(
            repo, StubCaptioner(), "case-1", GraphSettings()
        )

        assert events_created == 0
        assert links_created == 0

    def test_rerun_clears_prior_events_first(self):
        repo = FakeTimelineRepository([])
        build_timeline_events(repo, StubCaptioner(), "case-1", GraphSettings())
        assert repo.cleared is True


# --------------------------------------------------------------------------
# Phase 4: transcript <-> frame DESCRIBES linking (stub repository + CLIP)
# --------------------------------------------------------------------------

class StubClipEncoder:
    available = True
    unavailable_reason = None

    def __init__(self, text_vectors: dict[str, np.ndarray]):
        self._text_vectors = text_vectors

    def embed_text(self, text):
        return self._text_vectors.get(text)


class FakeCrossModalRepository:
    def __init__(self, transcripts, frames):
        self._transcripts = transcripts
        self._frames = frames
        self.inserted: list[tuple] = []
        self._seen: set[tuple] = set()

    def fetch_transcript_nodes(self, case_id):
        return self._transcripts

    def fetch_clip_frame_nodes(self, case_id, node_types):
        return self._frames

    def insert_alignment(self, case_id, a, b, alignment_type, score, metadata=None):
        key = tuple(sorted((a, b))) + (alignment_type,)
        if key in self._seen:
            return False
        self._seen.add(key)
        self.inserted.append((a, b, alignment_type, score, metadata))
        return True

    def commit(self):
        pass


class TestBuildTranscriptVisualLinks:
    def test_similar_frame_gets_a_describes_edge(self):
        transcript = TranscriptNode("t1", "scene_segment", "file-1", "he pulled out a knife")
        frame = ClipFrameNode(
            id="f1", node_type="scene_segment", source_file_id="file-2",
            embedding=_unit([1.0, 0.0, 0.0]), page_number=None,
            frame_timestamp=12.5, frame_path="/frames/f1.jpg",
        )
        clip = StubClipEncoder({"he pulled out a knife": _unit([0.99, 0.01, 0.0])})
        repo = FakeCrossModalRepository([transcript], [frame])

        created = build_transcript_visual_links(
            repo, clip, "case-1", GraphSettings(describes_similarity_threshold=0.3)
        )

        assert created == 1
        a, b, alignment_type, score, metadata = repo.inserted[0]
        assert {a, b} == {"t1", "f1"}
        assert alignment_type == "DESCRIBES"
        assert metadata["frame_timestamp"] == 12.5

    def test_dissimilar_frame_gets_no_edge(self):
        transcript = TranscriptNode("t1", "scene_segment", "file-1", "he pulled out a knife")
        frame = ClipFrameNode(
            id="f1", node_type="scene_segment", source_file_id="file-2",
            embedding=_unit([0.0, 1.0, 0.0]), page_number=None,
            frame_timestamp=1.0, frame_path="/frames/f1.jpg",
        )
        clip = StubClipEncoder({"he pulled out a knife": _unit([1.0, 0.0, 0.0])})
        repo = FakeCrossModalRepository([transcript], [frame])

        created = build_transcript_visual_links(
            repo, clip, "case-1", GraphSettings(describes_similarity_threshold=0.3)
        )

        assert created == 0

    def test_unavailable_clip_encoder_skips_cleanly(self):
        clip = StubClipEncoder({})
        clip.available = False
        clip.unavailable_reason = "no model"
        repo = FakeCrossModalRepository(
            [TranscriptNode("t1", "scene_segment", "file-1", "text")], []
        )

        assert build_transcript_visual_links(repo, clip, "case-1", GraphSettings()) == 0


# --------------------------------------------------------------------------
# Phase 5: JSON response parsing
# --------------------------------------------------------------------------

class TestParseJsonObject:
    def test_parses_a_clean_object(self):
        assert parse_json_object('{"relation": "contradicts"}') == {"relation": "contradicts"}

    def test_extracts_an_object_from_a_prose_wrapper(self):
        response = 'Here you go:\n```json\n{"relation": "unrelated"}\n```\nHope that helps.'
        assert parse_json_object(response) == {"relation": "unrelated"}

    def test_malformed_json_returns_none(self):
        assert parse_json_object("not json at all") is None

    def test_non_object_payload_returns_none(self):
        """Every prompt here asks for an object; a bare list means the model
        ignored the schema, not that there is a usable answer."""
        assert parse_json_object("[1, 2, 3]") is None

    def test_empty_response_returns_none(self):
        assert parse_json_object("") is None
        assert parse_json_object(None) is None


# --------------------------------------------------------------------------
# Phase 5: claim extraction and contradiction judging (prompt-response parsing)
# --------------------------------------------------------------------------

class ScriptedCaptioner:
    """A Captioner stand-in returning canned completions, in order."""

    available = True
    unavailable_reason = None

    def __init__(self, *responses):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt, *, json_mode=False):
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else None


class TestClaimExtractorParsing:
    def test_collapses_whitespace_into_one_claim(self):
        assert ClaimExtractor._parse("  The suspect   held\na knife.  ") == "The suspect held a knife."

    def test_none_sentinel_means_no_claim(self):
        assert ClaimExtractor._parse("NONE") is None
        assert ClaimExtractor._parse("none.") is None

    def test_empty_response_means_no_claim(self):
        assert ClaimExtractor._parse("") is None
        assert ClaimExtractor._parse(None) is None

    def test_a_claim_merely_containing_none_is_kept(self):
        """Only a whole-response NONE is the sentinel; a real claim that uses
        the word must not be thrown away."""
        claim = ClaimExtractor._parse("The witness saw none of the assault.")
        assert claim == "The witness saw none of the assault."

    def test_extract_truncates_long_text_before_prompting(self):
        captioner = ScriptedCaptioner("The suspect held a knife.")
        extractor = ClaimExtractor(captioner, GraphSettings(max_claim_chars=10))

        extractor.extract("x" * 500)

        assert "x" * 10 in captioner.prompts[0]
        assert "x" * 11 not in captioner.prompts[0]

    def test_blank_text_never_calls_the_model(self):
        captioner = ScriptedCaptioner("unused")
        extractor = ClaimExtractor(captioner, GraphSettings())

        assert extractor.extract("   ") is None
        assert captioner.prompts == []


class TestContradictionJudgeParsing:
    def _judge(self, *responses):
        return ContradictionJudge(ScriptedCaptioner(*responses), GraphSettings())

    def test_parses_a_contradiction_verdict(self):
        judge = self._judge(
            '{"relation": "contradicts", "confidence": 0.9, "explanation": "one denies the other"}'
        )
        verdict = judge.compare("No weapon was present.", "The suspect held a knife.")

        assert verdict == ClaimVerdict("contradicts", 0.9, "one denies the other")
        assert verdict.edge_type == "CONTRADICTS"

    def test_corroborates_maps_to_its_edge_type(self):
        judge = self._judge('{"relation": "corroborates", "confidence": 0.8, "explanation": "agree"}')
        assert judge.compare("a", "b").edge_type == "CORROBORATES"

    def test_unrelated_has_no_edge_type(self):
        judge = self._judge('{"relation": "unrelated", "confidence": 0.7, "explanation": "different"}')
        verdict = judge.compare("a", "b")

        assert verdict.relation == "unrelated"
        assert verdict.edge_type is None

    def test_unknown_relation_is_rejected_rather_than_coerced(self):
        judge = self._judge('{"relation": "maybe", "explanation": "unsure"}')
        assert judge.compare("a", "b") is None

    def test_malformed_json_returns_no_verdict(self):
        judge = self._judge("I think they disagree.")
        assert judge.compare("a", "b") is None

    def test_missing_confidence_falls_back_to_the_default(self):
        judge = ContradictionJudge(
            ScriptedCaptioner('{"relation": "contradicts", "explanation": "x"}'),
            GraphSettings(contradiction_default_confidence=0.42),
        )
        assert judge.compare("a", "b").confidence == 0.42

    def test_out_of_range_confidence_falls_back_to_the_default(self):
        judge = ContradictionJudge(
            ScriptedCaptioner('{"relation": "contradicts", "confidence": 95, "explanation": "x"}'),
            GraphSettings(contradiction_default_confidence=0.42),
        )
        assert judge.compare("a", "b").confidence == 0.42

    def test_empty_claim_never_calls_the_model(self):
        captioner = ScriptedCaptioner("unused")
        judge = ContradictionJudge(captioner, GraphSettings())

        assert judge.compare("", "a claim") is None
        assert captioner.prompts == []


# --------------------------------------------------------------------------
# Phase 5: candidate pairing (pure) and the similarity pre-filter
# --------------------------------------------------------------------------

class TestBuildCandidatePairs:
    def test_entity_co_mentions_become_pairs(self):
        pairs = build_candidate_pairs(
            entity_groups=[("e1", "knife", ["n1", "n2", "n3"])],
            alignment_pairs=[], event_groups=[], settings=GraphSettings(),
        )

        assert {(p.node_a_id, p.node_b_id) for p in pairs} == {
            ("n1", "n2"), ("n1", "n3"), ("n2", "n3")
        }
        assert all("entity:knife" in p.origins for p in pairs)

    def test_alignment_edges_become_pairs(self):
        pairs = build_candidate_pairs(
            entity_groups=[], alignment_pairs=[("n1", "n2", "ALIGNS_WITH")],
            event_groups=[], settings=GraphSettings(),
        )

        assert len(pairs) == 1
        assert pairs[0].origins == frozenset({"ALIGNS_WITH"})

    def test_event_groups_become_all_internal_pairs(self):
        pairs = build_candidate_pairs(
            entity_groups=[], alignment_pairs=[],
            event_groups=[("ev1", ["n1", "n2", "n3"])], settings=GraphSettings(),
        )

        assert len(pairs) == 3
        assert all(p.origins == frozenset({"SAME_EVENT"}) for p in pairs)

    def test_the_same_pair_from_two_sources_is_deduplicated_with_both_origins(self):
        pairs = build_candidate_pairs(
            entity_groups=[("e1", "knife", ["n1", "n2"])],
            alignment_pairs=[("n2", "n1", "REFERENCES")],
            event_groups=[("ev1", ["n1", "n2"])],
            settings=GraphSettings(),
        )

        assert len(pairs) == 1
        assert pairs[0].origins == frozenset({"entity:knife", "REFERENCES", "SAME_EVENT"})

    def test_pairs_are_stored_in_canonical_order(self):
        """(b, a) and (a, b) are one comparison, not two."""
        pairs = build_candidate_pairs(
            entity_groups=[], alignment_pairs=[("n9", "n1", "ALIGNS_WITH")],
            event_groups=[], settings=GraphSettings(),
        )

        assert (pairs[0].node_a_id, pairs[0].node_b_id) == ("n1", "n9")

    def test_a_node_is_never_paired_with_itself(self):
        pairs = build_candidate_pairs(
            entity_groups=[], alignment_pairs=[("n1", "n1", "ALIGNS_WITH")],
            event_groups=[], settings=GraphSettings(),
        )

        assert pairs == []

    def test_a_very_common_entity_is_capped(self):
        node_ids = [f"n{i:02d}" for i in range(10)]
        settings = GraphSettings(max_nodes_per_entity_for_pairs=3)

        pairs = build_candidate_pairs(
            entity_groups=[("e1", "person", node_ids)],
            alignment_pairs=[], event_groups=[], settings=settings,
        )

        # 3 capped nodes -> 3 pairs, not the 45 the full set would produce.
        assert len(pairs) == 3


def _claim(node_id, claim, embedding=None, node_type="scene_segment"):
    return ClaimRecord(id=node_id, node_type=node_type, claim=claim, text_embedding=embedding)


class TestFilterPairsBySimilarity:
    def _pairs(self):
        return build_candidate_pairs(
            entity_groups=[], alignment_pairs=[("n1", "n2", "ALIGNS_WITH")],
            event_groups=[], settings=GraphSettings(),
        )

    def test_similar_claims_survive(self):
        claims = {
            "n1": _claim("n1", "No weapon was present.", _unit([1.0, 0.0, 0.0])),
            "n2": _claim("n2", "A knife is visible.", _unit([0.9, 0.1, 0.0])),
        }
        survivors, skipped = filter_pairs_by_similarity(
            self._pairs(), claims, GraphSettings(contradiction_similarity_threshold=0.3)
        )

        assert len(survivors) == 1
        assert skipped == {"no_claim": 0, "below_similarity": 0}

    def test_dissimilar_claims_are_filtered_before_any_llm_call(self):
        claims = {
            "n1": _claim("n1", "The car was red.", _unit([1.0, 0.0, 0.0])),
            "n2": _claim("n2", "The meeting was on Tuesday.", _unit([0.0, 1.0, 0.0])),
        }
        survivors, skipped = filter_pairs_by_similarity(
            self._pairs(), claims, GraphSettings(contradiction_similarity_threshold=0.3)
        )

        assert survivors == []
        assert skipped["below_similarity"] == 1

    def test_a_pair_missing_a_claim_is_dropped(self):
        claims = {"n1": _claim("n1", "A knife is visible.")}
        survivors, skipped = filter_pairs_by_similarity(self._pairs(), claims, GraphSettings())

        assert survivors == []
        assert skipped["no_claim"] == 1

    def test_a_missing_embedding_keeps_the_pair(self):
        """The pre-filter is an optimisation; a missing vector is not evidence
        that two claims agree, so the pair still goes to the LLM."""
        claims = {
            "n1": _claim("n1", "No weapon was present.", None),
            "n2": _claim("n2", "A knife is visible.", _unit([1.0, 0.0, 0.0])),
        }
        survivors, _ = filter_pairs_by_similarity(self._pairs(), claims, GraphSettings())

        assert len(survivors) == 1


# --------------------------------------------------------------------------
# Phase 5: claim extraction + contradiction orchestration (stub repository)
# --------------------------------------------------------------------------

class StubClaimNode:
    def __init__(self, id, text_content, node_type="scene_segment"):
        self.id = id
        self.node_type = node_type
        self.text_content = text_content


class FakeClaimRepository:
    def __init__(self, candidates):
        self._candidates = candidates
        self.stored: dict[str, str | None] = {}

    def fetch_claim_candidates(self, case_id, node_types, only_pending=True):
        return self._candidates

    def store_claim(self, node_id, claim):
        self.stored[node_id] = claim

    def commit(self):
        pass


class StubClaimExtractor:
    available = True
    unavailable_reason = None

    def __init__(self, claims_by_text):
        self._claims = claims_by_text

    def extract(self, text):
        return self._claims.get(text)


class TestExtractClaims:
    def test_stores_a_claim_per_node(self):
        repo = FakeClaimRepository([
            StubClaimNode("n1", "the witness saw nothing"),
            StubClaimNode("n2", "a knife lay on the table"),
        ])
        extractor = StubClaimExtractor({
            "the witness saw nothing": "The witness saw no weapon.",
            "a knife lay on the table": "A knife is on the table.",
        })

        extracted = extract_claims(repo, extractor, "case-1", GraphSettings())

        assert extracted == 2
        assert repo.stored["n1"] == "The witness saw no weapon."

    def test_a_node_with_no_claim_is_still_marked_attempted(self):
        """Otherwise every run re-prompts the model for text it already
        determined asserts nothing."""
        repo = FakeClaimRepository([StubClaimNode("n1", "page header")])
        extractor = StubClaimExtractor({})

        extracted = extract_claims(repo, extractor, "case-1", GraphSettings())

        assert extracted == 0
        assert repo.stored == {"n1": None}

    def test_unavailable_model_skips_cleanly(self):
        repo = FakeClaimRepository([StubClaimNode("n1", "text")])
        extractor = StubClaimExtractor({})
        extractor.available = False
        extractor.unavailable_reason = "ollama down"

        assert extract_claims(repo, extractor, "case-1", GraphSettings()) == 0
        assert repo.stored == {}


class FakeContradictionRepository:
    def __init__(self, claims, entity_groups=None, alignment_pairs=None, event_groups=None):
        self._claims = claims
        self._entity_groups = entity_groups or []
        self._alignment_pairs = alignment_pairs or []
        self._event_groups = event_groups or []
        self.cleared = False
        self.relationships: list[dict] = []

    def clear_claim_relationships(self, case_id):
        self.cleared = True
        self.relationships = []

    def fetch_claims(self, case_id):
        return self._claims

    def fetch_entity_node_groups(self, case_id):
        return self._entity_groups

    def fetch_alignment_pairs(self, case_id, alignment_types):
        return self._alignment_pairs

    def fetch_event_node_groups(self, case_id):
        return self._event_groups

    def insert_claim_relationship(self, case_id, subject_node_id, object_node_id,
                                  relationship_type, confidence, explanation, metadata=None):
        self.relationships.append({
            "subject": subject_node_id, "object": object_node_id,
            "type": relationship_type, "confidence": confidence,
            "explanation": explanation, "metadata": metadata or {},
        })
        return True

    def commit(self):
        pass


class ScriptedJudge:
    available = True
    unavailable_reason = None

    def __init__(self, verdict):
        self._verdict = verdict
        self.comparisons: list[tuple[str, str]] = []

    def compare(self, claim_a, claim_b):
        self.comparisons.append((claim_a, claim_b))
        return self._verdict


class TestDetectContradictions:
    def _witness_vs_video(self):
        """The scenario from the phase 5 spec: a PDF witness statement denying
        a weapon, and a video segment showing a knife, linked because both
        mention the same 'knife' entity."""
        claims = {
            "pdf-page": _claim(
                "pdf-page", "The witness states no weapon was present.",
                _unit([1.0, 0.1, 0.0]), node_type="page",
            ),
            "video-seg": _claim(
                "video-seg", "The suspect is holding a knife.",
                _unit([0.9, 0.2, 0.0]), node_type="scene_segment",
            ),
        }
        return FakeContradictionRepository(
            claims, entity_groups=[("e1", "knife", ["pdf-page", "video-seg"])]
        )

    def test_contradicting_claims_produce_a_contradicts_edge(self):
        repo = self._witness_vs_video()
        judge = ScriptedJudge(
            ClaimVerdict("contradicts", 0.92, "One denies a weapon the other shows.")
        )

        report = detect_contradictions(repo, judge, "case-1", GraphSettings())

        assert report.contradicts == 1
        assert report.corroborates == 0
        assert len(repo.relationships) == 1
        edge = repo.relationships[0]
        assert edge["type"] == "CONTRADICTS"
        assert {edge["subject"], edge["object"]} == {"pdf-page", "video-seg"}
        assert edge["confidence"] == 0.92
        assert "denies" in edge["explanation"]

    def test_the_edge_records_why_the_pair_was_compared(self):
        repo = self._witness_vs_video()
        judge = ScriptedJudge(ClaimVerdict("contradicts", 0.9, "conflict"))

        detect_contradictions(repo, judge, "case-1", GraphSettings())

        metadata = repo.relationships[0]["metadata"]
        assert metadata["origins"] == ["entity:knife"]
        assert metadata["claim_a"] == "The witness states no weapon was present."
        assert metadata["claim_b"] == "The suspect is holding a knife."

    def test_corroborating_claims_produce_a_corroborates_edge(self):
        repo = self._witness_vs_video()
        judge = ScriptedJudge(ClaimVerdict("corroborates", 0.8, "Both describe a knife."))

        report = detect_contradictions(repo, judge, "case-1", GraphSettings())

        assert report.corroborates == 1
        assert repo.relationships[0]["type"] == "CORROBORATES"

    def test_unrelated_claims_produce_no_edge(self):
        repo = self._witness_vs_video()
        judge = ScriptedJudge(ClaimVerdict("unrelated", 0.7, "Different subjects."))

        report = detect_contradictions(repo, judge, "case-1", GraphSettings())

        assert report.unrelated == 1
        assert repo.relationships == []

    def test_dissimilar_pairs_never_reach_the_judge(self):
        claims = {
            "n1": _claim("n1", "The car was red.", _unit([1.0, 0.0, 0.0])),
            "n2": _claim("n2", "The meeting was Tuesday.", _unit([0.0, 1.0, 0.0])),
        }
        repo = FakeContradictionRepository(
            claims, entity_groups=[("e1", "thing", ["n1", "n2"])]
        )
        judge = ScriptedJudge(ClaimVerdict("contradicts", 0.9, "should never be asked"))

        report = detect_contradictions(repo, judge, "case-1", GraphSettings())

        assert judge.comparisons == []
        assert report.pairs_judged == 0
        assert report.skipped["below_similarity"] == 1

    def test_rerun_clears_prior_verdicts_first(self):
        repo = self._witness_vs_video()
        detect_contradictions(
            repo, ScriptedJudge(ClaimVerdict("contradicts", 0.9, "x")), "case-1", GraphSettings()
        )
        assert repo.cleared is True

    def test_unavailable_judge_skips_cleanly(self):
        repo = self._witness_vs_video()
        judge = ScriptedJudge(None)
        judge.available = False
        judge.unavailable_reason = "ollama down"

        report = detect_contradictions(repo, judge, "case-1", GraphSettings())

        assert report.contradicts == 0
        assert repo.cleared is False  # nothing was touched

    def test_exceeding_the_pair_cap_raises(self):
        node_ids = [f"n{i}" for i in range(10)]
        claims = {n: _claim(n, f"claim {n}", _unit([1.0, 0.0, 0.0])) for n in node_ids}
        repo = FakeContradictionRepository(
            claims, entity_groups=[("e1", "thing", node_ids)]
        )
        judge = ScriptedJudge(ClaimVerdict("unrelated", 0.5, "x"))

        with pytest.raises(ValueError, match="exceeds"):
            detect_contradictions(repo, judge, "case-1", GraphSettings(max_contradiction_pairs=5))
