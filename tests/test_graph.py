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
from graph.entities import build_entities_and_mentions
from graph.extraction.detections import entities_from_detections
from graph.extraction.entities import EntityExtractor, ExtractedEntity, normalize_name
from graph.repository import TimeWindow


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
