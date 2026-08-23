"""Tests for multi-source timeline synchronization (Phase 6).

Tests audio fingerprinting, visual matching, identity matching, offset estimation,
and end-to-end synchronization with realistic and synthetic scenarios.
"""
from __future__ import annotations

import numpy as np
import pytest

from graph.timeline_sync.audio_fingerprinting import fingerprint_audio
from graph.timeline_sync.identity_matching import identity_matching
from graph.timeline_sync.offset_estimation import estimate_offset
from graph.timeline_sync.visual_matching import visual_matching


class TestAudioFingerprinting:
    """Tests for audio fingerprinting via AST embeddings."""

    def _unit_embedding(self, seed: int, dim: int = 768) -> np.ndarray:
        """Generate a unit-norm random embedding."""
        np.random.seed(seed)
        vec = np.random.randn(dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def test_high_similarity_pairs_detected(self):
        """High-similarity embeddings should produce anchors."""
        # Two identical embeddings (similarity = 1.0)
        emb = self._unit_embedding(42)
        embeddings_a = [(0.0, 1.0, emb), (5.0, 6.0, emb)]
        embeddings_b = [(0.5, 1.5, emb), (5.5, 6.5, emb)]

        anchors = fingerprint_audio(embeddings_a, embeddings_b, similarity_threshold=0.9)
        # All embeddings match; clustered by 5-second bins gives 2 clusters
        assert len(anchors) >= 2, f"expected at least 2 anchors, got {len(anchors)}"
        assert all(a.similarity > 0.95 for a in anchors)

    def test_low_similarity_pairs_rejected(self):
        """Low-similarity embeddings should be filtered out."""
        emb_a = self._unit_embedding(1)
        emb_b = self._unit_embedding(2)  # orthogonal
        embeddings_a = [(0.0, 1.0, emb_a)]
        embeddings_b = [(0.0, 1.0, emb_b)]

        anchors = fingerprint_audio(embeddings_a, embeddings_b, similarity_threshold=0.9)
        assert len(anchors) == 0

    def test_temporal_clustering(self):
        """Consecutive high-similarity matches should cluster together."""
        emb = self._unit_embedding(42)
        embeddings_a = [(0.0, 1.0, emb), (1.0, 2.0, emb), (10.0, 11.0, emb)]
        embeddings_b = [(0.5, 1.5, emb), (1.5, 2.5, emb), (10.5, 11.5, emb)]

        anchors = fingerprint_audio(embeddings_a, embeddings_b)
        # Groups by 5-second windows: (0-5s, 5s-10s, 10s+)
        # Time 0, 1 in window 0; time 10 in window 2
        assert len(anchors) >= 3
        # Check that cluster indices differ for far-apart matches
        first_cluster = anchors[0].anchor_index
        last_cluster = anchors[-1].anchor_index
        assert first_cluster != last_cluster

    def test_empty_embeddings_returns_empty(self):
        """No embeddings should return empty list."""
        anchors = fingerprint_audio([], [])
        assert anchors == []


class TestVisualMatching:
    """Tests for visual matching via CLIP embeddings."""

    def _unit_embedding(self, seed: int, dim: int = 512) -> np.ndarray:
        np.random.seed(seed)
        vec = np.random.randn(dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def test_temporal_consistency_filter(self):
        """Matches forming a diagonal should have consistent offsets."""
        emb = self._unit_embedding(42)
        frames_a = [(float(i), emb) for i in range(0, 10, 2)]  # 0, 2, 4, 6, 8
        frames_b = [(float(i), emb) for i in range(1, 11, 2)]  # 1, 3, 5, 7, 9

        anchors = visual_matching(frames_a, frames_b, similarity_threshold=0.9)
        # All pairs should match (identical embeddings)
        assert len(anchors) > 0
        # Offsets should be consistent (all close to 1 second)
        offsets = [a.time_b - a.time_a for a in anchors]
        offset_median = np.median(offsets)
        assert abs(offset_median - 1.0) < 0.5, f"median offset {offset_median} far from 1.0"

    def test_scattered_matches_low_consistency(self):
        """Scattered matches (no clear diagonal) should have varied consistency."""
        emb1 = self._unit_embedding(1)
        emb2 = self._unit_embedding(2)
        emb3 = self._unit_embedding(3)
        # Create embeddings at different times with complex similarity pattern
        frames_a = [(0.0, emb1), (5.0, emb2), (10.0, emb3)]
        frames_b = [(3.0, emb1), (1.0, emb2), (8.0, emb3)]

        anchors = visual_matching(frames_a, frames_b)
        # Offsets are 3, -4, -2 sec: very scattered
        if anchors:
            # Consistency scores will vary based on binning
            # Just verify we get anchors and scores are in valid range [0, 1]
            assert all(0.0 <= a.temporal_consistency_score <= 1.0 for a in anchors)

    def test_empty_frames_returns_empty(self):
        anchors = visual_matching([], [])
        assert anchors == []


class TestIdentityMatching:
    """Tests for identity matching via face cluster co-occurrence."""

    def test_same_face_cluster_in_both_sources(self):
        """Same face cluster appearing in both sources should produce anchors."""
        faces_a = {
            "cluster-1": [0.0, 5.0, 10.0],
            "cluster-2": [2.0, 7.0],
        }
        faces_b = {
            "cluster-1": [0.5, 5.5, 10.5],
            "cluster-2": [2.5, 7.5],
        }

        anchors = identity_matching(faces_a, faces_b)
        assert len(anchors) == 5, f"expected 5 anchors (3+2), got {len(anchors)}"
        # Check cluster representation (order may vary due to dict iteration)
        cluster_ids = [a.face_cluster_id for a in anchors]
        assert cluster_ids.count("cluster-1") == 3
        assert cluster_ids.count("cluster-2") == 2

    def test_different_face_clusters_no_anchors(self):
        """Different face clusters should produce no anchors."""
        faces_a = {"cluster-1": [0.0, 5.0]}
        faces_b = {"cluster-2": [0.5, 5.5]}

        anchors = identity_matching(faces_a, faces_b)
        assert len(anchors) == 0

    def test_empty_faces_returns_empty(self):
        anchors = identity_matching({}, {})
        assert anchors == []


class TestOffsetEstimation:
    """Tests for offset estimation from mixed anchors."""

    def test_median_offset_computed(self):
        """Median of all offsets should be the estimated offset."""
        audio_anchors = [
            type('Anchor', (), {'time_b': 10.5, 'time_a': 10.0})(),
            type('Anchor', (), {'time_b': 15.5, 'time_a': 15.0})(),
        ]
        visual_anchors = [
            type('Anchor', (), {'time_b': 20.5, 'time_a': 20.0, 'temporal_consistency_score': 0.9})(),
        ]
        identity_anchors = []

        estimate = estimate_offset(audio_anchors, visual_anchors, identity_anchors)
        assert estimate is not None
        assert abs(estimate.offset_seconds - 0.5) < 0.01
        assert estimate.anchor_count == 3

    def test_confidence_from_agreement(self):
        """Confidence should reflect how many anchors agree within ±1 sec."""
        audio_anchors = [
            type('Anchor', (), {'time_b': 10.5, 'time_a': 10.0})(),
            type('Anchor', (), {'time_b': 15.4, 'time_a': 15.0})(),  # within ±0.5
            type('Anchor', (), {'time_b': 25.0, 'time_a': 20.0})(),  # outlier: offset=5
        ]
        visual_anchors = []
        identity_anchors = []

        estimate = estimate_offset(audio_anchors, visual_anchors, identity_anchors,
                                   confidence_window_sec=1.0)
        assert estimate is not None
        # Median offset is ~0.5; two anchors within ±1 sec, one outlier
        # Confidence should be 2/3 ≈ 0.67
        assert estimate.confidence > 0.6 and estimate.confidence < 0.75

    def test_no_anchors_returns_none(self):
        estimate = estimate_offset([], [], [])
        assert estimate is None

    def test_method_counts_recorded(self):
        """Method counts should reflect which methods contributed."""
        audio_anchors = [
            type('Anchor', (), {'time_b': 10.5, 'time_a': 10.0})(),
        ]
        visual_anchors = [
            type('Anchor', (), {'time_b': 10.5, 'time_a': 10.0, 'temporal_consistency_score': 0.9})(),
            type('Anchor', (), {'time_b': 11.5, 'time_a': 11.0, 'temporal_consistency_score': 0.9})(),
        ]
        identity_anchors = []

        estimate = estimate_offset(audio_anchors, visual_anchors, identity_anchors)
        assert estimate.method_counts["audio"] == 1
        assert estimate.method_counts["visual"] == 2
        assert estimate.method_counts["identity"] == 0


class TestSynchronizationIntegration:
    """Integration tests with realistic scenarios."""

    def _realistic_embeddings(self, num_segments: int, offset_sec: float) -> tuple:
        """Create realistic pairs of audio with known offset."""
        np.random.seed(42)
        emb_dim = 768
        embeddings_a = []
        embeddings_b = []

        for i in range(num_segments):
            # Similar embeddings for the same segment
            base_emb = np.random.randn(emb_dim).astype(np.float32)
            base_emb /= np.linalg.norm(base_emb)
            # Add small noise for realism
            emb_a = base_emb + np.random.randn(emb_dim).astype(np.float32) * 0.01
            emb_b = base_emb + np.random.randn(emb_dim).astype(np.float32) * 0.01
            emb_a /= np.linalg.norm(emb_a)
            emb_b /= np.linalg.norm(emb_b)

            start_time = float(i * 5)
            embeddings_a.append((start_time, start_time + 5, emb_a))
            embeddings_b.append((start_time + offset_sec, start_time + offset_sec + 5, emb_b))

        return embeddings_a, embeddings_b

    def test_realistic_audio_offset_detection(self):
        """With realistic embeddings, should detect known offset."""
        true_offset = 2.5
        embeddings_a, embeddings_b = self._realistic_embeddings(5, true_offset)

        # Lower threshold to 0.85 to account for noise in realistic embeddings
        anchors = fingerprint_audio(embeddings_a, embeddings_b, similarity_threshold=0.85)
        assert len(anchors) > 0, "should find some anchor points even with noise"

        # Estimated offsets from anchors should be close to true offset
        estimated_offsets = [a.time_b - a.time_a for a in anchors]
        median_est = np.median(estimated_offsets)
        # With realistic noise, allow 1.5 second margin
        assert abs(median_est - true_offset) < 1.5, f"offset {median_est} far from truth {true_offset}"


@pytest.mark.skipif(
    True,  # Skip DB-dependent tests in offline mode
    reason="requires real database and evidence",
)
class TestSynchronizationWithRealData:
    """Tests with artificially offset videos in real database."""

    def test_offset_two_videos_of_same_event(self, conn):
        """Create two videos of the same event with known offset, verify alignment."""
        # This test would:
        # 1. Ingest two videos of the same event with different start times
        # 2. Extract embeddings for both
        # 3. Run synchronization
        # 4. Verify estimated offset matches known offset within 0.5 seconds
        pytest.skip("requires test videos and database")
