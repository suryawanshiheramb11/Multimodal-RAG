"""Enrichment tests that run without downloading a single model.

Every model is stubbed, which is the payoff of the LazyModel/registry seam:
the orchestration, degradation, and fusion logic can all be exercised offline.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from enrichment.analyzers import AnalyzerRegistry, build_analyzer_registry
from enrichment.analyzers.base import EnrichmentResult, NodeAnalyzer, PendingNode
from enrichment.analyzers.text_fusion import build_text_embedding, fuse_text
from enrichment.analyzers.visual import VisualExtractor
from enrichment.config import EnrichmentSettings
from enrichment.errors import AnalyzerNotRegisteredError
from enrichment.models.base import LazyModel
from enrichment.models.clip import ViolenceScore
from enrichment.models.detection import Detection, ObjectDetector
from enrichment.pipeline import EnrichmentPipeline
from enrichment.registry import ModelRegistry

# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------

class StubModel:
    """Minimal stand-in matching LazyModel's availability surface."""

    def __init__(self, available: bool = True, reason: str | None = None) -> None:
        self._available = available
        self.unavailable_reason = reason if not available else None

    @property
    def available(self) -> bool:
        return self._available


class StubClip(StubModel):
    def __init__(self, available=True, reason=None, dim=512):
        super().__init__(available, reason)
        self.dim = dim

    def embed_image(self, path):
        return np.ones(self.dim, dtype=np.float32) if self._available else None

    def embed_text(self, text):
        return np.ones(self.dim, dtype=np.float32) if self._available else None

    def score_violence(self, frames, prompts, violent_count):
        if not self._available:
            return None
        return ViolenceScore(
            score=0.75,
            frames_scored=len(frames),
            per_prompt=dict.fromkeys(prompts, 0.2),
        )


class StubDetector(StubModel):
    def __init__(self, available=True, reason=None, detections=None):
        super().__init__(available, reason)
        self._detections = detections or []

    def detect(self, frames):
        return self._detections

    summarise = staticmethod(ObjectDetector.summarise)


class StubCaptioner(StubModel):
    def __init__(self, available=True, reason=None, text="a person holding a knife"):
        super().__init__(available, reason)
        self._text = text

    def caption(self, path, prompt):
        return self._text if self._available else None


class StubOcr(StubModel):
    def __init__(self, available=True, reason=None, text="EXIT"):
        super().__init__(available, reason)
        self._text = text

    def read(self, path):
        if not self._available:
            return None
        from enrichment.models.ocr import OcrResult

        return OcrResult(text=self._text, line_count=1, mean_confidence=0.9)


class StubTextEncoder(StubModel):
    def __init__(self, available=True, reason=None, dim=384):
        super().__init__(available, reason)
        self.dim = dim

    def embed(self, text):
        return np.ones(self.dim, dtype=np.float32) if self._available and text else None


class StubTranscriber(StubModel):
    def __init__(self, available=True, reason=None, text="put the weapon down"):
        super().__init__(available, reason)
        self._text = text

    def transcribe_file(self, path):
        if not self._available:
            return None
        from enrichment.models.asr import Transcript, TranscriptSegment

        return Transcript(
            segments=[TranscriptSegment(start=0.0, end=4.0, text=self._text)],
            language="en",
            language_probability=0.99,
        )


class StubAudioEvents(StubModel):
    def __init__(self, available=True, reason=None, dim=768):
        super().__init__(available, reason)
        self.dim = dim

    def analyse(self, path, start=None, end=None, top_k=5):
        if not self._available:
            return None
        from enrichment.models.audio_events import AudioAnalysis, AudioEvent

        return AudioAnalysis(
            events=[AudioEvent(label="Speech", probability=0.9)],
            embedding=np.ones(self.dim, dtype=np.float32),
        )


def stub_registry(**overrides) -> ModelRegistry:
    defaults = {
        "transcriber": StubTranscriber(),
        "audio_events": StubAudioEvents(),
        "clip": StubClip(),
        "detector": StubDetector(),
        "captioner": StubCaptioner(),
        "ocr": StubOcr(),
        "text_encoder": StubTextEncoder(),
    }
    defaults.update(overrides)
    return ModelRegistry(**defaults)


@pytest.fixture
def frames(tmp_path) -> list[Path]:
    from PIL import Image

    paths = []
    for i in range(4):
        path = tmp_path / f"seg0000_t{i:05d}.jpg"
        Image.new("RGB", (32, 24), color=(i * 40, 0, 0)).save(path)
        paths.append(path)
    return paths


def make_node(node_type="scene_segment", frames=None, audio=None, **kwargs) -> PendingNode:
    metadata = kwargs.pop("metadata", {})
    if frames:
        metadata["frames"] = [
            {"timestamp": float(i), "path": str(p)} for i, p in enumerate(frames)
        ]
    if audio:
        metadata["audio_path"] = str(audio)
    return PendingNode(
        id=kwargs.pop("id", "node-1"),
        node_type=node_type,
        source_file_id="src-1",
        source_path=Path("/evidence/clip.mp4"),
        source_type="video",
        start_time=kwargs.pop("start_time", 0.0),
        end_time=kwargs.pop("end_time", 5.0),
        page_number=kwargs.pop("page_number", None),
        text_content=kwargs.pop("text_content", None),
        file_path=kwargs.pop("file_path", None),
        metadata=metadata,
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

class TestLazyModel:
    def test_failed_load_is_reported_not_raised(self):
        class Broken(LazyModel):
            name = "broken"

            def _build(self):
                raise RuntimeError("no checkpoint")

        model = Broken()
        assert model.available is False
        assert "no checkpoint" in model.unavailable_reason

    def test_load_is_attempted_only_once(self):
        class Counting(LazyModel):
            name = "counting"
            builds = 0

            def _build(self):
                Counting.builds += 1
                raise RuntimeError("still broken")

        model = Counting()
        for _ in range(5):
            assert model.available is False
        assert Counting.builds == 1

    def test_successful_load_is_cached(self):
        class Once(LazyModel):
            name = "once"
            builds = 0

            def _build(self):
                Once.builds += 1
                return object()

        model = Once()
        first, second = model.load(), model.load()
        assert first is second
        assert Once.builds == 1


class TestTextFusion:
    def test_sections_are_labelled(self):
        text = fuse_text(transcript="drop it", caption="a knife", ocr="EXIT")
        assert "Transcript: drop it" in text
        assert "Visual description: a knife" in text
        assert "On-screen text: EXIT" in text

    def test_empty_sources_are_omitted(self):
        assert fuse_text(transcript="hello", caption=None, ocr="   ") == "Transcript: hello"

    def test_all_empty_returns_none(self):
        assert fuse_text() is None
        assert fuse_text(transcript="", caption=None) is None

    def test_embedding_skipped_when_no_text(self):
        result = EnrichmentResult()
        build_text_embedding(stub_registry(), result)
        assert result.text_embedding is None
        assert "no text" in result.skipped["text_embedding"]

    def test_embedding_records_encoder_unavailability(self):
        result = EnrichmentResult(text_content="something")
        registry = stub_registry(
            text_encoder=StubTextEncoder(available=False, reason="offline")
        )
        build_text_embedding(registry, result)
        assert result.text_embedding is None
        assert result.skipped["text_embedding"] == "offline"


class TestPendingNode:
    def test_representative_frame_is_the_middle_one(self, frames):
        node = make_node(frames=frames)
        assert node.representative_frame() == frames[2]

    def test_no_frames_gives_no_representative(self):
        assert make_node().representative_frame() is None

    def test_audio_path_from_metadata(self, tmp_path):
        node = make_node(audio=tmp_path / "a.wav")
        assert node.audio_path == tmp_path / "a.wav"

    def test_audio_track_falls_back_to_file_path(self, tmp_path):
        node = make_node(node_type="audio_track", file_path=str(tmp_path / "b.wav"))
        assert node.audio_path == tmp_path / "b.wav"


class TestVisualExtractor:
    def test_frame_budget_is_respected(self, frames):
        settings = EnrichmentSettings(max_frames_analyzed=2)
        extractor = VisualExtractor(stub_registry(), settings)

        features = extractor.extract(frames, frames[1], EnrichmentResult())

        assert features.metadata["frames_analyzed"] == 2
        assert features.metadata["frames_available"] == 4

    def test_missing_frames_are_reported(self, tmp_path):
        result = EnrichmentResult()
        extractor = VisualExtractor(stub_registry(), EnrichmentSettings())

        features = extractor.extract([tmp_path / "gone.jpg"], None, result)

        assert features.clip_embedding is None
        assert "no frames available" in result.skipped["visual"]

    def test_all_visual_features_populate(self, frames):
        result = EnrichmentResult()
        registry = stub_registry(
            detector=StubDetector(detections=[
                Detection("knife", 0.9, (1, 2, 3, 4), str(frames[0]))
            ])
        )
        features = VisualExtractor(registry, EnrichmentSettings()).extract(
            frames, frames[2], result
        )

        assert features.caption == "a person holding a knife"
        assert features.ocr_text == "EXIT"
        assert features.clip_embedding.shape == (512,)
        assert features.metadata["violence"]["score"] == 0.75
        assert features.metadata["detections"]["notable"] == ["knife"]
        assert result.skipped == {}

    def test_unavailable_models_are_recorded_individually(self, frames):
        result = EnrichmentResult()
        registry = stub_registry(
            captioner=StubCaptioner(available=False, reason="ollama down"),
            ocr=StubOcr(available=False, reason="paddle missing"),
        )
        features = VisualExtractor(registry, EnrichmentSettings()).extract(
            frames, frames[0], result
        )

        assert result.skipped["caption"] == "ollama down"
        assert result.skipped["ocr"] == "paddle missing"
        # The models that *are* available still ran.
        assert features.clip_embedding is not None
        assert "violence" in features.metadata

    def test_disabled_features_are_marked_disabled(self, frames):
        result = EnrichmentResult()
        settings = EnrichmentSettings(enable_caption=False, enable_detection=False)
        VisualExtractor(stub_registry(), settings).extract(frames, frames[0], result)

        assert result.skipped["caption"] == "disabled in config"
        assert result.skipped["detection"] == "disabled in config"


class TestAnalyzers:
    def test_video_segment_populates_everything(self, frames, tmp_path):
        audio = tmp_path / "track.wav"
        audio.write_bytes(b"RIFF")
        node = make_node(frames=frames, audio=audio)
        analyzers = build_analyzer_registry(stub_registry(), EnrichmentSettings())

        result = analyzers.get("scene_segment").analyze(node)

        assert "put the weapon down" in result.text_content
        assert "a person holding a knife" in result.text_content
        assert result.text_embedding.shape == (384,)
        assert result.clip_embedding.shape == (512,)
        assert result.audio_embedding.shape == (768,)
        assert result.metadata["audio_events"][0]["label"] == "Speech"

    def test_video_segment_without_audio_still_analyses_visuals(self, frames):
        node = make_node(frames=frames)
        analyzers = build_analyzer_registry(stub_registry(), EnrichmentSettings())

        result = analyzers.get("scene_segment").analyze(node)

        assert result.skipped["asr"] == "segment has no audio track"
        assert result.audio_embedding is None
        assert result.clip_embedding is not None

    def test_image_analyzer_has_no_temporal_features(self, frames):
        node = make_node(node_type="image", file_path=str(frames[0]))
        analyzers = build_analyzer_registry(stub_registry(), EnrichmentSettings())

        result = analyzers.get("image").analyze(node)

        assert result.audio_embedding is None
        assert "transcript" not in result.metadata
        assert result.clip_embedding is not None
        assert "a person holding a knife" in result.text_content

    def test_page_with_text_layer_skips_ocr(self, tmp_path):
        from PIL import Image

        page = tmp_path / "p1.png"
        Image.new("RGB", (40, 50)).save(page)
        node = make_node(
            node_type="page", file_path=str(page),
            text_content="A long embedded text layer that clearly came from the PDF itself.",
        )
        analyzers = build_analyzer_registry(stub_registry(), EnrichmentSettings())

        result = analyzers.get("page").analyze(node)

        assert "sufficient embedded text" in result.metadata["ocr_skipped"]
        assert "ocr" not in result.metadata
        assert result.text_embedding is not None

    def test_scanned_page_gets_ocr(self, tmp_path):
        from PIL import Image

        page = tmp_path / "scan.png"
        Image.new("RGB", (40, 50)).save(page)
        node = make_node(node_type="page", file_path=str(page), text_content="")
        analyzers = build_analyzer_registry(stub_registry(), EnrichmentSettings())

        result = analyzers.get("page").analyze(node)

        assert result.metadata["ocr"]["text"] == "EXIT"
        assert "On-screen text: EXIT" in result.text_content

    def test_audio_track_analyzer(self, tmp_path):
        audio = tmp_path / "interview.wav"
        audio.write_bytes(b"RIFF")
        node = make_node(node_type="audio_track", file_path=str(audio))
        analyzers = build_analyzer_registry(stub_registry(), EnrichmentSettings())

        result = analyzers.get("audio_track").analyze(node)

        assert "put the weapon down" in result.text_content
        assert result.audio_embedding.shape == (768,)
        assert result.clip_embedding is None


class TestAnalyzerRegistry:
    def test_unregistered_node_type_raises(self):
        with pytest.raises(AnalyzerNotRegisteredError, match="frame"):
            AnalyzerRegistry([]).get("frame")

    def test_registry_covers_every_ingested_node_type(self):
        """The four node types phase 1 writes must all have an analyzer."""
        registry = build_analyzer_registry(stub_registry(), EnrichmentSettings())
        assert registry.supported_types == {
            "scene_segment", "audio_track", "image", "page"
        }


class TestPipelineOrchestration:
    class FakeRepository:
        def __init__(self, nodes):
            self._nodes = nodes
            self.saved = {}
            self.failed = {}
            self.runs = []

        def fetch_nodes(self, case_id, only_pending=True, node_types=None):
            return self._nodes

        def save(self, node, result):
            self.saved[node.id] = result

        def mark_failed(self, node, error):
            self.failed[node.id] = error

        def record_run(self, case_id, availability, settings):
            self.runs.append(availability)

        def coverage(self, case_id):
            return {"by_type": []}

    def _pipeline(self, repo, analyzers, registry=None):
        return EnrichmentPipeline(
            models=registry or stub_registry(),
            analyzers=analyzers,
            repository=repo,
            settings=EnrichmentSettings(),
        )

    def test_one_failing_node_does_not_abort_the_run(self, frames):
        class Exploding(NodeAnalyzer):
            node_type = "scene_segment"

            def analyze(self, node):
                if node.id == "bad":
                    raise ValueError("decoder exploded")
                return EnrichmentResult(text_content="fine")

        nodes = [
            make_node(id="good", frames=frames),
            make_node(id="bad", frames=frames),
        ]
        repo = self.FakeRepository(nodes)
        report = self._pipeline(repo, AnalyzerRegistry([Exploding()])).run("case-1")

        assert [o.node_id for o in report.ok] == ["good"]
        assert [o.node_id for o in report.failed] == ["bad"]
        assert "decoder exploded" in repo.failed["bad"]
        assert "good" in repo.saved

    def test_unknown_node_type_is_skipped_not_failed(self):
        nodes = [make_node(node_type="mystery")]
        repo = self.FakeRepository(nodes)

        report = self._pipeline(repo, AnalyzerRegistry([])).run("case-1")

        assert len(report.skipped) == 1
        assert "no analyzer" in report.skipped[0].detail
        assert repo.saved == {}

    def test_availability_is_recorded_for_the_run(self, frames):
        repo = self.FakeRepository([make_node(frames=frames)])
        registry = stub_registry(clip=StubClip(available=False, reason="no torch"))
        analyzers = build_analyzer_registry(registry, EnrichmentSettings())

        report = self._pipeline(repo, analyzers, registry).run("case-1")

        assert report.availability["clip"] == "no torch"
        assert report.availability["ocr"] == "ready"
        assert repo.runs[0]["clip"] == "no torch"

    def test_run_completes_with_every_model_unavailable(self, frames):
        """Total degradation: nothing extracted, but nothing crashes either."""
        registry = stub_registry(
            transcriber=StubTranscriber(available=False, reason="x"),
            audio_events=StubAudioEvents(available=False, reason="x"),
            clip=StubClip(available=False, reason="x"),
            detector=StubDetector(available=False, reason="x"),
            captioner=StubCaptioner(available=False, reason="x"),
            ocr=StubOcr(available=False, reason="x"),
            text_encoder=StubTextEncoder(available=False, reason="x"),
        )
        repo = self.FakeRepository([make_node(frames=frames)])
        analyzers = build_analyzer_registry(registry, EnrichmentSettings())

        report = self._pipeline(repo, analyzers, registry).run("case-1")

        assert len(report.ok) == 1
        assert report.failed == []
        saved = repo.saved["node-1"]
        assert saved.text_content is None
        assert saved.text_embedding is None
        assert set(saved.skipped) >= {"violence", "detection", "caption", "ocr"}


class TestDetectionSummary:
    def test_counts_and_notable_weapons(self):
        detections = [
            Detection("knife", 0.9, (0, 0, 1, 1), "f1.jpg"),
            Detection("knife", 0.7, (0, 0, 1, 1), "f2.jpg"),
            Detection("person", 0.95, (0, 0, 1, 1), "f1.jpg"),
        ]
        summary = ObjectDetector.summarise(detections)

        assert summary["total"] == 3
        assert summary["labels"]["knife"]["count"] == 2
        assert summary["labels"]["knife"]["max_confidence"] == 0.9
        assert summary["notable"] == ["knife"]

    def test_empty_detections(self):
        assert ObjectDetector.summarise([]) == {"labels": {}, "total": 0, "notable": []}


class TestOcrReaderIsolation:
    def test_ocr_reader_nonexistent_file(self):
        from enrichment.models.ocr import OcrReader
        reader = OcrReader(language="en")
        try:
            assert reader.read(Path("/nonexistent/image.png")) is None
        finally:
            reader.close()

    def _fake_reader(self, res_queue):
        """An OcrReader wired to in-memory queues, with no worker process."""
        import queue

        from enrichment.models.ocr import OcrReader

        reader = OcrReader(language="en")
        reader._req_q = queue.Queue()
        reader._res_q = res_queue
        return reader

    def test_reply_is_matched_to_its_own_request(self):
        """A late reply from a timed-out call must not answer the next one.

        The worker answers requests in order, but a call that gave up leaves
        its reply behind. If the client took whatever was at the head of the
        queue, every later frame would be captioned with the previous frame's
        text — wrong output, no error.
        """
        import queue

        res_q = queue.Queue()
        reader = self._fake_reader(res_q)
        # Reply to a request that has already been abandoned.
        res_q.put((999, "OK", [("text from an abandoned call", 0.9)]))
        res_q.put((1, "OK", [("the real answer", 0.8)]))

        assert reader._request("READ_PATH", "/x.png") == [("the real answer", 0.8)]
        assert res_q.empty()

    def test_request_times_out_instead_of_blocking_forever(self):
        import queue

        import enrichment.models.ocr as ocr_mod

        reader = self._fake_reader(queue.Queue())
        with patch.object(ocr_mod, "_READ_TIMEOUT_SEC", 0.05):
            with pytest.raises((TimeoutError, queue.Empty)):
                reader._request("READ_PATH", "/x.png")

    def test_worker_error_is_raised_not_returned_as_text(self):
        import queue

        res_q = queue.Queue()
        reader = self._fake_reader(res_q)
        res_q.put((1, "ERROR", "paddle exploded"))

        with pytest.raises(RuntimeError, match="paddle exploded"):
            reader._request("READ_PATH", "/x.png")

