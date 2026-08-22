"""Tests for processors, the registry, and segmentation."""
from __future__ import annotations

from typing import ClassVar

import pytest

from ingestion.errors import ResourceLimitError, UnsupportedMediaTypeError
from ingestion.media.scenes import SceneSegmenter
from ingestion.models import EvidenceNodeDraft, MediaType, NodeType, ScannedFile
from ingestion.processors.base import FileProcessor, ProcessorRegistry
from ingestion.processors.image import ImageProcessor
from ingestion.processors.pdf import PdfProcessor
from ingestion.workspace import Workspace


def make_source(path, media_type, sha="ab" * 32) -> ScannedFile:
    return ScannedFile(
        path=path,
        file_name=path.name,
        media_type=media_type,
        sha256=sha,
        size_bytes=path.stat().st_size if path.exists() else 0,
    )


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    ws = Workspace(tmp_path / "audio", tmp_path / "frames", tmp_path / "pages")
    ws.prepare()
    return ws


class TestRegistry:
    def test_dispatches_by_media_type(self):
        processor = ImageProcessor(max_pixels=1000)
        registry = ProcessorRegistry([processor])

        assert registry.get(MediaType.IMAGE) is processor
        assert registry.supports(MediaType.IMAGE)

    def test_unregistered_type_raises(self):
        registry = ProcessorRegistry([])
        assert not registry.supports(MediaType.DOC)
        with pytest.raises(UnsupportedMediaTypeError, match="doc"):
            registry.get(MediaType.DOC)

    def test_a_new_media_type_needs_no_changes_elsewhere(self, tmp_path):
        """Open/Closed check: extending support is pure addition."""

        class DocProcessor(FileProcessor):
            media_type: ClassVar[MediaType] = MediaType.DOC

            def process(self, source: ScannedFile) -> list[EvidenceNodeDraft]:
                return [EvidenceNodeDraft(node_type=NodeType.PAGE, page_number=1)]

        registry = ProcessorRegistry([ImageProcessor(max_pixels=10)])
        registry.register(DocProcessor())

        source = make_source(tmp_path / "memo.docx", MediaType.DOC)
        drafts = registry.get(MediaType.DOC).process(source)

        assert len(drafts) == 1
        assert registry.supported_types == {MediaType.IMAGE, MediaType.DOC}


class TestImageProcessor:
    def test_records_dimensions(self, tmp_path):
        from PIL import Image

        path = tmp_path / "photo.png"
        Image.new("RGB", (40, 30)).save(path)

        drafts = ImageProcessor(max_pixels=10_000).process(make_source(path, MediaType.IMAGE))

        assert len(drafts) == 1
        assert drafts[0].node_type is NodeType.IMAGE
        assert drafts[0].metadata["width"] == 40
        assert drafts[0].metadata["height"] == 30

    def test_rejects_an_image_just_over_the_pixel_budget(self, tmp_path):
        """Our own check catches the band Pillow ignores.

        Pillow only raises above *twice* MAX_IMAGE_PIXELS, so an image between
        1x and 2x the budget would otherwise sail through.
        """
        from PIL import Image

        path = tmp_path / "big.png"
        Image.new("RGB", (200, 200)).save(path)  # 40,000 px

        with pytest.raises(ResourceLimitError, match="over the"):
            ImageProcessor(max_pixels=30_000).process(make_source(path, MediaType.IMAGE))

    def test_rejects_a_decompression_bomb(self, tmp_path):
        """Far over the budget, Pillow's own guard fires first; the processor
        must still surface it as a ResourceLimitError, not a raw PIL error."""
        from PIL import Image

        path = tmp_path / "bomb.png"
        Image.new("RGB", (200, 200)).save(path)  # 400x the budget below

        with pytest.raises(ResourceLimitError, match="decompression bomb"):
            ImageProcessor(max_pixels=100).process(make_source(path, MediaType.IMAGE))


class TestPdfProcessor:
    @staticmethod
    def _make_pdf(path, pages: int, text: str = "evidence text"):
        import pymupdf

        doc = pymupdf.open()
        for _ in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), text)
        doc.save(path)
        doc.close()
        return path

    def test_extracts_text_and_renders_each_page(self, tmp_path, workspace):
        path = self._make_pdf(tmp_path / "report.pdf", pages=3)
        processor = PdfProcessor(workspace, zoom=1.0, max_pages=10,
                                 max_text_chars=1000, max_pixels=10_000_000)

        drafts = processor.process(make_source(path, MediaType.PDF))

        assert [d.page_number for d in drafts] == [1, 2, 3]
        assert all("evidence text" in d.text_content for d in drafts)
        assert all(d.file_path and d.file_path.endswith(".png") for d in drafts)

    def test_rejects_a_pdf_over_the_page_limit(self, tmp_path, workspace):
        path = self._make_pdf(tmp_path / "long.pdf", pages=5)
        processor = PdfProcessor(workspace, zoom=1.0, max_pages=2,
                                 max_text_chars=1000, max_pixels=10_000_000)

        with pytest.raises(ResourceLimitError, match="page limit"):
            processor.process(make_source(path, MediaType.PDF))

    def test_truncates_oversized_page_text(self, tmp_path, workspace):
        path = self._make_pdf(tmp_path / "wordy.pdf", pages=1, text="x" * 200)
        processor = PdfProcessor(workspace, zoom=1.0, max_pages=10,
                                 max_text_chars=10, max_pixels=10_000_000)

        draft = processor.process(make_source(path, MediaType.PDF))[0]

        assert len(draft.text_content) == 10
        assert draft.metadata["text_truncated"] is True

    def test_render_zoom_is_clamped_to_the_pixel_budget(self, tmp_path, workspace):
        path = self._make_pdf(tmp_path / "page.pdf", pages=1)
        processor = PdfProcessor(workspace, zoom=8.0, max_pages=10,
                                 max_text_chars=1000, max_pixels=10_000)

        draft = processor.process(make_source(path, MediaType.PDF))[0]

        assert draft.metadata["render_zoom"] < 8.0
        assert draft.metadata["width"] * draft.metadata["height"] <= 10_000 * 1.05


class TestSceneSegmenter:
    def test_fixed_windows_cover_the_whole_duration(self):
        segmenter = SceneSegmenter(threshold=27.0, fallback_window_sec=5.0,
                                   max_duration_sec=1000)

        segments = segmenter.fixed_windows(12.0)

        assert [(s.start, s.end) for s in segments] == [(0.0, 5.0), (5.0, 10.0), (10.0, 12.0)]
        assert [s.index for s in segments] == [0, 1, 2]

    def test_zero_duration_still_yields_one_segment(self):
        segmenter = SceneSegmenter(threshold=27.0, fallback_window_sec=5.0,
                                   max_duration_sec=1000)
        assert len(segmenter.fixed_windows(0.0)) == 1

    def test_overlong_video_is_rejected(self, tmp_path):
        segmenter = SceneSegmenter(threshold=27.0, fallback_window_sec=5.0,
                                   max_duration_sec=60)

        with pytest.raises(ResourceLimitError, match="over the"):
            segmenter.segment(tmp_path / "long.mp4", duration_sec=3600)


class TestWorkspace:
    def test_output_names_come_from_the_hash_not_the_filename(self, tmp_path, workspace):
        """Derived paths must not be influenced by attacker-chosen names."""
        hostile = tmp_path / "../../etc/passwd.mp4"
        source = ScannedFile(
            path=hostile, file_name=hostile.name, media_type=MediaType.VIDEO,
            sha256="deadbeef" * 8, size_bytes=1,
        )

        audio_path = workspace.audio_path(source)

        assert audio_path.name == "deadbeef" * 2 + ".wav"
        assert "passwd" not in str(audio_path)
        assert audio_path.parent == tmp_path / "audio"
