"""Pipeline orchestration tests.

These run with fake repositories and no Postgres, which is the payoff of
injecting collaborators rather than constructing them inside the pipeline.
"""
from __future__ import annotations

from typing import ClassVar

import pytest

from ingestion.errors import MediaProcessingError
from ingestion.models import EvidenceNodeDraft, MediaType, NodeType, ScannedFile
from ingestion.pipeline import IngestionPipeline
from ingestion.processors.base import FileProcessor, ProcessorRegistry
from ingestion.repositories import RegisteredSource
from ingestion.scanner import CaseScanner
from ingestion.workspace import Workspace


class FakeCaseRepository:
    def get_or_create(self, case_number, title, description):
        return "case-uuid"


class FakeSourceRepository:
    def __init__(self, content_changed: bool = False) -> None:
        self.registered: list[ScannedFile] = []
        self._content_changed = content_changed

    def register(self, case_id, source):
        self.registered.append(source)
        return RegisteredSource(
            id=f"src-{len(self.registered)}",
            is_new=True,
            content_changed=self._content_changed,
            previous_sha256="0" * 64 if self._content_changed else None,
        )


class FakeNodeRepository:
    def __init__(self) -> None:
        self.writes: dict[str, list[EvidenceNodeDraft]] = {}

    def replace_for_source(self, source_file_id, drafts):
        self.writes[source_file_id] = drafts
        return len(drafts)


class StubProcessor(FileProcessor):
    media_type: ClassVar[MediaType] = MediaType.PDF

    def __init__(self, drafts=None, error: Exception | None = None) -> None:
        self._drafts = drafts if drafts is not None else [
            EvidenceNodeDraft(node_type=NodeType.PAGE, page_number=1)
        ]
        self._error = error

    def process(self, source):
        if self._error:
            raise self._error
        return self._drafts


def build(config, registry, source_repo=None, node_repo=None):
    workspace = Workspace(
        config.resolve(config.paths.audio_dir),
        config.resolve(config.paths.frames_dir),
        config.resolve(config.paths.pages_dir),
    )
    return IngestionPipeline(
        config=config,
        scanner=CaseScanner(config),
        workspace=workspace,
        registry=registry,
        case_repository=FakeCaseRepository(),
        source_repository=source_repo or FakeSourceRepository(),
        node_repository=node_repo or FakeNodeRepository(),
    )


def write_pdf(case_folder, name="doc.pdf"):
    path = case_folder / name
    path.write_bytes(b"%PDF-1.4\n%\xc7\xec\x8f\xa2\ntrailer\n")
    return path


class TestHappyPath:
    def test_processes_and_persists(self, config, case_folder):
        write_pdf(case_folder)
        nodes = FakeNodeRepository()
        pipeline = build(config, ProcessorRegistry([StubProcessor()]), node_repo=nodes)

        report = pipeline.run()

        assert len(report.ok) == 1
        assert report.total_nodes == 1
        assert nodes.writes == {"src-1": [EvidenceNodeDraft(node_type=NodeType.PAGE, page_number=1)]}


class TestErrorIsolation:
    def test_a_failing_file_does_not_abort_the_run(self, config, case_folder):
        """The whole point of the per-file try: one corrupt exhibit must not
        cost you the other twenty."""
        write_pdf(case_folder, "good.pdf")
        write_pdf(case_folder, "bad.pdf")

        class SelectiveProcessor(StubProcessor):
            def process(self, source):
                if source.file_name == "bad.pdf":
                    raise MediaProcessingError("corrupt xref table")
                return [EvidenceNodeDraft(node_type=NodeType.PAGE, page_number=1)]

        report = build(config, ProcessorRegistry([SelectiveProcessor()])).run()

        assert {f.file_name for f in report.ok} == {"good.pdf"}
        assert {f.file_name for f in report.failed} == {"bad.pdf"}
        assert "corrupt xref table" in report.failed[0].detail
        assert report.total_nodes == 1

    def test_unexpected_errors_are_contained_too(self, config, case_folder):
        """Decoders raise things that are not IngestionError; a segfault-
        adjacent ValueError from a C extension must not kill the case."""
        write_pdf(case_folder)
        registry = ProcessorRegistry([StubProcessor(error=ValueError("libav exploded"))])

        report = build(config, registry).run()

        assert len(report.failed) == 1
        assert "libav exploded" in report.failed[0].detail

    def test_unsupported_type_is_registered_but_skipped(self, config, case_folder):
        """A .docx is still evidence: hash it and record it even with no
        processor, so it is not silently absent from the case."""
        write_pdf(case_folder)
        sources = FakeSourceRepository()

        report = build(config, ProcessorRegistry([]), source_repo=sources).run()

        assert len(report.skipped) == 1
        assert "no processor" in report.skipped[0].detail
        assert len(sources.registered) == 1  # registered despite not being processed


class TestIntegrityReporting:
    def test_changed_content_is_surfaced(self, config, case_folder):
        write_pdf(case_folder)
        sources = FakeSourceRepository(content_changed=True)

        report = build(config, ProcessorRegistry([StubProcessor()]), source_repo=sources).run()

        assert "content changed since last ingest" in report.ok[0].detail

    def test_type_mismatch_is_surfaced(self, config, case_folder):
        """A PDF named .mp4 is ingested as a PDF, and the lie is reported."""
        spoofed = case_folder / "clip.mp4"
        spoofed.write_bytes(b"%PDF-1.4\n%\xc7\xec\x8f\xa2\ntrailer\n")

        report = build(config, ProcessorRegistry([StubProcessor()])).run()

        assert len(report.ok) == 1
        assert "content is application/pdf" in report.ok[0].detail


class TestScanFailuresPropagate:
    def test_skipped_scan_entries_appear_in_the_report(self, config, case_folder, tmp_path):
        outside = tmp_path / "elsewhere.pdf"
        outside.write_bytes(b"%PDF-1.4\n")
        (case_folder / "link.pdf").symlink_to(outside)

        report = build(config, ProcessorRegistry([StubProcessor()])).run()

        assert len(report.skipped) == 1
        assert "escapes the case folder" in report.skipped[0].detail

    def test_missing_case_folder_is_fatal(self, config):
        """Config-level problems should stop the run, unlike per-file ones."""
        config.paths.case_folder = config.root / "does-not-exist"
        with pytest.raises(Exception, match="case folder not found"):
            build(config, ProcessorRegistry([StubProcessor()])).run()
