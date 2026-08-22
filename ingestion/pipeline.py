"""Pipeline orchestration.

The pipeline owns sequencing and error policy, nothing else. Scanning,
processing, and persistence all arrive as injected collaborators, so this class
can be tested against fakes and a new media type never requires editing it.

Error policy: one bad file must not abort a case. Anything raised by a single
file is recorded against that file and the run continues; only configuration,
connection, and schema failures are fatal.
"""
from __future__ import annotations

import logging

from .config import AppConfig
from .errors import IngestionError
from .models import FileReport, IngestionReport, ScannedFile
from .processors.base import ProcessorRegistry
from .repositories import CaseRepository, EvidenceNodeRepository, SourceFileRepository
from .scanner import CaseScanner
from .workspace import Workspace

log = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        config: AppConfig,
        scanner: CaseScanner,
        workspace: Workspace,
        registry: ProcessorRegistry,
        case_repository: CaseRepository,
        source_repository: SourceFileRepository,
        node_repository: EvidenceNodeRepository,
    ) -> None:
        self._config = config
        self._scanner = scanner
        self._workspace = workspace
        self._registry = registry
        self._cases = case_repository
        self._sources = source_repository
        self._nodes = node_repository

    def run(self) -> IngestionReport:
        case = self._config.case
        case_id = self._cases.get_or_create(case.case_number, case.title, case.description)
        self._workspace.prepare()

        scan = self._scanner.scan()
        report = IngestionReport(case_id=case_id, case_number=case.case_number)
        report.files.extend(scan.skipped)

        for source in scan.files:
            report.files.append(self._ingest_one(case_id, source))

        log.info(
            "case %s: %d ok, %d skipped, %d failed, %d evidence nodes",
            case.case_number, len(report.ok), len(report.skipped),
            len(report.failed), report.total_nodes,
        )
        return report

    def _ingest_one(self, case_id: str, source: ScannedFile) -> FileReport:
        """Register and process one file, converting any failure into a report."""
        try:
            registered = self._sources.register(case_id, source)
        except Exception as exc:  # noqa: BLE001 - one file must not kill the run
            log.exception("failed to register %s", source.file_name)
            return FileReport(source.file_name, source.media_type.value, "failed", detail=str(exc))

        if not self._registry.supports(source.media_type):
            detail = f"no processor for media type '{source.media_type.value}'"
            log.info("registered but not processed: %s (%s)", source.file_name, detail)
            return FileReport(source.file_name, source.media_type.value, "skipped", detail=detail)

        try:
            processor = self._registry.get(source.media_type)
            drafts = processor.process(source)
            count = self._nodes.replace_for_source(registered.id, drafts)
        except IngestionError as exc:
            log.warning("failed to process %s: %s", source.file_name, exc)
            return FileReport(source.file_name, source.media_type.value, "failed", detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - unexpected decoder faults
            log.exception("unexpected failure processing %s", source.file_name)
            return FileReport(source.file_name, source.media_type.value, "failed", detail=str(exc))

        return FileReport(
            file_name=source.file_name,
            media_type=source.media_type.value,
            status="ok",
            node_count=count,
            detail=self._integrity_note(source, registered),
        )

    @staticmethod
    def _integrity_note(source: ScannedFile, registered) -> str | None:
        notes = []
        if registered.content_changed:
            notes.append(
                f"content changed since last ingest (was {registered.previous_sha256[:12]})"
            )
        if source.type_mismatch:
            # An undeclared file has no config claim to name, so report what
            # its extension implied instead of an unhelpful 'None'.
            claimed_by = "config" if source.declared_type else "extension"
            claimed = source.declared_type or source.path.suffix.lstrip(".") or "unknown"
            notes.append(
                f"{claimed_by} claims '{claimed}' but content is {source.detected_mime}"
            )
        return "; ".join(notes) or None


def build_pipeline(config: AppConfig, conn) -> IngestionPipeline:
    """Composition root: assemble the pipeline from config and a connection."""
    from .processors import build_registry

    workspace = Workspace(
        audio_dir=config.resolve(config.paths.audio_dir),
        frames_dir=config.resolve(config.paths.frames_dir),
        pages_dir=config.resolve(config.paths.pages_dir),
    )
    return IngestionPipeline(
        config=config,
        scanner=CaseScanner(config),
        workspace=workspace,
        registry=build_registry(config, workspace),
        case_repository=CaseRepository(conn),
        source_repository=SourceFileRepository(conn),
        node_repository=EvidenceNodeRepository(conn),
    )


def run(config_path: str = "config.yaml") -> IngestionReport:
    """Convenience entry point: load config, connect, apply schema, ingest."""
    from .config import load_config
    from .db import apply_schema, connect

    config = load_config(config_path)
    with connect(config.database) as conn:
        apply_schema(conn)
        return build_pipeline(config, conn).run()
