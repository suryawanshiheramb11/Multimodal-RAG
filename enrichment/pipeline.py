"""Enrichment orchestration.

Same error policy as ingestion: a node that blows up in a decoder is recorded
against that node and the run continues. Models are loaded once up front so the
availability report reflects the whole run.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .analyzers import AnalyzerRegistry, PendingNode
from .config import EnrichmentSettings
from .errors import EnrichmentError
from .registry import ModelRegistry
from .repository import EnrichmentRepository

log = logging.getLogger(__name__)


@dataclass
class NodeOutcome:
    node_id: str
    node_type: str
    status: str  # 'ok' | 'skipped' | 'failed'
    features: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    detail: str | None = None
    seconds: float = 0.0


@dataclass
class EnrichmentReport:
    case_id: str
    availability: dict[str, str] = field(default_factory=dict)
    outcomes: list[NodeOutcome] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)

    @property
    def ok(self) -> list[NodeOutcome]:
        return [o for o in self.outcomes if o.status == "ok"]

    @property
    def failed(self) -> list[NodeOutcome]:
        return [o for o in self.outcomes if o.status == "failed"]

    @property
    def skipped(self) -> list[NodeOutcome]:
        return [o for o in self.outcomes if o.status == "skipped"]


class EnrichmentPipeline:
    def __init__(
        self,
        models: ModelRegistry,
        analyzers: AnalyzerRegistry,
        repository: EnrichmentRepository,
        settings: EnrichmentSettings,
    ) -> None:
        self._models = models
        self._analyzers = analyzers
        self._repository = repository
        self._settings = settings

    def run(
        self, case_id: str, *, only_pending: bool = True, node_types: list[str] | None = None
    ) -> EnrichmentReport:
        # Load every model before touching a node: one clear availability
        # table beats the same failure repeated per node.
        availability = self._models.availability()
        for feature, status in availability.items():
            level = log.info if status == "ready" else log.warning
            level("model %-14s %s", feature, status)

        report = EnrichmentReport(case_id=case_id, availability=availability)
        self._repository.record_run(
            case_id, availability, self._settings.model_dump(mode="json")
        )

        nodes = self._repository.fetch_nodes(
            case_id, only_pending=only_pending, node_types=node_types
        )
        log.info("enriching %d node(s)", len(nodes))

        for index, node in enumerate(nodes, start=1):
            log.info("[%d/%d] %s %s", index, len(nodes), node.node_type, node.id[:8])
            report.outcomes.append(self._enrich_one(node))

        report.coverage = self._repository.coverage(case_id)
        log.info(
            "enrichment complete: %d ok, %d skipped, %d failed",
            len(report.ok), len(report.skipped), len(report.failed),
        )
        return report

    def _enrich_one(self, node: PendingNode) -> NodeOutcome:
        if not self._analyzers.supports(node.node_type):
            detail = f"no analyzer for node type '{node.node_type}'"
            log.info("skipping %s: %s", node.id[:8], detail)
            return NodeOutcome(node.id, node.node_type, "skipped", detail=detail)

        started = time.monotonic()
        try:
            result = self._analyzers.get(node.node_type).analyze(node)
            self._repository.save(node, result)
        except EnrichmentError as exc:
            log.warning("failed to enrich %s: %s", node.id[:8], exc)
            self._repository.mark_failed(node, str(exc))
            return NodeOutcome(node.id, node.node_type, "failed", detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - model faults must not kill the run
            log.exception("unexpected failure enriching %s", node.id[:8])
            self._repository.mark_failed(node, f"{type(exc).__name__}: {exc}")
            return NodeOutcome(
                node.id, node.node_type, "failed", detail=f"{type(exc).__name__}: {exc}"
            )

        return NodeOutcome(
            node_id=node.id,
            node_type=node.node_type,
            status="ok",
            features=_features_populated(result),
            skipped=result.skipped,
            seconds=round(time.monotonic() - started, 2),
        )


def _features_populated(result) -> list[str]:
    """Which outputs actually got a value, for the run report."""
    populated = []
    if result.text_content:
        populated.append("text")
    if result.text_embedding is not None:
        populated.append("text_vec")
    if result.clip_embedding is not None:
        populated.append("clip_vec")
    if result.audio_embedding is not None:
        populated.append("audio_vec")
    for key in ("transcript", "caption", "ocr", "violence", "detections", "audio_events"):
        if key in result.metadata:
            populated.append(key)
    return populated


def build_enrichment_pipeline(
    conn, settings: EnrichmentSettings | None = None
) -> EnrichmentPipeline:
    """Composition root for the enrichment phase."""
    from .analyzers import build_analyzer_registry

    settings = settings or EnrichmentSettings()
    models = ModelRegistry.build(settings)
    return EnrichmentPipeline(
        models=models,
        analyzers=build_analyzer_registry(models, settings),
        repository=EnrichmentRepository(conn),
        settings=settings,
    )
