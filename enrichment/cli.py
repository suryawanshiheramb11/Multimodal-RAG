"""CLI for the feature-extraction phase."""
from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ingestion.config import load_config
from ingestion.db import apply_schema, connect
from ingestion.errors import IngestionError

from .config import EnrichmentSettings
from .pipeline import EnrichmentReport, build_enrichment_pipeline

app = typer.Typer(add_completion=False, help="Feature extraction over ingested evidence.")
console = Console()

_STATUS_STYLE = {"ok": "green", "skipped": "yellow", "failed": "red"}


def _render(report: EnrichmentReport) -> None:
    models = Table(title="Model availability")
    models.add_column("Feature")
    models.add_column("Status", overflow="fold")
    for feature, status in report.availability.items():
        style = "green" if status == "ready" else "yellow"
        models.add_row(feature, f"[{style}]{status}[/{style}]")
    console.print(models)

    nodes = Table(title="Enriched nodes")
    nodes.add_column("Node")
    nodes.add_column("Type")
    nodes.add_column("Status")
    nodes.add_column("Populated", overflow="fold")
    nodes.add_column("Sec", justify="right")
    for outcome in report.outcomes:
        style = _STATUS_STYLE.get(outcome.status, "")
        nodes.add_row(
            outcome.node_id[:8],
            outcome.node_type,
            f"[{style}]{outcome.status}[/{style}]" if style else outcome.status,
            ", ".join(outcome.features) or (outcome.detail or ""),
            f"{outcome.seconds:g}",
        )
    console.print(nodes)

    coverage = Table(title="Coverage by node type")
    headers = ("Type", "Total", "Enriched", "Text", "Text vec",
               "CLIP vec", "Audio vec", "Failed")
    for column in headers:
        coverage.add_column(column, justify="right" if column != "Type" else "left")
    for row in report.coverage.get("by_type", []):
        coverage.add_row(
            row["node_type"], str(row["total"]), str(row["enriched"]),
            str(row["with_text"]), str(row["with_text_vec"]),
            str(row["with_clip_vec"]), str(row["with_audio_vec"]), str(row["failed"]),
        )
    console.print(coverage)


def _resolve_case_id(conn, case_number: str) -> str:
    with conn.cursor() as cur:
        cur.execute('SELECT id FROM "case" WHERE case_number = %s', (case_number,))
        row = cur.fetchone()
    if row is None:
        raise IngestionError(f"case {case_number} has not been ingested yet")
    return str(row[0])


@app.command()
def enrich(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    force: bool = typer.Option(
        False, "--force", help="Re-enrich nodes that already have results."
    ),
    node_type: list[str] = typer.Option(
        None, "--node-type", help="Restrict to these node types (repeatable)."
    ),
    no_caption: bool = typer.Option(False, "--no-caption", help="Skip the vision-language model."),
    no_asr: bool = typer.Option(False, "--no-asr", help="Skip speech recognition."),
    asr_model: str = typer.Option(
        None, "--asr-model", help="Override the Whisper checkpoint (e.g. 'base')."
    ),
) -> None:
    """Run ML feature extraction over the case's evidence nodes."""
    from ingestion.cli import _configure_logging

    _configure_logging(verbose)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)

    try:
        app_config = load_config(config)
        settings = EnrichmentSettings()
        if no_caption:
            settings.enable_caption = False
        if no_asr:
            settings.enable_asr = False
        if asr_model:
            settings.models.asr = asr_model

        with connect(app_config.database) as conn:
            apply_schema(conn)
            case_id = _resolve_case_id(conn, app_config.case.case_number)
            pipeline = build_enrichment_pipeline(conn, settings)
            report = pipeline.run(
                case_id, only_pending=not force, node_types=list(node_type) if node_type else None
            )
    except IngestionError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _render(report)
    raise typer.Exit(code=1 if report.failed else 0)


@app.command()
def models(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Check which ML models can be loaded, without touching the database."""
    from ingestion.cli import _configure_logging

    from .registry import ModelRegistry

    _configure_logging(False)
    load_config(config)  # validates config before any slow model load
    availability = ModelRegistry.build(EnrichmentSettings()).availability()

    table = Table(title="Model availability")
    table.add_column("Feature")
    table.add_column("Status", overflow="fold")
    for feature, status in availability.items():
        style = "green" if status == "ready" else "yellow"
        table.add_row(feature, f"[{style}]{status}[/{style}]")
    console.print(table)

    missing = [f for f, s in availability.items() if s != "ready"]
    raise typer.Exit(code=1 if missing else 0)
