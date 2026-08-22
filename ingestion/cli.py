"""Command-line interface.

Typer and Rich are already dependencies (both arrive with ultralytics), so
argument parsing and table rendering are theirs rather than hand-rolled.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .config import load_config
from .db import apply_schema, connect
from .errors import IngestionError
from .models import IngestionReport
from .pipeline import build_pipeline
from .repositories import EvidenceNodeRepository, SourceFileRepository

app = typer.Typer(add_completion=False, help="Multi-modal evidence ingestion pipeline.")
console = Console()

_STATUS_STYLE = {"ok": "green", "unchanged": "cyan", "skipped": "yellow", "failed": "red"}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=verbose)],
    )
    # These libraries log a line per file at INFO; keep the run readable.
    for noisy in ("pyscenedetect", "PIL", "libav"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _render(report: IngestionReport) -> None:
    table = Table(title=f"Ingestion report — case {report.case_number}")
    table.add_column("File", overflow="fold")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Nodes", justify="right")
    table.add_column("Detail", overflow="fold")

    for entry in report.files:
        style = _STATUS_STYLE.get(entry.status, "")
        table.add_row(
            entry.file_name,
            entry.media_type,
            f"[{style}]{entry.status}[/{style}]" if style else entry.status,
            str(entry.node_count),
            entry.detail or "",
        )

    console.print(table)
    console.print(
        f"case_id={report.case_id}  files_ok={len(report.ok)}  "
        f"unchanged={len(report.unchanged)}  skipped={len(report.skipped)}  "
        f"failed={len(report.failed)}  evidence_nodes={report.total_nodes}"
    )
    # Newly extracted nodes have no embeddings yet, so say so rather than
    # leaving the operator to discover it when a graph query returns nothing.
    if report.ok:
        console.print(
            f"[yellow]note:[/yellow] {len(report.ok)} file(s) were (re)extracted — "
            "run `enrich` to populate their embeddings before `build`."
        )


@app.command()
def ingest(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    skip_schema: bool = typer.Option(
        False, "--skip-schema", help="Do not apply db/schema.sql (for least-privilege roles)."
    ),
    reprocess: bool = typer.Option(
        False,
        "--reprocess",
        help="Re-extract files even if unchanged. Discards their enrichment; re-run `enrich` after.",
    ),
) -> None:
    """Scan the case folder and ingest every evidence file.

    Files whose bytes are identical to the last ingest keep their existing
    nodes — and the enrichment on them — instead of being rebuilt.
    """
    _configure_logging(verbose)
    try:
        app_config = load_config(config)
        with connect(app_config.database) as conn:
            if not skip_schema:
                apply_schema(conn)
            report = build_pipeline(app_config, conn, reprocess).run()
    except IngestionError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _render(report)
    # A failed file is a non-zero exit so CI and shell callers notice.
    raise typer.Exit(code=1 if report.failed else 0)


@app.command()
def verify(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml."),
) -> None:
    """Report what is currently stored for the configured case."""
    _configure_logging(False)
    try:
        app_config = load_config(config)
        with connect(app_config.database) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT id FROM "case" WHERE case_number = %s',
                    (app_config.case.case_number,),
                )
                row = cur.fetchone()
            if row is None:
                console.print(
                    f"[yellow]case {app_config.case.case_number} "
                    "has not been ingested yet[/yellow]"
                )
                raise typer.Exit(code=1)

            case_id = str(row[0])
            sources = SourceFileRepository(conn).count_for_case(case_id)
            stats = EvidenceNodeRepository(conn).stats_for_case(case_id)
    except IngestionError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"case {app_config.case.case_number} ({case_id}): "
        f"{sources} source file(s), {stats.total} evidence node(s), "
        f"{stats.enriched} enriched"
    )
    pending = stats.total - stats.enriched
    if pending:
        console.print(
            f"[yellow]note:[/yellow] {pending} node(s) have no enrichment — "
            "run `enrich` before `build`, or the graph phases will find no "
            "text or embeddings to work with."
        )


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()
