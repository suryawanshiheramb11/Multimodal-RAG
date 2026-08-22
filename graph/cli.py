"""CLI for phase 3: graph construction."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ingestion.config import load_config
from ingestion.db import apply_schema, connect
from ingestion.errors import IngestionError

from .config import GraphSettings
from .pipeline import GraphReport, build_graph_pipeline
from .repository import GraphRepository

app = typer.Typer(add_completion=False, help="Entity extraction and graph construction.")
console = Console()


def _resolve_case_id(conn, case_number: str) -> str:
    with conn.cursor() as cur:
        cur.execute('SELECT id FROM "case" WHERE case_number = %s', (case_number,))
        row = cur.fetchone()
    if row is None:
        raise IngestionError(f"case {case_number} has not been ingested yet")
    return str(row[0])


def _render(report: GraphReport) -> None:
    steps = Table(title="Graph construction steps")
    steps.add_column("Step")
    steps.add_column("Status", overflow="fold")
    for name, status in report.step_status.items():
        style = "green" if status == "ok" else ("yellow" if status == "skipped" else "red")
        steps.add_row(name, f"[{style}]{status}[/{style}]")
    console.print(steps)

    console.print(
        f"entities={report.entities}  mentions={report.mentions}  "
        f"ALIGNS_WITH={report.aligns_with}  SIMILAR_TO={report.similar_to}  "
        f"faces_detected={report.faces_detected}  face_clusters={report.face_clusters}"
    )


@app.command()
def build(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    no_faces: bool = typer.Option(False, "--no-faces", help="Skip face detection/clustering."),
) -> None:
    """Extract entities and build the evidence graph for the configured case."""
    from ingestion.cli import _configure_logging

    _configure_logging(verbose)
    try:
        app_config = load_config(config)
        settings = GraphSettings()
        if no_faces:
            settings.enable_face_detection = False
            settings.enable_face_clustering = False

        with connect(app_config.database) as conn:
            apply_schema(conn)
            case_id = _resolve_case_id(conn, app_config.case.case_number)
            report = build_graph_pipeline(conn, settings).run(case_id)
    except IngestionError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _render(report)
    failed = [s for s in report.step_status.values() if s.startswith("failed")]
    raise typer.Exit(code=1 if failed else 0)


@app.command("query-entity")
def query_entity(
    name: str = typer.Argument(..., help="Substring to search for in entity names."),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Test helper: list nodes mentioning entities matching `name`."""
    from ingestion.cli import _configure_logging

    _configure_logging(False)
    try:
        app_config = load_config(config)
        with connect(app_config.database) as conn:
            case_id = _resolve_case_id(conn, app_config.case.case_number)
            matches = GraphRepository(conn).entities_mentioning_text(case_id, name)
    except IngestionError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not matches:
        console.print(f"[yellow]no entities matching '{name}'[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title=f"Entities matching '{name}'")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Mentions", justify="right")
    table.add_column("Node IDs", overflow="fold")
    for row in matches:
        node_ids = ", ".join(str(n)[:8] for n in row["node_ids"])
        table.add_row(
            row["canonical_name"], row["entity_type"], str(row["mention_count"]), node_ids
        )
    console.print(table)
