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
    console.print(
        f"REFERENCES={report.references}  DESCRIBES={report.describes}  "
        f"timeline_events={report.timeline_events}  SAME_EVENT={report.same_event_links}"
    )
    console.print(
        f"claims={report.claims_extracted}  pairs_judged={report.pairs_judged}  "
        f"[red]CONTRADICTS={report.contradicts}[/red]  "
        f"[green]CORROBORATES={report.corroborates}[/green]"
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


def _format_relation(relation: dict) -> str:
    """One CONTRADICTS/CORROBORATES edge as a coloured line. Contradictions are
    red because a disagreement between two exhibits is the finding a reviewer
    most needs to see first."""
    colour = "red" if relation["relationship_type"] == "CONTRADICTS" else "green"
    confidence = (
        f" ({relation['confidence']:.2f})" if relation["confidence"] is not None else ""
    )
    return (
        f"[{colour}]{relation['relationship_type']}[/{colour}] "
        f"{relation['other_node_id'][:8]}{confidence}: {relation['explanation']}"
    )


@app.command("evidence-pack")
def evidence_pack(
    entity: str = typer.Option(None, "--entity", "-e", help="Filter to nodes mentioning this entity."),
    start: float = typer.Option(None, "--start", help="Window start, in seconds."),
    end: float = typer.Option(None, "--end", help="Window end, in seconds."),
    only_conflicts: bool = typer.Option(
        False, "--only-conflicts", help="Show only nodes carrying a CONTRADICTS/CORROBORATES edge."
    ),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Evidence for an entity and/or time window, with disagreements highlighted."""
    from ingestion.cli import _configure_logging

    _configure_logging(False)
    try:
        app_config = load_config(config)
        with connect(app_config.database) as conn:
            case_id = _resolve_case_id(conn, app_config.case.case_number)
            nodes = GraphRepository(conn).fetch_evidence_pack(case_id, entity, start, end)
    except IngestionError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if only_conflicts:
        nodes = [n for n in nodes if n["relations"]]

    if not nodes:
        console.print("[yellow]no matching evidence[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="Evidence pack")
    table.add_column("Node")
    table.add_column("Type")
    table.add_column("When / page")
    table.add_column("Claim", overflow="fold")
    table.add_column("Conflicts", overflow="fold")
    for node in nodes:
        if node["start_time"] is not None:
            when = f"{node['start_time']:.1f}-{node['end_time']:.1f}s"
        elif node["page_number"] is not None:
            when = f"p{node['page_number']}"
        else:
            when = "-"

        relations = "\n".join(_format_relation(r) for r in node["relations"])
        table.add_row(
            node["node_id"][:8], node["node_type"], when,
            node["claim"] or "[dim]none[/dim]", relations or "-",
        )
    console.print(table)


@app.command("node-edges")
def node_edges(
    node_id: str = typer.Argument(..., help="evidence_node id to inspect."),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Test helper: list every edge (of any type) touching one node."""
    from ingestion.cli import _configure_logging

    _configure_logging(False)
    try:
        app_config = load_config(config)
        with connect(app_config.database) as conn:
            case_id = _resolve_case_id(conn, app_config.case.case_number)
            edges = GraphRepository(conn).fetch_edges_for_node(case_id, node_id)
    except IngestionError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not edges:
        console.print(f"[yellow]no edges found for node {node_id}[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title=f"Edges touching {node_id}")
    table.add_column("Type")
    table.add_column("Other node / event")
    table.add_column("Score", justify="right")
    table.add_column("Metadata", overflow="fold")
    for edge in edges:
        score = f"{edge['score']:.3f}" if edge["score"] is not None else "-"
        table.add_row(edge["alignment_type"], edge["other_node_id"], score, str(edge["metadata"]))
    console.print(table)
