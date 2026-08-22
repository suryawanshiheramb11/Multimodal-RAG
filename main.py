"""Unified entry point for every phase of the pipeline.

Each phase owns its own Typer app; this module flattens their commands into one
`case` CLI so the operator has a single tool:

    python main.py ingest         # phase 1: register and extract raw evidence
    python main.py enrich         # phase 2: run ML feature extraction
    python main.py models         # check which enrichment models can load
    python main.py build          # phase 3: entities, mentions, alignment, faces
    python main.py query-entity   # phase 3: nodes mentioning a given entity
    python main.py verify         # what is currently stored

`python -m ingestion ...` still works for the phase 1 commands alone.
"""
from __future__ import annotations

import sys

import typer

from enrichment.cli import app as enrichment_app
from graph.cli import app as graph_app
from ingestion.cli import app as ingestion_app

app = typer.Typer(
    add_completion=False,
    help="Multi-modal forensic evidence pipeline.",
    no_args_is_help=True,
)

# Flatten every phase into one command surface rather than nesting them behind
# sub-app names: the operator thinks in verbs, not in package boundaries.
app.registered_commands.extend(ingestion_app.registered_commands)
app.registered_commands.extend(enrichment_app.registered_commands)
app.registered_commands.extend(graph_app.registered_commands)


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()
