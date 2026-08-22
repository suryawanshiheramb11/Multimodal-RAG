"""Database connection management and schema bootstrap."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection as PgConnection

from .config import DatabaseSettings
from .errors import IngestionError

log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


@contextmanager
def connect(settings: DatabaseSettings) -> Iterator[PgConnection]:
    """Open a connection, registering pgvector's type adapters.

    Only `safe_dsn()` is ever logged — the password lives in a SecretStr and
    never reaches a log record or traceback frame.
    """
    log.debug("connecting to %s", settings.safe_dsn())
    try:
        conn = psycopg2.connect(**settings.connect_kwargs())
    except psycopg2.Error as exc:
        raise IngestionError(
            f"cannot connect to {settings.safe_dsn()}: {exc}"
        ) from exc

    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def apply_schema(conn: PgConnection, schema_path: Path = SCHEMA_PATH) -> None:
    """Apply the idempotent schema definition.

    The statements are all CREATE/ALTER ... IF NOT EXISTS, so this is safe to
    run on every startup and doubles as the migration path for an existing
    database.
    """
    if not schema_path.is_file():
        raise IngestionError(f"schema file not found: {schema_path}")

    try:
        with conn.cursor() as cur:
            cur.execute(schema_path.read_text())
        conn.commit()
    except psycopg2.Error as exc:
        conn.rollback()
        raise IngestionError(f"failed to apply schema: {exc}") from exc
    log.debug("schema applied from %s", schema_path)
