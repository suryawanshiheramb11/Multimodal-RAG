"""Database connection helpers and schema bootstrap."""
from __future__ import annotations

from pathlib import Path

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from .config import Config

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def connect(cfg: Config):
    conn = psycopg2.connect(
        host=cfg.db["host"],
        port=cfg.db["port"],
        dbname=cfg.db["dbname"],
        user=cfg.db["user"],
        password=cfg.db.get("password") or None,
    )
    register_vector(conn)
    return conn


def init_schema(conn) -> None:
    sql = SCHEMA_PATH.read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def get_or_create_case(conn, cfg: Config) -> str:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            INSERT INTO "case" (case_number, title, description)
            VALUES (%s, %s, %s)
            ON CONFLICT (case_number) DO UPDATE
                SET title = EXCLUDED.title, description = EXCLUDED.description
            RETURNING id
            """,
            (cfg.case_number, cfg.case_title, cfg.case_description),
        )
        case_id = cur.fetchone()["id"]
    conn.commit()
    return case_id
