"""PostgreSQL connection helpers."""

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(database_url: str | None) -> psycopg.Connection:
    """Open a connection with the pgvector type registered."""
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set (see .env.example)")
    conn = psycopg.connect(database_url)
    # The vector type must exist before it can be registered.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def apply_schema(conn: psycopg.Connection) -> None:
    """Create extensions and tables if they do not exist."""
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
