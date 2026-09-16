"""Postgres connection helper shared by the API and the worker loop.

Local dev / docker-compose: DATABASE_URL defaults to the compose Postgres
service reachable from another container as `postgres`; override with an
env var when running scripts against a port-forwarded localhost instead.
"""
import contextlib
import os

import psycopg2

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://forecast:forecast@localhost:5432/forecast"
)


@contextlib.contextmanager
def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()
