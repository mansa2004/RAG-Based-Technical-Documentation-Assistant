"""Minimal SQLite-backed feedback store. A full DB felt like overkill for a
2-day take-home, but a flat file (JSON) would make concurrent writes from
multiple requests unsafe -- SQLite gives us safe concurrent writes for free
with zero extra infrastructure."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating TEXT NOT NULL,
    comment TEXT,
    session_id TEXT,
    created_at TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    Path(config.FEEDBACK_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.FEEDBACK_DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def save_feedback(question: str, answer: str, rating: str, comment: str | None, session_id: str) -> int:
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO feedback (question, answer, rating, comment, session_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (question, answer, rating, comment, session_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()
