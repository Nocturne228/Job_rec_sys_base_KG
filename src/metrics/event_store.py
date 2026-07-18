"""SQLite-backed exposure and feedback store for multi-worker-safe demo state."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List


class EventStore:
    def __init__(self, path: str = "data/jobrec_events.sqlite3"):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    payload TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def record(
        self,
        event_type: str,
        user_id: str,
        job_id: str,
        model_version: str,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO events(event_type,user_id,job_id,model_version,payload) VALUES(?,?,?,?,?)",
                (
                    event_type,
                    user_id,
                    job_id,
                    model_version,
                    json.dumps(payload or {}, sort_keys=True),
                ),
            )

    def record_feedback(
        self,
        user_id: str,
        job_id: str,
        model_version: str,
        satisfied: bool,
    ) -> None:
        """Record feedback only when the same model exposed the job first."""
        with self._connect() as db:
            impression = db.execute(
                """
                SELECT 1 FROM events
                WHERE event_type='impression'
                  AND user_id=? AND job_id=? AND model_version=?
                LIMIT 1
                """,
                (user_id, job_id, model_version),
            ).fetchone()
            if impression is None:
                raise ValueError("feedback requires a matching impression")
            db.execute(
                """
                INSERT INTO events(event_type,user_id,job_id,model_version,payload)
                VALUES('feedback',?,?,?,?)
                """,
                (
                    user_id,
                    job_id,
                    model_version,
                    json.dumps({"satisfied": satisfied}, sort_keys=True),
                ),
            )

    def effectiveness(self) -> Dict[str, Any]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT user_id, payload FROM events WHERE event_type='feedback'"
            ).fetchall()
        totals: Dict[str, int] = {}
        satisfied: Dict[str, int] = {}
        for row in rows:
            uid = row["user_id"]
            value = bool(json.loads(row["payload"]).get("satisfied"))
            totals[uid] = totals.get(uid, 0) + 1
            satisfied[uid] = satisfied.get(uid, 0) + int(value)
        n_total = sum(totals.values())
        n_satisfied = sum(satisfied.values())
        return {
            "n_total": n_total,
            "n_satisfied": n_satisfied,
            "effectiveness": n_satisfied / max(n_total, 1),
            "by_user": {
                uid: satisfied.get(uid, 0) / total for uid, total in totals.items()
            },
        }

    def list_events(self, limit: int = 100) -> List[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]
