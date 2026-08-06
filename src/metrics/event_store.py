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
            # Serialize the one-time migration when several workers start against
            # the same pre-existing demo database.
            db.execute("BEGIN IMMEDIATE")
            db.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    impression_id INTEGER,
                    payload TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(events)").fetchall()
            }
            if "impression_id" not in columns:
                db.execute("ALTER TABLE events ADD COLUMN impression_id INTEGER")
            db.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS one_feedback_per_impression
                ON events(impression_id)
                WHERE event_type='feedback'
                """)

    def record_impression(
        self,
        user_id: str,
        job_id: str,
        model_version: str,
        payload: Dict[str, Any] | None = None,
    ) -> int:
        with self._connect() as db:
            cursor = db.execute(
                """
                INSERT INTO events(event_type,user_id,job_id,model_version,payload)
                VALUES('impression',?,?,?,?)
                """,
                (
                    user_id,
                    job_id,
                    model_version,
                    json.dumps(payload or {}, sort_keys=True),
                ),
            )
            return int(cursor.lastrowid)

    def record_feedback(
        self,
        impression_id: int,
        user_id: str,
        job_id: str,
        model_version: str,
        satisfied: bool | None,
        *,
        clicked: bool = False,
        dwell_seconds: float = 0.0,
        saved: bool = False,
        applied: bool = False,
    ) -> None:
        """Record one feedback event for the exact matching impression."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            impression = db.execute(
                """
                SELECT 1 FROM events
                WHERE event_type='impression'
                  AND id=? AND user_id=? AND job_id=? AND model_version=?
                LIMIT 1
                """,
                (impression_id, user_id, job_id, model_version),
            ).fetchone()
            if impression is None:
                raise ValueError("feedback requires the exact matching impression")
            duplicate = db.execute(
                """
                SELECT 1 FROM events
                WHERE event_type='feedback' AND impression_id=?
                LIMIT 1
                """,
                (impression_id,),
            ).fetchone()
            if duplicate is not None:
                raise ValueError("feedback already recorded for impression")
            db.execute(
                """
                INSERT INTO events(
                    event_type,user_id,job_id,model_version,impression_id,payload
                )
                VALUES('feedback',?,?,?,?,?)
                """,
                (
                    user_id,
                    job_id,
                    model_version,
                    impression_id,
                    json.dumps(
                        {
                            "satisfied": satisfied,
                            "clicked": clicked,
                            "dwell_seconds": dwell_seconds,
                            "saved": saved,
                            "applied": applied,
                        },
                        sort_keys=True,
                    ),
                ),
            )

    def effectiveness(self) -> Dict[str, Any]:
        with self._connect() as db:
            rows = db.execute("""
                SELECT user_id, payload FROM events
                WHERE event_type='feedback' AND impression_id IS NOT NULL
                """).fetchall()
        totals: Dict[str, int] = {}
        satisfied: Dict[str, int] = {}
        for row in rows:
            uid = row["user_id"]
            payload = json.loads(row["payload"])
            if payload.get("satisfied") is None:
                continue
            value = bool(payload["satisfied"])
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
