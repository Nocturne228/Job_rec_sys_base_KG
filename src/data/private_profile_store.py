"""SQLite store that persists the contest's four required personal fields encrypted."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator

from pydantic import BaseModel, ConfigDict, Field

from src.utils.crypto import decrypt_personal_info, encrypt_personal_info

PRIVATE_PROFILE_FIELDS = ("name", "phone", "email", "address")


class PrivateProfile(BaseModel):
    """Plaintext exists only at the validated API/application boundary."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    phone: str = Field(min_length=1, max_length=50)
    email: str = Field(min_length=3, max_length=320)
    address: str = Field(min_length=1, max_length=500)


class EncryptedProfileStore:
    """Encrypt four fields before SQLite and authenticate them on every read."""

    def __init__(self, path: str, master_key: str):
        if not master_key:
            raise ValueError("profile master key must not be empty")
        self.path = path
        self._master_key = master_key
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
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
                CREATE TABLE IF NOT EXISTS private_profiles (
                    user_id TEXT PRIMARY KEY,
                    name_ciphertext TEXT NOT NULL,
                    phone_ciphertext TEXT NOT NULL,
                    email_ciphertext TEXT NOT NULL,
                    address_ciphertext TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """)

    @staticmethod
    def _context(user_id: str, field: str) -> str:
        return f"private-profile:{user_id}:{field}"

    def upsert(self, user_id: str, profile: PrivateProfile) -> None:
        """Encrypt every required field before any value reaches SQLite."""
        encrypted = {
            field: encrypt_personal_info(
                str(getattr(profile, field)),
                self._master_key,
                context=self._context(user_id, field),
            )
            for field in PRIVATE_PROFILE_FIELDS
        }
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO private_profiles(
                    user_id, name_ciphertext, phone_ciphertext,
                    email_ciphertext, address_ciphertext
                )
                VALUES(?,?,?,?,?)
                ON CONFLICT(user_id) DO UPDATE SET
                    name_ciphertext=excluded.name_ciphertext,
                    phone_ciphertext=excluded.phone_ciphertext,
                    email_ciphertext=excluded.email_ciphertext,
                    address_ciphertext=excluded.address_ciphertext,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    user_id,
                    encrypted["name"],
                    encrypted["phone"],
                    encrypted["email"],
                    encrypted["address"],
                ),
            )

    def get(self, user_id: str) -> PrivateProfile | None:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT name_ciphertext, phone_ciphertext,
                       email_ciphertext, address_ciphertext
                FROM private_profiles
                WHERE user_id=?
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        values: Dict[str, str] = {}
        for field in PRIVATE_PROFILE_FIELDS:
            values[field] = decrypt_personal_info(
                row[f"{field}_ciphertext"],
                self._master_key,
                context=self._context(user_id, field),
            )
        return PrivateProfile.model_validate(values)

    def delete(self, user_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute(
                "DELETE FROM private_profiles WHERE user_id=?", (user_id,)
            )
            return cursor.rowcount > 0
