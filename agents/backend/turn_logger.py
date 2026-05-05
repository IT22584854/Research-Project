"""SQLite logger for per-turn agent interaction records used in evaluation."""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

from dotenv import load_dotenv

try:
    from supabase import create_client
except Exception:  # pragma: no cover - optional dependency at runtime
    create_client = None


logger = logging.getLogger(__name__)

_AGENTS_ROOT = Path(__file__).resolve().parents[1]
# Load envs here because TurnLogger is initialized before other modules that load .env.
load_dotenv(_AGENTS_ROOT / ".env", override=True)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TurnLogRecord:
    session_id: str
    user_message: str
    assistant_message: str
    active_agent: Optional[str]
    rag_query: Optional[str]
    is_clarification: bool
    is_final_answer: bool
    created_at: str
    responded_at: str
    latency_ms: int


class TurnLogger:
    """Persist interaction turns with deterministic per-session turn ordering."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        # Keep corpus table and eval table separate.
        self._supabase_table = os.getenv("SUPABASE_EVAL_TABLE") or "agent_turn_logs"
        self._supabase_validate_write = _env_flag("SUPABASE_SYNC_VALIDATE_WRITE", default=True)
        self._supabase = self._init_supabase_client()
        self._supabase_enabled = False
        self._supabase_status_reason = "disabled"
        self._supabase_sync_success = 0
        self._supabase_sync_fail = 0
        self._supabase_warning_emitted = False
        self._supabase_conflict_warning_emitted = False
        self._validate_supabase_target()
        self._ensure_schema()

    def _init_supabase_client(self):
        """Initialize Supabase client only when required env vars are available."""
        if create_client is None:
            return None

        supabase_url = os.getenv("SUPABASE_URL", "").strip()
        # Preferred: service-role key. Fallback to SUPABASE_KEY for compatibility.
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        fallback_key = os.getenv("SUPABASE_KEY", "").strip()
        supabase_key = service_role_key or fallback_key

        if not supabase_url or not supabase_key:
            return None

        if not service_role_key and fallback_key:
            logger.warning(
                "Using SUPABASE_KEY fallback for eval sync. "
                "Set SUPABASE_SERVICE_ROLE_KEY to avoid RLS write failures."
            )

        try:
            return create_client(supabase_url, supabase_key)
        except Exception as exc:  # pragma: no cover - network/env dependent
            logger.warning("Supabase client init failed; continuing with SQLite only: %s", exc)
            return None

    def _validate_supabase_target(self) -> None:
        """Validate Supabase table availability and permissions at startup."""
        if self._supabase is None:
            self._supabase_enabled = False
            self._supabase_status_reason = (
                "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing or client unavailable"
            )
            return

        try:
            self._supabase.table(self._supabase_table).select("session_id").limit(1).execute()
            self._supabase_enabled = True
            self._supabase_status_reason = f"enabled(table={self._supabase_table})"
            logger.info("Supabase sync enabled for table '%s'", self._supabase_table)
        except Exception as exc:  # pragma: no cover - network/env dependent
            self._supabase_enabled = False
            self._supabase_status_reason = f"validation failed: {exc}"
            logger.warning(
                "Supabase sync disabled after validation failure for table '%s': %s",
                self._supabase_table,
                exc,
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_turn_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    user_message TEXT NOT NULL,
                    assistant_message TEXT NOT NULL,
                    active_agent TEXT,
                    rag_query TEXT NULL,
                    is_clarification INTEGER NOT NULL CHECK (is_clarification IN (0, 1)),
                    is_final_answer INTEGER NOT NULL CHECK (is_final_answer IN (0, 1)),
                    created_at TEXT NOT NULL,
                    responded_at TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL CHECK (latency_ms >= 0),
                    CHECK (responded_at >= created_at),
                    CHECK (NOT (is_clarification = 1 AND is_final_answer = 1)),
                    UNIQUE(session_id, turn_index)
                )
                """
            )
            conn.commit()

    def log_turn(self, record: TurnLogRecord) -> int:
        """Insert one turn and return its session-local turn_index."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                cursor = conn.execute(
                    "SELECT COALESCE(MAX(turn_index), 0) + 1 FROM agent_turn_logs WHERE session_id = ?",
                    (record.session_id,),
                )
                turn_index = int(cursor.fetchone()[0])
                conn.execute(
                    """
                    INSERT INTO agent_turn_logs (
                        session_id,
                        turn_index,
                        user_message,
                        assistant_message,
                        active_agent,
                        rag_query,
                        is_clarification,
                        is_final_answer,
                        created_at,
                        responded_at,
                        latency_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.session_id,
                        turn_index,
                        record.user_message,
                        record.assistant_message,
                        record.active_agent,
                        record.rag_query,
                        int(record.is_clarification),
                        int(record.is_final_answer),
                        record.created_at,
                        record.responded_at,
                        record.latency_ms,
                    ),
                )
                conn.commit()

            self._sync_to_supabase(record, turn_index)
            return turn_index

    def _sync_to_supabase(self, record: TurnLogRecord, turn_index: int) -> None:
        """Upsert the persisted turn to Supabase when configured."""
        if not self._supabase_enabled:
            if not self._supabase_warning_emitted:
                logger.warning("Supabase sync inactive: %s", self._supabase_status_reason)
                self._supabase_warning_emitted = True
            return

        payload = {
            "session_id": record.session_id,
            "turn_index": turn_index,
            "user_message": record.user_message,
            "assistant_message": record.assistant_message,
            "active_agent": record.active_agent,
            "rag_query": record.rag_query,
            "is_clarification": bool(record.is_clarification),
            "is_final_answer": bool(record.is_final_answer),
            "created_at": record.created_at,
            "responded_at": record.responded_at,
            "latency_ms": record.latency_ms,
        }

        try:
            self._supabase.table(self._supabase_table).upsert(
                payload,
                on_conflict="session_id,turn_index",
            ).execute()
            if self._supabase_validate_write:
                verify = (
                    self._supabase.table(self._supabase_table)
                    .select("session_id")
                    .eq("session_id", record.session_id)
                    .eq("turn_index", turn_index)
                    .limit(1)
                    .execute()
                )
                if not getattr(verify, "data", None):
                    raise RuntimeError(
                        "Supabase row verification failed after upsert "
                        f"(session_id={record.session_id}, turn_index={turn_index})"
                    )

            self._supabase_sync_success += 1
        except Exception as exc:  # pragma: no cover - network/env dependent
            error_text = str(exc)
            missing_conflict_constraint = (
                "42P10" in error_text
                or "no unique or exclusion constraint matching the ON CONFLICT specification" in error_text
            )

            # Compatibility fallback for tables without UNIQUE(session_id, turn_index).
            if missing_conflict_constraint:
                if not self._supabase_conflict_warning_emitted:
                    logger.warning(
                        "Supabase table '%s' is missing UNIQUE(session_id, turn_index); "
                        "falling back to insert-only sync until schema is fixed.",
                        self._supabase_table,
                    )
                    self._supabase_conflict_warning_emitted = True

                try:
                    self._supabase.table(self._supabase_table).insert(payload).execute()
                    self._supabase_sync_success += 1
                    return
                except Exception as insert_exc:  # pragma: no cover - network/env dependent
                    self._supabase_sync_fail += 1
                    logger.warning(
                        "Supabase insert fallback failed; SQLite write already committed: %s",
                        insert_exc,
                    )
                    return

            self._supabase_sync_fail += 1
            logger.warning("Supabase upsert failed; SQLite write already committed: %s", exc)

    def get_sync_status(self) -> dict:
        """Return current sync health for diagnostics and health endpoints."""
        return {
            "enabled": self._supabase_enabled,
            "reason": self._supabase_status_reason,
            "table": self._supabase_table,
            "validate_write": self._supabase_validate_write,
            "sync_success": self._supabase_sync_success,
            "sync_fail": self._supabase_sync_fail,
        }


def utc_iso_now() -> str:
    """Return current UTC timestamp in ISO-8601 Z format."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
