"""Supabase-only logger for intent router outputs."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from supabase import create_client
except Exception:  # pragma: no cover
    create_client = None

logger = logging.getLogger(__name__)

_AGENTS_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_AGENTS_ROOT / ".env", override=True)


@dataclass
class RouterLogRecord:
    session_id: str
    turn_index: Optional[int]
    user_message: str
    language: Optional[str]
    intent: str
    keywords: List[str]
    english_translation_or_summary: Optional[str]
    router_model: str
    router_base_url: Optional[str]
    router_payload: Dict[str, Any]
    created_at: str
    latency_ms: int


class RouterLogger:
    """Write router classification metadata to a dedicated Supabase table."""

    def __init__(self):
        self._supabase_table = os.getenv("SUPABASE_ROUTER_TABLE") or "agent_router_logs"
        self._supabase = self._init_supabase_client()
        self._enabled = False
        self._reason = "disabled"
        self._write_success = 0
        self._write_fail = 0
        self._warning_emitted = False
        self._last_validate_attempt = 0.0
        self._validate_target()

    def _init_supabase_client(self):
        if create_client is None:
            return None

        supabase_url = os.getenv("SUPABASE_URL", "").strip()
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        fallback_key = os.getenv("SUPABASE_KEY", "").strip()
        supabase_key = service_role_key or fallback_key

        if not supabase_url or not supabase_key:
            return None

        try:
            return create_client(supabase_url, supabase_key)
        except Exception as exc:  # pragma: no cover
            logger.warning("Router Supabase client init failed: %s", exc)
            return None

    def _validate_target(self) -> None:
        self._last_validate_attempt = time.time()
        if self._supabase is None:
            self._enabled = False
            self._reason = "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing or client unavailable"
            return

        try:
            self._supabase.table(self._supabase_table).select("session_id").limit(1).execute()
            self._enabled = True
            self._reason = f"enabled(table={self._supabase_table})"
            logger.info("Router Supabase logging enabled for table '%s'", self._supabase_table)
        except Exception as exc:  # pragma: no cover
            self._enabled = False
            self._reason = f"validation failed: {exc}"
            logger.warning(
                "Router Supabase logging disabled for table '%s': %s",
                self._supabase_table,
                exc,
            )

    def _retry_validate_if_needed(self) -> None:
        if self._enabled:
            return
        if time.time() - self._last_validate_attempt < 30:
            return
        self._validate_target()

    @staticmethod
    def sanitize_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        payload = payload or {}
        keywords = payload.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        if not isinstance(keywords, list):
            keywords = []

        return {
            "language": str(payload.get("language") or "").strip(),
            "intent": str(payload.get("intent") or "Unclear").strip(),
            "keywords": [str(keyword).strip() for keyword in keywords if str(keyword).strip()],
            "english_translation_or_summary": str(payload.get("english_translation_or_summary") or "").strip(),
        }

    def log_router_result(self, record: RouterLogRecord) -> None:
        self._retry_validate_if_needed()
        if not self._enabled:
            if not self._warning_emitted:
                logger.warning("Router Supabase logging inactive: %s", self._reason)
                self._warning_emitted = True
            return

        payload = self.sanitize_payload(record.router_payload)
        row = {
            "session_id": record.session_id,
            "turn_index": record.turn_index,
            "user_message": record.user_message,
            "language": record.language,
            "intent": record.intent,
            "keywords": record.keywords,
            "english_translation_or_summary": record.english_translation_or_summary,
            "router_model": record.router_model,
            "router_base_url": record.router_base_url,
            "router_payload": payload,
            "created_at": record.created_at,
            "latency_ms": max(record.latency_ms, 0),
        }

        try:
            self._supabase.table(self._supabase_table).insert(row).execute()
            self._write_success += 1
        except Exception as exc:  # pragma: no cover
            self._write_fail += 1
            logger.warning("Router Supabase insert failed; continuing chat response: %s", exc)

    def get_status(self) -> dict:
        self._retry_validate_if_needed()
        return {
            "enabled": self._enabled,
            "reason": self._reason,
            "table": self._supabase_table,
            "write_success": self._write_success,
            "write_fail": self._write_fail,
        }
