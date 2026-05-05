"""FastAPI HTTP server exposing the LangGraph multi-agent system via REST API."""
from __future__ import annotations

import os
import sys
import types
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path setup ───────────────────────────────────────────────────────────────
# Railway deploys only the agents/ folder to /app, so `import agents` fails
# because the package can't find itself. Fix: register a virtual 'agents'
# package pointing to this directory before any agents.src.* imports occur.
# This also fixes sub-modules (system.py, config.py, etc.) that use the same
# `from agents.src.*` pattern.
_HERE = Path(__file__).resolve().parent
if "agents" not in sys.modules:
    _agents_pkg = types.ModuleType("agents")
    _agents_pkg.__path__ = [str(_HERE)]
    _agents_pkg.__package__ = "agents"
    sys.modules["agents"] = _agents_pkg

from langchain_core.messages import AIMessage, HumanMessage

from backend.agent_service import extract_sources
from backend.router_logger import RouterLogger, RouterLogRecord
from backend.turn_logger import TurnLogRecord, TurnLogger, utc_iso_now
from agents.src.utils import sanitize_input, setup_logger

logger = setup_logger("api_server")
_LOG_DB_PATH = Path(os.getenv("EVAL_LOG_DB_PATH", str(_HERE / "data" / "evaluation_logs.db")))
turn_logger = TurnLogger(_LOG_DB_PATH)
router_logger = RouterLogger()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Nenagov API", version="1.0.0")

# CORS — allow requests from the deployed frontend (or * if not set)
_frontend_url = os.getenv("FRONTEND_URL", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_frontend_url] if _frontend_url != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    sources: List[dict]
    session_id: str


@lru_cache(maxsize=1)
def get_agent_graph():
    """Load the LangGraph workflow lazily so health checks can succeed even if RAG startup is slow."""
    system_module = import_module("agents.src.agent.system")
    return system_module.agent_supervisor_graph


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "backend",
        "turn_log_sync": turn_logger.get_sync_status(),
        "router_log_sync": router_logger.get_status(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    created_at = utc_iso_now()
    t_start = datetime.now(timezone.utc)

    safe_message = sanitize_input(req.message)
    if not safe_message:
        raise HTTPException(status_code=400, detail="Message is empty or invalid.")

    logger.info(
        "Received chat request | session_id=%s | preview=%s",
        session_id,
        safe_message[:120].replace("\n", " "),
    )

    state = {
        "messages": [HumanMessage(content=safe_message)],
        "session_id": session_id,
    }
    config = {"configurable": {"thread_id": session_id}}

    try:
        agent_supervisor_graph = get_agent_graph()
        result = await agent_supervisor_graph.ainvoke(state, config=config)
    except Exception as e:
        logger.exception(f"Graph invocation error for session {session_id}: {e}")
        raise HTTPException(
            status_code=503,
            detail="Agent backend is unavailable or still initializing. Please try again.",
        )

    responded_at = utc_iso_now()
    latency_ms = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)

    # Extract the last AI message as the response text
    messages = result.get("messages", [])
    ai_reply = next(
        (m.content for m in reversed(messages) if isinstance(m, AIMessage)),
        "I'm sorry, I couldn't generate a response. Please try again.",
    )

    sources = extract_sources(messages)

    turn_type = result.get("turn_type")
    is_clarification = turn_type == "clarification"
    is_final_answer = turn_type == "final"
    active_agent = result.get("last_active_agent") or result.get("active_agent")
    rag_query = result.get("last_rag_query")

    try:
        turn_index = turn_logger.log_turn(
            TurnLogRecord(
                session_id=session_id,
                user_message=safe_message,
                assistant_message=ai_reply,
                active_agent=active_agent,
                rag_query=rag_query,
                is_clarification=is_clarification,
                is_final_answer=is_final_answer,
                created_at=created_at,
                responded_at=responded_at,
                latency_ms=max(latency_ms, 0),
            )
        )
        logger.info(
            "Stored turn log | session_id=%s | turn_index=%s | type=%s",
            session_id,
            turn_index,
            turn_type or "unknown",
        )

        router_payload = result.get("router_payload") or {}
        router_intent = result.get("route_intent") or router_payload.get("intent")
        if router_intent:
            router_log_stored = router_logger.log_router_result(
                RouterLogRecord(
                    session_id=session_id,
                    turn_index=turn_index,
                    user_message=safe_message,
                    language=result.get("route_language") or router_payload.get("language"),
                    intent=router_intent,
                    keywords=result.get("route_keywords") or router_payload.get("keywords") or [],
                    english_translation_or_summary=(
                        result.get("route_summary")
                        or router_payload.get("english_translation_or_summary")
                    ),
                    router_model=result.get("router_model") or "",
                    router_base_url=result.get("router_base_url"),
                    router_payload=router_payload,
                    created_at=created_at,
                    latency_ms=int(result.get("router_latency_ms") or 0),
                )
            )
            if router_log_stored:
                logger.info(
                    "Stored router log | session_id=%s | turn_index=%s | intent=%s",
                    session_id,
                    turn_index,
                    router_intent,
                )
            else:
                logger.warning(
                    "Router log not stored | session_id=%s | turn_index=%s | intent=%s | status=%s",
                    session_id,
                    turn_index,
                    router_intent,
                    router_logger.get_status(),
                )
        else:
            logger.warning(
                "Router log skipped because router metadata is missing | session_id=%s | turn_index=%s | keys=%s",
                session_id,
                turn_index,
                sorted(result.keys()),
            )
    except Exception as exc:
        logger.exception("Turn log persistence failed | session_id=%s | error=%s", session_id, exc)

    return ChatResponse(response=ai_reply, sources=sources, session_id=session_id)


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    # MemorySaver is in-memory; sessions naturally expire with the process.
    return {"status": "deleted", "session_id": session_id}
