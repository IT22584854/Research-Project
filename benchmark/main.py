from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

import utils.observability as obs
from engine import EvaluationEngine
from utils.supabase_loader import load_supabase_markdown
from supabase import create_client

load_dotenv()


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Evaluation API")

HERE = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "frontend")), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = EvaluationEngine(
    config_path="config.yaml",
    index_name=os.getenv("PINECONE_INDEX")
)

# ── Supabase client ───────────────────────────────────────────────────────────

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# ─────────────────────────────────────────────────────────────────────────────
# Timestamp helper
# Supabase stores timestamptz as text: "2026-04-17 17:27:42.386+00"
# The evaluation engine expects Unix floats.
# ─────────────────────────────────────────────────────────────────────────────

def _to_unix(ts: str) -> float:
    """
    Convert Supabase timestamptz string to Unix float.
    Handles:
      "2026-04-17 17:27:42.386+00"
      "2026-04-17T17:30:37.237000+00:00"
    """
    ts = ts.strip().replace(" ", "T")
    if ts.endswith("+00"):
        ts += ":00"
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        ts = ts.split(".")[0] + "+00:00"
        dt = datetime.fromisoformat(ts)
    return dt.astimezone(timezone.utc).timestamp()


# ─────────────────────────────────────────────────────────────────────────────
# Row to payload converter
# ─────────────────────────────────────────────────────────────────────────────

def _row_to_payload(row: dict, mode: str) -> dict:
    """
    Map one agent_turn_logs row to the dict the EvaluationEngine expects.

    Field mapping
    -------------
    rag_query (preferred) or user_message  → question
    assistant_message                       → answer
    created_at                              → start_timestamp
    responded_at                            → end_timestamp
    """
    # Prefer the intent-classifier's reformulated query over the raw user message
    # because it is cleaner and more specific for retrieval evaluation.
    question = row.get("rag_query") or row.get("user_message") or ""
    answer   = row.get("assistant_message") or ""

    if not question:
        raise ValueError(f"Row id={row.get('id')} has no question (user_message and rag_query are both empty).")
    if not answer:
        raise ValueError(f"Row id={row.get('id')} has no answer (assistant_message is empty).")

    start_ts = _to_unix(row["created_at"])
    end_ts   = _to_unix(row["responded_at"]) if row.get("responded_at") else start_ts

    return {
        "question":        question,
        "answer":          answer,
        "start_timestamp": start_ts,
        "end_timestamp":   end_ts,
        "mode":            mode,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return FileResponse(os.path.join(HERE, "frontend", "index.html"))


def load_supabase_markdown():
    try:
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )

        res = supabase.table("agent_turn_logs").select("*").execute()
        return res.data

    except Exception as e:
        print("❌ Supabase error:", str(e))
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Route 1: List all final-answer turns available for evaluation
# GET /agent-turns
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/agent-turns")
def list_agent_turns(limit: int = 50, offset: int = 0):
    """
    Return all rows from agent_turn_logs where is_final_answer = true.
    Supports pagination via limit and offset.
    These are the rows eligible for evaluation.
    """
    res = (
        supabase.table("agent_turn_logs")
        .select(
            "id, session_id, turn_index, user_message, assistant_message, "
            "active_agent, rag_query, created_at, responded_at, latency_ms"
        )
        .eq("is_final_answer", True)
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    rows = res.data or []
    return {"turns": rows, "count": len(rows), "offset": offset, "limit": limit}


# ─────────────────────────────────────────────────────────────────────────────
# Route 2: Evaluate a single turn by its DB primary key
# POST /evaluate/turn/{turn_id}?mode=high_risk_medical
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/evaluate/turn/{turn_id}")
def evaluate_turn(turn_id: int, mode: str = "high_risk_medical"):
    """
    Fetch one row from agent_turn_logs by id and evaluate it.

    Only rows with is_final_answer=true can be evaluated.

    Query params
    ------------
    mode : high_risk_medical (default) | with_ground_truth | web_search_medical

    Returns
    -------
    Full evaluation result JSON including metrics, final_score, and rating.
    """
    res = (
        supabase.table("agent_turn_logs")
        .select("*")
        .eq("id", turn_id)
        .single()
        .execute()
    )
    row = res.data
    if not row:
        raise HTTPException(status_code=404, detail=f"Turn {turn_id} not found.")

    if not row.get("is_final_answer"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Turn {turn_id} has is_final_answer=false. "
                "Only final-answer turns can be evaluated."
            ),
        )

    try:
        payload = _row_to_payload(row, mode)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = engine.evaluate(agent_response=payload, mode=mode)

    # Attach tracing metadata to the result
    result["turn_id"]    = row.get("id")
    result["session_id"] = row.get("session_id")
    result["turn_index"] = row.get("turn_index")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Route 3: Evaluate all final-answer turns in a session
# POST /evaluate/session/{session_id}?mode=high_risk_medical
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/evaluate/session/{session_id}")
def evaluate_session(session_id: str, mode: str = "high_risk_medical"):
    """
    Evaluate every is_final_answer=true turn in a given session,
    in turn_index order.

    Returns a list of per-turn evaluation results.
    Turns that fail (e.g. empty answer) return an error entry instead of crashing.
    """
    res = (
        supabase.table("agent_turn_logs")
        .select("*")
        .eq("session_id", session_id)
        .eq("is_final_answer", True)
        .order("turn_index")
        .execute()
    )
    rows = res.data or []

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No final-answer turns found for session {session_id}.",
        )

    results = []
    for row in rows:
        try:
            payload = _row_to_payload(row, mode)
            result  = engine.evaluate(agent_response=payload, mode=mode)
            result["turn_id"]    = row.get("id")
            result["session_id"] = session_id
            result["turn_index"] = row.get("turn_index")
            results.append(result)
        except Exception as e:
            results.append({
                "turn_id":    row.get("id"),
                "session_id": session_id,
                "turn_index": row.get("turn_index"),
                "status":     "error",
                "error":      str(e),
            })

    return {
        "session_id": session_id,
        "mode":       mode,
        "evaluated":  len(results),
        "results":    results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Route 4: Batch evaluate all unevaluated final-answer turns (background)
# POST /evaluate/batch?mode=high_risk_medical&limit=20
#
# Requires evaluated_at column on agent_turn_logs.
# Run this SQL first:
#   ALTER TABLE agent_turn_logs ADD COLUMN IF NOT EXISTS evaluated_at timestamptz;
# ─────────────────────────────────────────────────────────────────────────────

def _run_batch(mode: str, limit: int):
    """Background task: fetch and evaluate up to `limit` unevaluated turns."""
    res = (
        supabase.table("agent_turn_logs")
        .select("*")
        .eq("is_final_answer", True)
        .is_("evaluated_at", "null")
        .order("created_at")
        .limit(limit)
        .execute()
    )
    rows = res.data or []

    for row in rows:
        try:
            payload = _row_to_payload(row, mode)
            engine.evaluate(agent_response=payload, mode=mode)
            # Stamp evaluated_at so this row is skipped on the next batch run
            supabase.table("agent_turn_logs").update(
                {"evaluated_at": datetime.now(timezone.utc).isoformat()}
            ).eq("id", row["id"]).execute()
            print(f"[batch] Turn {row['id']} evaluated OK.")
        except Exception as e:
            print(f"[batch] Turn {row.get('id')} failed: {e}")


@app.post("/evaluate/batch")
def evaluate_batch(
    background_tasks: BackgroundTasks,
    mode: str = "high_risk_medical",
    limit: int = 20,
):
    """
    Queue a background batch evaluation of up to `limit` unevaluated
    final-answer turns.  Returns immediately with the pending count.

    Rows are stamped with evaluated_at after successful evaluation so they
    are never re-processed on the next batch run.

    Prerequisite SQL (run once in Supabase SQL editor):
        ALTER TABLE agent_turn_logs
          ADD COLUMN IF NOT EXISTS evaluated_at timestamptz;
    """
    try:
        pending_res = (
            supabase.table("agent_turn_logs")
            .select("id", count="exact")
            .eq("is_final_answer", True)
            .is_("evaluated_at", "null")
            .execute()
        )
        pending = pending_res.count or 0
    except Exception:
        pending = -1  # column may not exist yet — batch will still run

    background_tasks.add_task(_run_batch, mode, limit)

    return {
        "status":  "queued",
        "mode":    mode,
        "pending": pending,
        "queued":  min(limit, pending) if pending >= 0 else limit,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Route 5: Original manual evaluation endpoints (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class AgentResponse(BaseModel):
    question:        str
    answer:          str
    start_timestamp: float
    end_timestamp:   float
    mode:            str = "with_ground_truth"


@app.post("/evaluate/ground-truth")
def evaluate_ground_truth(response: AgentResponse):
    result = engine.evaluate(
        agent_response=response.dict(),
        mode="with_ground_truth"
    )
    return result


@app.post("/evaluate/high-risk")
def evaluate_high_risk(response: AgentResponse):
    result = engine.evaluate(
        agent_response=response.dict(),
        mode="high_risk_medical"
    )
    return result