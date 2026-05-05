from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")

# Use the service role key (SUPABASE_KEY in your .env) — bypasses RLS entirely
# so SELECT works without needing explicit read policies on agent_turn_logs.
# Safe here because this runs server-side only. Never expose to frontend.
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

AGENT_TURNS_TABLE = "agent_turn_logs"
CORPUS_TABLE      = "sl_med_corpus"


def load_supabase_markdown():
    """Load all rows from the corpus table (used by /supabase/rows endpoint)."""
    res = supabase.table(CORPUS_TABLE).select("*").execute()
    return res.data or []


def load_agent_turns(is_final_answer: bool = True, limit: int = 200):
    """
    Load rows from agent_turn_logs.
    Used by the evaluation batch pipeline when listing unevaluated turns.
    """
    query = supabase.table(AGENT_TURNS_TABLE).select("*")
    if is_final_answer is not None:
        query = query.eq("is_final_answer", is_final_answer)
    res = query.order("created_at", desc=False).limit(limit).execute()
    return res.data or []