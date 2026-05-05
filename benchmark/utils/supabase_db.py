import json
import math
import os
from typing import Any
 
import requests
from dotenv import load_dotenv
 
load_dotenv()
 
 
def _clean(value: Any) -> Any:
    """
    Coerce a value into something Postgres / the REST API can accept.
 
      - dict / list   → passed through (will be serialised as JSONB)
      - int / float   → float, with nan/inf → None
      - None          → None  (becomes SQL NULL)
      - everything else → str
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value                        # JSONB columns
    if isinstance(value, bool):
        return value                        # bool before int check
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    return str(value)
 
 
def store_evaluation(db_row: dict) -> dict | None:
    """
    Insert one evaluation row into the Supabase `evaluations` table.
 
    Parameters
    ----------
    db_row : dict
        Must include every column you want stored.  Any key absent from the
        dict will be left as NULL in Postgres (schema default applies).
 
    Returns
    -------
    dict | None
        The inserted row returned by Supabase (includes the auto-assigned id),
        or None on failure.
    """
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = os.getenv("SUPABASE_KEY", "").strip()
 
    if not url or not key:
        print("❌ Supabase: missing SUPABASE_URL or SUPABASE_KEY env vars")
        return None
 
    endpoint = f"{url}/rest/v1/evaluations"
 
    headers = {
        "apikey":        key,
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",   # return the inserted row
    }
 
    payload = {k: _clean(v) for k, v in db_row.items()}
 
    # ── Debug: log the payload keys being sent ────────────────────────────
    non_null = [k for k, v in payload.items() if v is not None]
    null_keys = [k for k, v in payload.items() if v is None]
    print(f"[Supabase] Sending {len(non_null)} non-null fields: {non_null}")
    if null_keys:
        print(f"[Supabase] NULL fields: {null_keys}")
 
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            # Use json.dumps explicitly so we control serialisation
            data=json.dumps(payload, ensure_ascii=False),
            timeout=15,
        )
 
        if response.status_code in (200, 201):
            inserted = response.json()
            row_id = inserted[0].get("id") if isinstance(inserted, list) else "?"
            print(f"✅ Supabase insert success (id={row_id})")
            return inserted[0] if isinstance(inserted, list) else inserted
 
        # Surface the full error body so we know exactly what Postgres rejected
        print(f"❌ Supabase REST error  status={response.status_code}")
        print(f"   Response: {response.text[:800]}")
        return None
 
    except requests.exceptions.Timeout:
        print("[Supabase] Request timed out")
        return None
    except Exception as exc:
        print(f"[Supabase] Unexpected error: {exc}")
        return None