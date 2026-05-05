import csv
import io
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

try:
    from supabase import create_client
except ImportError:
    create_client = None

router = APIRouter()

def get_supabase_client():
    if not create_client:
        raise HTTPException(status_code=500, detail="Supabase client not installed")

    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    fallback_key = os.getenv("SUPABASE_KEY", "").strip()
    supabase_key = service_role_key or fallback_key

    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Supabase credentials missing")

    return create_client(supabase_url, supabase_key)

def get_table_name():
    return os.getenv("SUPABASE_ROUTER_TABLE") or "agent_router_logs"

@router.get("/router-logs")
async def get_router_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    search: Optional[str] = None,
    intent: Optional[str] = None,
    language: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    client = get_supabase_client()
    table = get_table_name()

    query = client.table(table).select("*", count="exact")

    if search:
        query = query.ilike("user_message", f"%{search}%")
    if intent and intent != "All Intents":
        query = query.eq("intent", intent)
    if language and language != "All Languages":
        query = query.eq("language", language)
    if date_from:
        query = query.gte("created_at", date_from)
    if date_to:
        query = query.lte("created_at", date_to)

    # Order and Pagination
    query = query.order("created_at", desc=True)
    start = (page - 1) * page_size
    end = start + page_size - 1
    query = query.range(start, end)

    try:
        response = query.execute()
        return {
            "data": response.data,
            "total_count": response.count,
            "page": page,
            "page_size": page_size,
            "total_pages": (response.count + page_size - 1) // page_size if response.count else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/router-logs/stats")
async def get_router_logs_stats():
    client = get_supabase_client()
    table = get_table_name()

    try:
        # Fetch up to 5000 recent logs for stats calculation since Supabase standard API 
        # doesn't support complex GROUP BY out of the box without RPC
        response = client.table(table).select("session_id, intent, language, latency_ms").order("created_at", desc=True).limit(5000).execute()
        data = response.data

        total_queries = len(data)
        unique_sessions = len(set(d.get("session_id") for d in data if d.get("session_id")))
        
        intent_counts = Counter(d.get("intent") for d in data if d.get("intent"))
        language_counts = Counter(d.get("language") for d in data if d.get("language"))
        
        valid_latencies = [d.get("latency_ms") for d in data if d.get("latency_ms") is not None]
        avg_latency = sum(valid_latencies) // len(valid_latencies) if valid_latencies else 0

        # Radar data format
        intent_radar_data = [{"intent": k, "count": v} for k, v in intent_counts.items()]
        
        # Language donut data format
        language_donut_data = [{"language": k, "count": v} for k, v in language_counts.items()]

        top_intent = intent_counts.most_common(1)[0][0] if intent_counts else "N/A"

        return {
            "totalQueries": total_queries,
            "uniqueSessions": unique_sessions,
            "topIntent": top_intent,
            "avgLatencyMs": avg_latency,
            "intentDistribution": intent_radar_data,
            "languageDistribution": language_donut_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/router-logs/keywords")
async def get_router_logs_keywords():
    client = get_supabase_client()
    table = get_table_name()

    try:
        response = client.table(table).select("keywords").order("created_at", desc=True).limit(1000).execute()
        data = response.data

        keyword_freq = Counter()
        co_occurrence = Counter()

        for row in data:
            keywords = row.get("keywords") or []
            # Filtering out empty and standardizing
            keywords = [k.lower() for k in keywords if k]
            
            for k in keywords:
                keyword_freq[k] += 1
                
            for i in range(len(keywords)):
                for j in range(i + 1, len(keywords)):
                    pair = tuple(sorted([keywords[i], keywords[j]]))
                    co_occurrence[pair] += 1

        # Keep top 30 keywords
        top_keywords = [k for k, _ in keyword_freq.most_common(30)]
        
        nodes = []
        for k in top_keywords:
            nodes.append({"id": k, "group": "keyword", "size": keyword_freq[k]})
            
        links = []
        for pair, count in co_occurrence.items():
            if pair[0] in top_keywords and pair[1] in top_keywords and count > 1:
                links.append({"source": pair[0], "target": pair[1], "value": count})

        return {"nodes": nodes, "links": links}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/router-logs/export")
async def export_router_logs(
    search: Optional[str] = None,
    intent: Optional[str] = None,
    language: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    client = get_supabase_client()
    table = get_table_name()

    query = client.table(table).select("*")

    if search:
        query = query.ilike("user_message", f"%{search}%")
    if intent and intent != "All Intents":
        query = query.eq("intent", intent)
    if language and language != "All Languages":
        query = query.eq("language", language)
    if date_from:
        query = query.gte("created_at", date_from)
    if date_to:
        query = query.lte("created_at", date_to)

    query = query.order("created_at", desc=True)

    try:
        # Supabase default limit is usually 1000. Let's try up to 10000 for export.
        response = query.limit(10000).execute()
        data = response.data

        output = io.StringIO()
        if data:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                # Handle lists in CSV
                if "keywords" in row and isinstance(row["keywords"], list):
                    row["keywords"] = ", ".join(row["keywords"])
                if "router_payload" in row and isinstance(row["router_payload"], dict):
                    import json
                    row["router_payload"] = json.dumps(row["router_payload"])
                writer.writerow(row)
        
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=router_logs.csv"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
