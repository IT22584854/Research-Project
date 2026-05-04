"""Load medical documents from Supabase into LangChain Document objects."""
from __future__ import annotations

import sys
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.src.config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_TABLE,
    SUPABASE_PAGE_SIZE,
    SUPABASE_NON_NULL_COLUMN,
)
from agents.src.utils import setup_logger

logger = setup_logger("supabase_loader")

_FETCH_COLUMNS = "site_root_url,source_pdf,clean_text"


def _get(url: str) -> list:
    """Simple GET using stdlib urllib to avoid httpcore DNS issues on Windows."""
    req = urllib.request.Request(
        url,
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def load_documents_from_supabase() -> List[Document]:
    """
    Fetch all rows from the Supabase medical corpus table and return them
    as LangChain Document objects.

        Each Document has:
            - page_content: the 'clean_text' field of the row
            - metadata:
                    source_reference: site_root_url → source_pdf (used for citations)
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set in agents/.env"
        )

    base_url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{SUPABASE_TABLE}"

    documents: List[Document] = []
    offset = 0
    skipped = 0

    filter_note = (
        f" (filtering {SUPABASE_NON_NULL_COLUMN} != NULL)"
        if SUPABASE_NON_NULL_COLUMN
        else ""
    )
    logger.info(
        f"Fetching documents from Supabase table '{SUPABASE_TABLE}'{filter_note} ..."
    )

    while True:
        query_params = {
            "select": _FETCH_COLUMNS,
            "offset": offset,
            "limit": SUPABASE_PAGE_SIZE,
        }
        if SUPABASE_NON_NULL_COLUMN:
            # PostgREST filter: column=not.is.null
            query_params[SUPABASE_NON_NULL_COLUMN] = "not.is.null"

        params = urllib.parse.urlencode(query_params)
        rows = _get(f"{base_url}?{params}")

        if not rows:
            break

        for row in rows:
            clean_text = (row.get("clean_text") or "").strip()
            if not clean_text:
                skipped += 1
                continue

            site_root_url = (row.get("site_root_url") or "").strip()
            source_pdf = (row.get("source_pdf") or "").strip()
            source_reference = site_root_url or source_pdf

            documents.append(
                Document(
                    page_content=clean_text,
                    metadata={
                        "source_reference": source_reference,
                    },
                )
            )

        logger.info(
            f"  Fetched rows {offset}–{offset + len(rows) - 1} "
            f"({len(documents)} documents loaded so far)"
        )

        if len(rows) < SUPABASE_PAGE_SIZE:
            break

        offset += SUPABASE_PAGE_SIZE

    logger.info(
        f"Supabase load complete — {len(documents)} documents loaded, "
        f"{skipped} rows skipped (empty clean_text)"
    )
    return documents
