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
)
from agents.src.utils import setup_logger

logger = setup_logger("supabase_loader")

_FETCH_COLUMNS = "doc_id,markdown,source_url,source_pdf,title,site"


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
      - page_content: the 'markdown' field of the row
      - metadata:
          source_reference: source_url → source_pdf → doc_id  (used for citations)
          title:            document title if available
          site:             originating site name
          doc_id:           primary key of the row
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set in agents/.env"
        )

    base_url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{SUPABASE_TABLE}"

    documents: List[Document] = []
    offset = 0
    skipped = 0

    logger.info(f"Fetching documents from Supabase table '{SUPABASE_TABLE}' ...")

    while True:
        params = urllib.parse.urlencode({
            "select": _FETCH_COLUMNS,
            "offset": offset,
            "limit": SUPABASE_PAGE_SIZE,
        })
        rows = _get(f"{base_url}?{params}")

        if not rows:
            break

        for row in rows:
            markdown = (row.get("markdown") or "").strip()
            if not markdown:
                skipped += 1
                continue

            source_url = (row.get("source_url") or "").strip()
            source_pdf = (row.get("source_pdf") or "").strip()
            doc_id = (row.get("doc_id") or "").strip()

            source_reference = source_url or source_pdf or doc_id

            documents.append(
                Document(
                    page_content=markdown,
                    metadata={
                        "source_reference": source_reference,
                        "title": (row.get("title") or "").strip(),
                        "site": (row.get("site") or "").strip(),
                        "doc_id": doc_id,
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
        f"{skipped} rows skipped (empty markdown)"
    )
    return documents
