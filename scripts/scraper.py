#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# =========================
# 1) DIRECTORIES & CONFIG
# =========================
BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "0_raw"
PDF_DIR = RAW_DIR / "pdfs"
PDF_META_DIR = PDF_DIR / "_meta"
ARCHIVE_DIR = PDF_DIR / "archive"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)
PDF_META_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

RESEARCH_USER_AGENT = "SriLanka-Medical-Training-Bot/1.0"

# Crawl limits
MAX_DEPTH = 2
MAX_PAGES_PER_SITE = 15
EPID_MAX_DEPTH = 1
EPID_MAX_PAGES = 5

# EPID: prefer literal filenames where possible
EPID_LITERAL_FILENAME_SITES = {"EPID_Reports"}
EPID_FILENAME_KEYS = ("file", "filename", "download_file", "report", "name")

# Control chars
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

# Drop tracking query keys for stable normalization
DROP_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "igshid",
}

# Year detection
YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")

# Downloader heuristics
MIN_PDF_BYTES = 2048
DEFAULT_TIMEOUT_S = 60.0

DEFAULT_CONCURRENCY = 6
POLITE_SLEEP_RANGE = (0.6, 1.2)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 3

# =========================
# PDF INDEX (JSONL LEDGER)
# =========================
PDF_INDEX_PATH = PDF_META_DIR / "pdf_index.jsonl"


def utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


async def append_pdf_index(event: dict, lock: asyncio.Lock) -> None:
    """
    Append one JSON object per line to data/0_raw/pdfs/_meta/pdf_index.jsonl.
    Uses an asyncio lock because downloads run concurrently.
    """
    line = json.dumps(event, ensure_ascii=False) + "\n"
    async with lock:
        PDF_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PDF_INDEX_PATH.open("a", encoding="utf-8") as f:
            f.write(line)


# =========================
# 2) MARKDOWN HELPERS
# =========================
def sanitize_text(s: str) -> str:
    if not s:
        return ""
    return CONTROL_CHARS_RE.sub("", s)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_md_for_compare(md: str) -> str:
    md = sanitize_text(md or "")
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = re.sub(r"[ \t]+", " ", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def build_md_payload(source_url: str, md_text: str) -> str:
    return f"Source URL: {source_url}\n\n{md_text}"


def read_existing_md(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def strip_source_url_header(payload: str) -> str:
    if not payload:
        return ""
    if payload.startswith("Source URL:"):
        parts = payload.split("\n\n", 1)
        if len(parts) == 2:
            return parts[1]
    return payload


def save_markdown_delta(md_path: Path, source_url: str, md_text: str) -> bool:
    md_text = sanitize_text(md_text or "")
    new_body_norm = normalize_md_for_compare(md_text)

    existing_payload = read_existing_md(md_path)
    existing_body = strip_source_url_header(existing_payload)
    existing_body_norm = normalize_md_for_compare(existing_body)

    if sha256_text(existing_body_norm) == sha256_text(new_body_norm):
        print(f"UNCHANGED_MD {md_path.name}")
        return False

    md_path.write_text(build_md_payload(source_url, md_text), encoding="utf-8")
    print(f"UPDATED_MD {md_path.name}")
    return True


# =========================
# 3) URL NORMALIZATION
# =========================
def normalize_url(u: str) -> str:
    """
    - Accept only http/https
    - Fix protocol-relative //example.com/x.pdf
    - Fix scheme-less domain example.com/x.pdf
    - Drop fragments
    - Drop tracking params, keep others in stable order
    """
    if not u:
        return ""
    u = u.strip()
    if not u:
        return ""

    if u.startswith("//"):
        u = "https:" + u

    if "://" not in u and re.match(r"^[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(/|$)", u):
        u = "https://" + u

    parts = urlsplit(u)
    scheme = (parts.scheme or "").lower()
    if scheme not in ("http", "https"):
        return ""

    netloc = (parts.netloc or "").lower()
    if not netloc:
        return ""

    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    fragment = ""

    if parts.query:
        q = [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k not in DROP_QUERY_KEYS
        ]
        q_sorted = sorted(q, key=lambda kv: (kv[0], kv[1]))
        query = urlencode(q_sorted, doseq=True)
    else:
        query = ""

    return urlunsplit((scheme, netloc, path, query, fragment))


def detect_year_from_text(s: str) -> Optional[int]:
    m = YEAR_RE.search(s or "")
    if not m:
        return None
    try:
        y = int(m.group(1))
        if 1900 <= y <= 2099:
            return y
    except Exception:
        return None
    return None


# =========================
# 4) PDF FILENAME STRATEGY
# =========================
def epid_filename_from_url(norm_url: str) -> str:
    parts = urlsplit(norm_url)
    base = Path(parts.path).name
    if base.lower().endswith(".pdf"):
        return base

    qs = dict(parse_qsl(parts.query, keep_blank_values=True))
    for k in EPID_FILENAME_KEYS:
        v = qs.get(k)
        if v and v.lower().endswith(".pdf"):
            return Path(v).name

    return ""


def make_pdf_filename(site_prefix: str, norm_url: str) -> str:
    """
    Deterministic filename:
    - EPID_Reports: prefer literal filename if possible
    - others: stable hash of normalized URL
    """
    if site_prefix in EPID_LITERAL_FILENAME_SITES:
        lit = epid_filename_from_url(norm_url)
        if lit:
            lit = re.sub(r"[^A-Za-z0-9._-]+", "_", lit)[:180]
            if not lit.lower().endswith(".pdf"):
                lit += ".pdf"
            return f"{site_prefix}__{lit}"
        key = sha256_text(norm_url)[:16]
        return f"{site_prefix}__{key}.pdf"

    key = sha256_text(norm_url)[:16]
    return f"{site_prefix}__{key}.pdf"


# =========================
# 5) EXISTENCE + ARCHIVE
# =========================
def resolve_existing_pdf_path(filename: str) -> Optional[Path]:
    """
    Check active + archived folders for an exact filename match.
      active: data/0_raw/pdfs/<filename>
      archive: data/0_raw/pdfs/archive/YYYY/<filename>
    """
    p_active = PDF_DIR / filename
    if p_active.exists():
        return p_active

    hits = list(ARCHIVE_DIR.rglob(filename))
    if hits:
        return hits[0]
    return None


def archive_pdf_by_year_if_needed(filename: str, norm_url: str, cutoff_year: int) -> bool:
    year = detect_year_from_text(norm_url) or detect_year_from_text(filename)
    if year is None or year >= cutoff_year:
        return False

    src = PDF_DIR / filename
    if not src.exists():
        return False

    year_dir = ARCHIVE_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    dest = year_dir / filename

    if dest.exists():
        src.unlink(missing_ok=True)
        print(f"PDF_ARCHIVED_ALREADY {filename} -> pdfs/archive/{year}/")
        return True

    shutil.move(str(src), str(dest))
    print(f"PDF_ARCHIVED {filename} -> pdfs/archive/{year}/")
    return True


# =========================
# 6) PDF DOWNLOADER CORE
# =========================
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def looks_like_pdf(content: bytes) -> bool:
    return len(content) >= 5 and content[:5] == b"%PDF-"


def existing_pdf_is_healthy(path: Path) -> bool:
    try:
        if not path.exists():
            return False
        if path.stat().st_size < MIN_PDF_BYTES:
            return False
        with path.open("rb") as f:
            head = f.read(5)
        return head == b"%PDF-"
    except Exception:
        return False


def atomic_write_bytes_no_overwrite(path: Path, content: bytes) -> bool:
    """
    Atomic write, but will NOT overwrite if the target appears at the last moment.
    Returns True if wrote, False if skipped.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_bytes(content)

    if path.exists():
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False

    os.replace(tmp, path)
    return True


def atomic_replace_bytes(path: Path, content: bytes) -> None:
    """
    Atomic replace (OVERWRITES). Only used when repair_bad=True and existing file is unhealthy.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, path)


async def fetch_with_retries(client: httpx.AsyncClient, url: str) -> Optional[httpx.Response]:
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = await client.get(url)

            if r.status_code not in RETRYABLE_STATUS:
                return r

            if attempt == MAX_RETRIES:
                return r

            retry_after = r.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                sleep_s = min(60.0, float(retry_after))
            else:
                sleep_s = (2**attempt) + random.uniform(0.2, 0.8)

            await asyncio.sleep(sleep_s)

        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError):
            if attempt == MAX_RETRIES:
                return None
            await asyncio.sleep((2**attempt) + random.uniform(0.2, 0.8))
        except Exception:
            return None

    return None


@dataclass(frozen=True)
class PdfJob:
    url: str
    site_prefix: str


async def download_one_pdf(
    client: httpx.AsyncClient,
    job: PdfJob,
    *,
    backfill: bool,
    archive_before_year: Optional[int],
    repair_bad: bool,
    index_lock: asyncio.Lock,
) -> bool:
    """
    Rules:
    - If PDF exists (active OR archive) and is healthy => SKIP (no overwrite, no rewrite).
    - If PDF exists and is unhealthy:
        - repair_bad=False => SKIP
        - repair_bad=True  => re-download and OVERWRITE ONLY that unhealthy active file
          (never overwrites archive copies)
    - If file doesn't exist => write once, atomically, with a final no-overwrite gate.
    - If a race happens and file appears between fetch and write => skip.
    """
    norm_url = normalize_url(job.url)
    if not norm_url:
        await append_pdf_index(
            {
                "ts": utc_ts(),
                "site": job.site_prefix,
                "url": job.url,
                "norm_url": "",
                "filename": "",
                "status": "failed",
                "reason": "bad_url",
            },
            index_lock,
        )
        return False

    filename = make_pdf_filename(job.site_prefix, norm_url)
    existing = resolve_existing_pdf_path(filename)

    # If file exists anywhere and is healthy, skip early (no network hit)
    if existing and existing_pdf_is_healthy(existing):
        print(f"✅ SKIP_EXISTS: {filename}")
        await append_pdf_index(
            {
                "ts": utc_ts(),
                "site": job.site_prefix,
                "url": job.url,
                "norm_url": norm_url,
                "filename": filename,
                "path": str(existing),
                "status": "skipped_exists",
            },
            index_lock,
        )
        return True

    # If exists but unhealthy
    if existing and not existing_pdf_is_healthy(existing):
        if (ARCHIVE_DIR in existing.parents) or (existing.parent == ARCHIVE_DIR):
            print(f"⚠️  SKIP_BAD_ARCHIVE_EXISTS: {filename}")
            await append_pdf_index(
                {
                    "ts": utc_ts(),
                    "site": job.site_prefix,
                    "url": job.url,
                    "norm_url": norm_url,
                    "filename": filename,
                    "path": str(existing),
                    "status": "skipped_bad_archive_exists",
                },
                index_lock,
            )
            return True

        if not repair_bad:
            print(f"⚠️  SKIP_BAD_EXISTS (repair disabled): {filename}")
            await append_pdf_index(
                {
                    "ts": utc_ts(),
                    "site": job.site_prefix,
                    "url": job.url,
                    "norm_url": norm_url,
                    "filename": filename,
                    "path": str(existing),
                    "status": "skipped_bad_exists",
                },
                index_lock,
            )
            return True

        # else: we will attempt to repair by overwriting this unhealthy active file

    # Polite delay before network hit
    await asyncio.sleep(random.uniform(*POLITE_SLEEP_RANGE))

    r = await fetch_with_retries(client, norm_url)
    if r is None or r.status_code != 200:
        await append_pdf_index(
            {
                "ts": utc_ts(),
                "site": job.site_prefix,
                "url": job.url,
                "norm_url": norm_url,
                "filename": filename,
                "status": "failed",
                "reason": "http_error" if r is not None else "no_response",
                "http_status": None if r is None else r.status_code,
            },
            index_lock,
        )
        return False

    ctype = (r.headers.get("Content-Type") or "").lower()
    content = r.content

    is_pdf = ("application/pdf" in ctype) or looks_like_pdf(content) or norm_url.lower().endswith(".pdf")
    if not is_pdf:
        await append_pdf_index(
            {
                "ts": utc_ts(),
                "site": job.site_prefix,
                "url": job.url,
                "norm_url": norm_url,
                "filename": filename,
                "status": "failed",
                "reason": "not_pdf",
                "content_type": ctype,
                "bytes": len(content),
            },
            index_lock,
        )
        return False

    if len(content) < MIN_PDF_BYTES:
        await append_pdf_index(
            {
                "ts": utc_ts(),
                "site": job.site_prefix,
                "url": job.url,
                "norm_url": norm_url,
                "filename": filename,
                "status": "failed",
                "reason": "too_small",
                "content_type": ctype,
                "bytes": len(content),
            },
            index_lock,
        )
        return False

    out_path = PDF_DIR / filename

    # If file exists now, avoid rewriting identical bytes
    if out_path.exists():
        # If healthy now, skip (someone else saved it)
        if existing_pdf_is_healthy(out_path):
            print(f"✅ SKIP_EXISTS: {filename}")
            await append_pdf_index(
                {
                    "ts": utc_ts(),
                    "site": job.site_prefix,
                    "url": job.url,
                    "norm_url": norm_url,
                    "filename": filename,
                    "path": str(out_path),
                    "status": "skipped_exists",
                    "note": "appeared_during_fetch",
                },
                index_lock,
            )
            return True

        # If unhealthy and repair_bad enabled, only overwrite if content differs
        if repair_bad:
            try:
                old_hash = file_sha256(out_path) if out_path.stat().st_size > 0 else ""
                new_hash = sha256_bytes(content)
                if old_hash and old_hash == new_hash:
                    print(f"✅ SKIP_SAME_CONTENT (bad file but same bytes): {filename}")
                    await append_pdf_index(
                        {
                            "ts": utc_ts(),
                            "site": job.site_prefix,
                            "url": job.url,
                            "norm_url": norm_url,
                            "filename": filename,
                            "path": str(out_path),
                            "status": "skipped_same_content",
                            "sha256": new_hash,
                            "bytes": len(content),
                            "content_type": ctype,
                        },
                        index_lock,
                    )
                    return True

                atomic_replace_bytes(out_path, content)
                print(f"🛠️  PDF_REPAIRED: {filename}")
                await append_pdf_index(
                    {
                        "ts": utc_ts(),
                        "site": job.site_prefix,
                        "url": job.url,
                        "norm_url": norm_url,
                        "filename": filename,
                        "path": str(out_path),
                        "status": "repaired",
                        "sha256": new_hash,
                        "bytes": len(content),
                        "content_type": ctype,
                    },
                    index_lock,
                )
            except Exception as e:
                print(f"PDF_REPAIR_ERROR {job.site_prefix}: {e}")
                await append_pdf_index(
                    {
                        "ts": utc_ts(),
                        "site": job.site_prefix,
                        "url": job.url,
                        "norm_url": norm_url,
                        "filename": filename,
                        "path": str(out_path),
                        "status": "failed",
                        "reason": "repair_error",
                        "error": str(e),
                    },
                    index_lock,
                )
                return False
        else:
            print(f"⚠️  SKIP_EXISTING_UNSAFE: {filename}")
            await append_pdf_index(
                {
                    "ts": utc_ts(),
                    "site": job.site_prefix,
                    "url": job.url,
                    "norm_url": norm_url,
                    "filename": filename,
                    "path": str(out_path),
                    "status": "skipped_existing_unsafe",
                },
                index_lock,
            )
            return True
    else:
        wrote = atomic_write_bytes_no_overwrite(out_path, content)
        if not wrote:
            print(f"✅ SKIP_RACE_EXISTS: {filename}")
            await append_pdf_index(
                {
                    "ts": utc_ts(),
                    "site": job.site_prefix,
                    "url": job.url,
                    "norm_url": norm_url,
                    "filename": filename,
                    "path": str(out_path),
                    "status": "skipped_race_exists",
                },
                index_lock,
            )
            return True

        print(f"💾 PDF_SAVED: {filename}")
        await append_pdf_index(
            {
                "ts": utc_ts(),
                "site": job.site_prefix,
                "url": job.url,
                "norm_url": norm_url,
                "filename": filename,
                "path": str(out_path),
                "status": "saved",
                "sha256": sha256_bytes(content),
                "bytes": len(content),
                "content_type": ctype,
            },
            index_lock,
        )

    if archive_before_year is not None:
        archived = archive_pdf_by_year_if_needed(filename, norm_url, archive_before_year)
        if archived:
            # log archive move as a separate event (optional but helpful)
            year = detect_year_from_text(norm_url) or detect_year_from_text(filename)
            await append_pdf_index(
                {
                    "ts": utc_ts(),
                    "site": job.site_prefix,
                    "url": job.url,
                    "norm_url": norm_url,
                    "filename": filename,
                    "status": "archived",
                    "archive_year": year,
                    "path": str((ARCHIVE_DIR / str(year) / filename)) if year else None,
                },
                index_lock,
            )

    return True


async def run_pdf_downloads(
    client: httpx.AsyncClient,
    jobs: list[PdfJob],
    *,
    backfill: bool,
    archive_before_year: Optional[int],
    repair_bad: bool,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> tuple[int, int]:
    sem = asyncio.Semaphore(max(1, concurrency))
    index_lock = asyncio.Lock()
    ok = 0
    fail = 0

    async def _one(job: PdfJob) -> bool:
        async with sem:
            return await download_one_pdf(
                client,
                job,
                backfill=backfill,
                archive_before_year=archive_before_year,
                repair_bad=repair_bad,
                index_lock=index_lock,
            )

    tasks = [asyncio.create_task(_one(j)) for j in jobs]
    for t in asyncio.as_completed(tasks):
        try:
            if await t:
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1

    return ok, fail


# =========================
# 7) PDF URL EXTRACTION
# =========================
def extract_pdf_urls(page_result) -> set[str]:
    """
    - accept .pdf in path
    - accept query param values that end with .pdf (e.g. ?file=abc.pdf)
    """
    urls: set[str] = set()
    try:
        links = (page_result.links.get("internal", []) or []) + (page_result.links.get("external", []) or [])
        for l in links:
            href = (l.get("href") or "").strip()
            if not href:
                continue

            abs_url = urljoin(page_result.url, href)
            norm = normalize_url(abs_url)
            if not norm:
                continue

            parts = urlsplit(norm)
            path = (parts.path or "").lower()
            if path.endswith(".pdf"):
                urls.add(norm)
                continue

            qs = dict(parse_qsl(parts.query, keep_blank_values=True))
            if any((v or "").lower().endswith(".pdf") for v in qs.values()):
                urls.add(norm)
                continue

    except Exception:
        pass
    return urls


# =========================
# 8) MAIN PIPELINE
# =========================
async def main(*, backfill: bool, archive_before_year: Optional[int], repair_bad: bool, concurrency: int):
    medical_filter = PruningContentFilter(threshold=0.45, threshold_type="dynamic")
    md_generator = DefaultMarkdownGenerator(content_filter=medical_filter)

    browser_config = BrowserConfig(headless=True, user_agent=RESEARCH_USER_AGENT)

    GOVT_TARGETS = [
        {"name": "MOH_Main", "url": "https://www.health.gov.lk/"},
        {"name": "HPB_HealthTopics", "url": "https://www.hpb.health.gov.lk/en/health-topics"},
        {"name": "FHB_FamilyHealthBureau", "url": "https://www.fhb.health.gov.lk/"},
        {"name": "NIHS_HealthSciences", "url": "https://www.nihs.gov.lk/"},
        {"name": "NMRA_MedicinesInfo", "url": "https://www.nmra.gov.lk/index.php?option=com_content&view=article&id=138&lang=en"},
        {"name": "Dengue_HealthGov", "url": "https://www.dengue.health.gov.lk/web/"},
        {"name": "MRI_MedicalResearchInstitute", "url": "https://www.mri.gov.lk/"},
        {"name": "NHSL_NationalHospitalSL", "url": "https://www.nhsl.health.gov.lk/"},
        {"name": "NHK_NationalHospitalKandy", "url": "https://nhkandy.org/"},
        {"name": "LRH_LadyRidgeway", "url": "https://lrh.health.gov.lk/"},
        {"name": "CSHW_CastleStreetWomen", "url": "https://www.cshw.health.gov.lk/"},
        {"name": "NCI_Apeksha_NCISL", "url": "https://www.ncisl.health.gov.lk/"},
        {"name": "NEH_NationalEyeHospital", "url": "https://nationaleyehospital.health.gov.lk/"},
        {"name": "NIMH_MentalHealth", "url": "https://nimh.health.gov.lk/"},
        {"name": "EPID_Immunization", "url": "https://www.epid.gov.lk/vaccine-preventable-diseases"},
        {"name": "EPID_Reports", "url": "https://www.epid.gov.lk/weekly-epidemiological-report"},
    ]

    async with httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT_S,
        follow_redirects=True,
        headers={"User-Agent": RESEARCH_USER_AGENT},
    ) as pdf_client, AsyncWebCrawler(config=browser_config) as crawler:
        for target in GOVT_TARGETS:
            print(f"\n--- SCANNING {target['name']} ---")

            is_epid = target["name"].startswith("EPID_")
            depth = EPID_MAX_DEPTH if is_epid else MAX_DEPTH
            pages = EPID_MAX_PAGES if is_epid else MAX_PAGES_PER_SITE

            seen_pdfs_site: set[str] = set()
            jobs: list[PdfJob] = []

            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                markdown_generator=md_generator,
                deep_crawl_strategy=BFSDeepCrawlStrategy(max_depth=depth, max_pages=pages),
                mean_delay=2.0,
                wait_until="networkidle",
            )

            try:
                results = await crawler.arun(url=target["url"], config=run_config)
                crawl_list = results if isinstance(results, list) else [results]
                print(f"FOUND_PAGES {len(crawl_list)} (depth={depth}, max_pages={pages})")

                for page in crawl_list:
                    if not page or not page.success:
                        continue

                    page_url = page.url
                    page_key = sha256_text(page_url)[:16]

                    md_text = page.markdown.fit_markdown or page.markdown.raw_markdown or ""
                    md_text = sanitize_text(md_text)

                    md_filename = RAW_DIR / f"{target['name']}__{page_key}.md"
                    save_markdown_delta(md_filename, page_url, md_text)

                    for pdf_url in extract_pdf_urls(page):
                        pdf_norm = normalize_url(pdf_url)
                        if not pdf_norm or pdf_norm in seen_pdfs_site:
                            continue
                        seen_pdfs_site.add(pdf_norm)
                        jobs.append(PdfJob(url=pdf_norm, site_prefix=target["name"]))

            except Exception as e:
                print(f"SITE_ERROR {target['name']}: {e}")
                continue

            if jobs:
                ok, fail = await run_pdf_downloads(
                    pdf_client,
                    jobs,
                    backfill=backfill,
                    archive_before_year=archive_before_year,
                    repair_bad=repair_bad,
                    concurrency=concurrency,
                )
                print(f"PDF_RESULTS {target['name']} ok={ok} fail={fail} total={len(jobs)}")
            else:
                print(f"NO_PDFS_FOUND {target['name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sri Lankan medical govt scraper (web + PDFs, merged)")
    parser.add_argument("--backfill", action="store_true", help="Download missing PDFs (still no overwrites).")
    parser.add_argument(
        "--archive-before-year",
        type=int,
        default=2023,
        help="Archive PDFs with detectable year < this into data/0_raw/pdfs/archive/YYYY/. Use 0 to disable.",
    )
    parser.add_argument(
        "--repair-bad",
        action="store_true",
        help="Overwrite ONLY if an existing PDF in active folder is clearly invalid/tiny/not-a-PDF.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent PDF downloads per site (default {DEFAULT_CONCURRENCY}).",
    )

    args = parser.parse_args()
    archive_before_year = None if args.archive_before_year == 0 else args.archive_before_year

    asyncio.run(
        main(
            backfill=args.backfill,
            archive_before_year=archive_before_year,
            repair_bad=args.repair_bad,
            concurrency=max(1, int(args.concurrency)),
        )
    )