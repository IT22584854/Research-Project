import json
import hashlib
import re
import csv
from pathlib import Path
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from collections import Counter

CLEAN_ROOT   = Path("data/1_cleaned")
RAW_ROOT     = Path("data/0_raw")
REFINED_ROOT = Path("data/2_refined")
OUT_PATH     = Path("data/2_processed/manifest.jsonl")
SKIP_LOG     = Path("data/2_processed/skipped_pages.csv")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SKIP_DIRS = {"history", "_duplicates"}

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

TARGET_NAMES = [t["name"] for t in GOVT_TARGETS]
TARGET_NAMES_UPPER = [n.upper() for n in TARGET_NAMES]
TARGET_URL_BY_NAME = {t["name"]: t["url"] for t in GOVT_TARGETS}

SOURCE_URL_RE = re.compile(r"^Source URL:\s*(.+?)\s*$", re.MULTILINE)
SOURCE_PDF_RE = re.compile(r"^Source PDF:\s*(.+?)\s*$", re.MULTILINE)
SOURCE_PDF_URL_RE = re.compile(r"^Source PDF URL:\s*(.+?)\s*$", re.MULTILINE)

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "have", "has", "had", "will", "shall", "can", "could", "would", "should",
    "you", "your", "about", "into", "than", "then", "them", "they", "their",
    "his", "her", "its", "our", "out", "not", "but", "who", "what", "when",
    "where", "why", "how", "all", "any", "each", "more", "most", "some",
    "such", "only", "also", "may", "might", "been", "being", "is", "am", "an",
    "a", "of", "to", "in", "on", "at", "by", "or", "as", "it",
    "source", "url", "pdf", "http", "https", "www", "gov", "lk", "html",
    "page", "home", "main", "content", "article", "index", "view", "lang",
    "health", "ministry", "hospital", "national", "department", "sri", "lanka"
}

DROP_QUERY_KEYS = {
    "utm_source","utm_medium","utm_campaign","utm_term","utm_content",
    "fbclid","gclid","igshid"
}

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def should_skip(path: Path) -> bool:
    return any(part.lower() in SKIP_DIRS for part in path.parts)

def normalize_url(u: str) -> str | None:
    if not u:
        return None
    u = u.strip()

    parts = urlsplit(u)
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()

    if not netloc and parts.path.startswith("www."):
        parts2 = urlsplit("https://" + u)
        scheme, netloc = parts2.scheme.lower(), parts2.netloc.lower()
        path, query, frag = parts2.path, parts2.query, parts2.fragment
    else:
        path, query, frag = parts.path, parts.query, parts.fragment

    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    if query:
        q = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True) if k not in DROP_QUERY_KEYS]
        query = urlencode(q, doseq=True)

    return urlunsplit((scheme, netloc, path or "/", query, ""))

def make_source_key(source_type: str,
                    source_url: str | None,
                    source_pdf_url: str | None,
                    source_pdf: str | None) -> str | None:
    """
    Returns a stable source identity hash:
      - web: sha256("web:" + normalized_source_url)
      - pdf: sha256("pdf:" + normalized_source_pdf_url) else sha256("pdffile:" + filename)
    """
    if source_type == "web" and source_url:
        return sha256_text("web:" + source_url)
    if source_type == "pdf":
        if source_pdf_url:
            return sha256_text("pdf:" + source_pdf_url)
        if source_pdf:
            return sha256_text("pdffile:" + source_pdf)
    return None

def language_profile(text: str) -> dict:
    si = sum(1 for ch in text if "\u0D80" <= ch <= "\u0DFF")
    ta = sum(1 for ch in text if "\u0B80" <= ch <= "\u0BFF")
    en = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    total = si + ta + en
    if total == 0:
        return {"en": 0.0, "si": 0.0, "ta": 0.0}
    return {"en": en / total, "si": si / total, "ta": ta / total}

def classify_language(text: str, mixed_threshold: float = 0.15) -> dict:
    prof = language_profile(text)
    ranked = sorted(prof.items(), key=lambda kv: kv[1], reverse=True)
    top_lang, top_val = ranked[0]
    second_lang, second_val = ranked[1]

    mixed_langs = [k for k, v in prof.items() if v >= mixed_threshold]
    label = "mixed" if len(mixed_langs) >= 2 else top_lang

    top2 = [top_lang]
    if second_val >= 0.20 * top_val and second_val > 0:
        top2.append(second_lang)

    return {
        "language_label": label,
        "language_primary": top_lang,
        "language_top": top2,
        "language_profile": prof,
    }

def extract_title(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]

    for ln in lines:
        if ln.startswith("# "):
            return ln[2:].strip()[:200]
        if ln.startswith("## "):
            return ln[3:].strip()[:200]

    for ln in lines:
        if not ln:
            continue
        low = ln.lower()
        if low.startswith("source url:") or low.startswith("source pdf:") or low.startswith("source pdf url:") or low.startswith("source "):
            continue
        if len(ln) < 8:
            continue
        if not any(ch.isalpha() for ch in ln):
            continue
        return ln[:200]

    return ""

def junk_reason(title: str, text: str) -> str | None:
    t = (title or "").strip().lower()
    low = (text or "").lower()

    if t in {"404", "page not found", "not found"}:
        return "title_404"
    if "look like something went wrong" in low:
        return "body_error_message"
    if "the page you were looking for is not here" in low:
        return "body_not_here"
    if re.search(r"\b404\b", low) and "not found" in low:
        return "body_404_not_found"

    return None

def infer_site_from_filename(filename: str) -> str | None:
    stem = Path(filename).stem
    if stem.startswith("PDF_EXTRACT_"):
        stem = stem[len("PDF_EXTRACT_"):]

    upper = stem.upper()

    for target_name, target_upper in zip(TARGET_NAMES, TARGET_NAMES_UPPER):
        if upper.startswith(target_upper):
            return target_name
        if (target_upper + "_") in upper:
            return target_name
        if (target_upper + "__") in upper:
            return target_name

    first_token = upper.split("_", 1)[0]
    candidates = [
        name for name in TARGET_NAMES
        if name.upper().startswith(first_token + "_") or name.upper() == first_token
    ]
    if len(candidates) == 1:
        return candidates[0]

    return None

def infer_raw_path(clean_rel: str) -> str:
    return (RAW_ROOT / clean_rel).as_posix()

def extract_keywords(text: str, top_k: int = 12) -> list[str]:
    if not text:
        return []

    text = text.lower()

    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text)

    filtered = []
    for tok in tokens:
        tok = tok.strip("-")
        if len(tok) < 3:
            continue
        if tok in STOPWORDS:
            continue
        if tok.isdigit():
            continue
        filtered.append(tok)

    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_k)]

def build_keyword_source(title: str | None,
                         site: str | None,
                         source_url: str | None,
                         markdown: str,
                         max_chars: int = 4000) -> str:
    parts = [
        title or "",
        (site or "").replace("_", " "),
        source_url or "",
        markdown[:max_chars] if markdown else "",
    ]
    return "\n".join(p for p in parts if p).strip()

def main():
    files = sorted(CLEAN_ROOT.rglob("*.md"))
    written = 0
    skipped_junk = 0
    skipped_duplicate_doc_ids = 0
    seen_doc_ids = set()

    with SKIP_LOG.open("w", newline="", encoding="utf-8") as flog:
        wlog = csv.writer(flog)
        wlog.writerow([
            "clean_rel", "source_url", "source_pdf", "source_pdf_url",
            "site", "title", "reason"
        ])

        with OUT_PATH.open("w", encoding="utf-8") as out:
            for p in files:
                if should_skip(p):
                    continue

                text = p.read_text(encoding="utf-8", errors="ignore")
                clean_rel = p.relative_to(CLEAN_ROOT).as_posix()

                m_url = SOURCE_URL_RE.search(text)
                m_pdf = SOURCE_PDF_RE.search(text)
                m_pdf_url = SOURCE_PDF_URL_RE.search(text)

                source_url = normalize_url(m_url.group(1)) if m_url else None
                source_pdf = m_pdf.group(1).strip() if m_pdf else None
                source_pdf_url = normalize_url(m_pdf_url.group(1)) if m_pdf_url else None

                if source_url:
                    source_type = "web"
                elif source_pdf_url or source_pdf:
                    source_type = "pdf"
                else:
                    source_type = "unknown"

                source_key = make_source_key(source_type, source_url, source_pdf_url, source_pdf)

                site = infer_site_from_filename(p.name)
                site_root_url = normalize_url(TARGET_URL_BY_NAME.get(site)) if site else None

                refined_candidate = REFINED_ROOT / clean_rel
                refined_path = refined_candidate.as_posix() if refined_candidate.exists() else None

                title = extract_title(text)
                reason = junk_reason(title, text)
                if reason:
                    skipped_junk += 1
                    wlog.writerow([clean_rel, source_url, source_pdf, source_pdf_url, site, title, reason])
                    continue

                lang = classify_language(text)

                keyword_source = build_keyword_source(
                    title=title,
                    site=site,
                    source_url=source_url,
                    markdown=text
                )
                keywords = extract_keywords(keyword_source, top_k=12)

                st = p.stat()
                retrieved_at = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")

                doc_id = source_key if source_key else sha256_text(text)

                if doc_id in seen_doc_ids:
                    skipped_duplicate_doc_ids += 1
                    wlog.writerow([clean_rel, source_url, source_pdf, source_pdf_url, site, title, "duplicate_doc_id"])
                    continue

                seen_doc_ids.add(doc_id)

                record = {
                    "doc_id": doc_id,
                    "source_key": source_key,
                    "source_type": source_type,
                    "source_url": source_url,
                    "source_pdf": source_pdf,
                    "source_pdf_url": source_pdf_url,
                    "site": site,
                    "site_root_url": site_root_url,
                    "raw_path": infer_raw_path(clean_rel),
                    "clean_path": p.as_posix(),
                    "refined_path": refined_path,
                    "clean_rel": clean_rel,
                    "retrieved_at": retrieved_at,
                    "bytes": st.st_size,
                    "title": title,
                    "keywords": keywords,
                    "language": lang["language_label"],
                    "language_primary": lang["language_primary"],
                    "language_top": lang["language_top"],
                    "language_profile": lang["language_profile"],
                    "markdown": text,
                    "clean_text": None,
                }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote manifest: {OUT_PATH} ({written} records)")
    print(f"Skipped junk/404 pages: {skipped_junk}")
    print(f"Skipped duplicate doc_ids: {skipped_duplicate_doc_ids}")
    print(f"Wrote skip log: {SKIP_LOG}")

if __name__ == "__main__":
    main()
