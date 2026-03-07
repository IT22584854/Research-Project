import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# =========================
# PATHS
# =========================
PDF_DIR = Path("data/0_raw/pdfs")
PDF_META_DIR = PDF_DIR / "_meta"

OUT_DIR = Path("data/0_raw/pdf_md")
HISTORY_DIR = OUT_DIR / "history"
SIG_DIR = OUT_DIR / "sigs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
SIG_DIR.mkdir(parents=True, exist_ok=True)

QC_REPORT = OUT_DIR / "qc_report.csv"
FAIL_LOG = OUT_DIR / "failures.txt"

# =========================
# SETTINGS
# =========================
PIPELINE_VERSION = "v5_meta_provenance_contentSig_textFirst_scale0p5_tablesOff_ocrRetrySizeGate"

IMAGES_SCALE_TEXT = 0.5
IMAGES_SCALE_OCR = 0.5
DO_TABLES_TEXT_PASS = False
DO_TABLES_OCR_PASS = False

RAPIDOCR_BACKEND = "onnxruntime"
FORCE_FULL_PAGE_OCR = False

MAX_OCR_MB = 35
MAX_NUM_PAGES = 300
MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB

# QC thresholds
MIN_TEXT_CHARS = 400
MIN_ALPHA_RATIO = 0.25
MIN_KEYWORD_HITS = 1

MEDICAL_KEYWORDS = [
    "dengue", "influenza", "measles", "rabies", "leptospirosis",
    "cholera", "malaria", "tuberculosis", "antibiotic", "paracetamol",
    "vaccine", "immunization", "guideline", "surveillance",
    "ප්‍රතිජීවක", "පැරසිටමෝල්", "එන්නත", "වෛද්‍ය",
    "மருந்து", "தடுப்பூசி", "சிகிச்சை"
]

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

# tracking params for stable URLs (match your scraper + manifest)
DROP_QUERY_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "igshid"
}

# =========================
# SMALL UTILITIES (SAFE WRITES)
# =========================
def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)

def atomic_write_bytes(path: Path, b: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(b)
    tmp.replace(path)

def normalize_url(u: str) -> str:
    if not u:
        return ""
    u = u.strip()
    parts = urlsplit(u)

    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()

    # if someone stored "www.xxx.lk/abc" without scheme
    if not netloc and parts.path.startswith("www."):
        parts2 = urlsplit("https://" + u)
        scheme, netloc = parts2.scheme.lower(), parts2.netloc.lower()
        path, query = parts2.path, parts2.query
    else:
        path, query = parts.path, parts.query

    if not path:
        path = "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    if query:
        q = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True) if k not in DROP_QUERY_KEYS]
        query = urlencode(q, doseq=True)

    return urlunsplit((scheme, netloc, path, query, ""))

# =========================
# TEXT HELPERS
# =========================
def sanitize_text(s: str) -> str:
    return CONTROL_CHARS_RE.sub("", s or "")

def normalize_for_hash(s: str) -> str:
    s = sanitize_text(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def archive_old(out_path: Path) -> None:
    if not out_path.exists():
        return
    stem, suffix = out_path.stem, out_path.suffix
    pattern = re.compile(rf"^{re.escape(stem)}__v(\d{{4}}){re.escape(suffix)}$")
    versions = []
    for f in HISTORY_DIR.glob(f"{stem}__v*{suffix}"):
        m = pattern.match(f.name)
        if m:
            versions.append(int(m.group(1)))
    next_v = (max(versions) + 1) if versions else 1
    shutil.copy2(out_path, HISTORY_DIR / f"{stem}__v{next_v:04d}{suffix}")

def save_delta(out_path: Path, payload: str, archive_on_change: bool = True) -> bool:
    new_norm = normalize_for_hash(payload)
    if out_path.exists():
        old_norm = normalize_for_hash(out_path.read_text(encoding="utf-8", errors="ignore"))
        if sha256_text(old_norm) == sha256_text(new_norm):
            return False
        if archive_on_change:
            archive_old(out_path)

    # atomic write prevents half-written MD on crash
    atomic_write_text(out_path, payload, encoding="utf-8")
    return True

# =========================
# PDF META
# =========================
def load_pdf_meta(pdf_path: Path) -> dict:
    """
    For pdfs named like: SITE__<hash>.pdf
    meta lives at: data/0_raw/pdfs/_meta/SITE__<hash>.json
    """
    meta_path = PDF_META_DIR / f"{pdf_path.stem}.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

# =========================
# SIGNATURE CACHE (ROBUST)
# =========================
def pipeline_fingerprint() -> str:
    # If you change any of these knobs, we WANT a re-run.
    return sha256_text(json.dumps({
        "PIPELINE_VERSION": PIPELINE_VERSION,
        "IMAGES_SCALE_TEXT": IMAGES_SCALE_TEXT,
        "IMAGES_SCALE_OCR": IMAGES_SCALE_OCR,
        "DO_TABLES_TEXT_PASS": DO_TABLES_TEXT_PASS,
        "DO_TABLES_OCR_PASS": DO_TABLES_OCR_PASS,
        "RAPIDOCR_BACKEND": RAPIDOCR_BACKEND,
        "FORCE_FULL_PAGE_OCR": FORCE_FULL_PAGE_OCR,
        "MAX_OCR_MB": MAX_OCR_MB,
        "MAX_NUM_PAGES": MAX_NUM_PAGES,
        "MAX_FILE_SIZE_BYTES": MAX_FILE_SIZE_BYTES,
    }, sort_keys=True))[:16]

PIPE_FP = pipeline_fingerprint()

def pdf_signature(pdf_path: Path) -> str:
    meta = load_pdf_meta(pdf_path)

    # Prefer content hash when available (true idempotency)
    if meta.get("sha256"):
        return f"sha256:{meta['sha256']}|pipe:{PIPE_FP}"

    # Fallback if meta missing: size+mtime (best-effort)
    st = pdf_path.stat()
    return f"size_mtime:{st.st_size}-{int(st.st_mtime)}|pipe:{PIPE_FP}"

def sig_path_for(pdf_path: Path) -> Path:
    return SIG_DIR / f"{pdf_path.stem}.sig"

def should_skip_pdf(pdf_path: Path, out_path: Path) -> bool:
    sp = sig_path_for(pdf_path)
    if not (out_path.exists() and sp.exists()):
        return False
    try:
        old_sig = sp.read_text(encoding="utf-8").strip()
    except Exception:
        return False
    return old_sig == pdf_signature(pdf_path)

def mark_pdf_done(pdf_path: Path) -> None:
    # atomic so a crash doesn't leave partial sigs
    atomic_write_text(sig_path_for(pdf_path), pdf_signature(pdf_path), encoding="utf-8")

# =========================
# QC
# =========================
def alpha_ratio(text: str) -> float:
    text = text or ""
    if not text.strip():
        return 0.0
    alpha = sum(1 for ch in text if ch.isalpha())
    total = sum(1 for ch in text if not ch.isspace())
    return (alpha / total) if total else 0.0

def keyword_hits(text: str, keywords: list[str]) -> int:
    t = (text or "").lower()
    return sum(1 for kw in keywords if kw.lower() in t)

@dataclass
class QCResult:
    alpha_ratio: float
    keyword_hits: int
    flagged: bool
    reason: str

def run_qc(extracted_text_only: str) -> QCResult:
    # IMPORTANT: QC should judge extracted content, not your header
    txt = extracted_text_only or ""
    ar = alpha_ratio(txt)
    kh = keyword_hits(txt, MEDICAL_KEYWORDS)
    reasons = []
    if len(txt.strip()) < MIN_TEXT_CHARS:
        reasons.append(f"text_chars<{MIN_TEXT_CHARS}")
    if ar < MIN_ALPHA_RATIO:
        reasons.append(f"alpha_ratio<{MIN_ALPHA_RATIO}")
    if kh < MIN_KEYWORD_HITS:
        reasons.append(f"keyword_hits<{MIN_KEYWORD_HITS}")
    return QCResult(ar, kh, bool(reasons), ";".join(reasons) if reasons else "")

def is_memory_alloc_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(x in msg for x in (
        "std::bad_alloc", "bad allocation", "out of memory",
        "onnxruntimeerror", "runtime_exception",
    ))

# =========================
# DOCLING CONVERTER
# =========================
def make_converter(do_ocr: bool, do_tables: bool, images_scale: float) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    pipeline_options.do_table_structure = do_tables

    if hasattr(pipeline_options, "images_scale"):
        pipeline_options.images_scale = images_scale

    for attr in ("batch_size", "max_pages_per_batch", "num_workers", "workers",
                 "layout_batch_size", "ocr_batch_size", "table_batch_size"):
        if hasattr(pipeline_options, attr):
            try:
                setattr(pipeline_options, attr, 1)
            except Exception:
                pass

    if do_ocr:
        ocr_options = RapidOcrOptions(
            force_full_page_ocr=FORCE_FULL_PAGE_OCR,
            backend=RAPIDOCR_BACKEND,
        )
        pipeline_options.ocr_options = ocr_options

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

# =========================
# WORKER: convert one PDF
# =========================
def convert_one_pdf(pdf_path: Path) -> int:
    md_name = f"PDF_EXTRACT_{pdf_path.stem}.md"
    out_path = OUT_DIR / md_name
    size_mb = pdf_path.stat().st_size / (1024 * 1024)

    if should_skip_pdf(pdf_path, out_path):
        print(f"SKIP\t{pdf_path.name}")
        return 0

    meta = load_pdf_meta(pdf_path)
    pdf_url = normalize_url(meta.get("url") or "")
    pdf_sha = (meta.get("sha256") or "").strip()

    converter_text = make_converter(do_ocr=False, do_tables=DO_TABLES_TEXT_PASS, images_scale=IMAGES_SCALE_TEXT)
    converter_ocr  = make_converter(do_ocr=True,  do_tables=DO_TABLES_OCR_PASS,  images_scale=IMAGES_SCALE_OCR)

    path_used = "text_no_ocr"

    try:
        # text-first
        try:
            res = converter_text.convert(pdf_path, max_num_pages=MAX_NUM_PAGES, max_file_size=MAX_FILE_SIZE_BYTES)
            md = sanitize_text(res.document.export_to_markdown())
        except Exception as e_text:
            if is_memory_alloc_error(e_text):
                md = ""
                path_used = "text_no_ocr__failed_bad_alloc"
            else:
                raise

        # QC on extracted content only
        qc = run_qc(md)

        # OCR retry if needed and safe
        if qc.flagged and size_mb <= MAX_OCR_MB and path_used != "text_no_ocr__failed_bad_alloc":
            try:
                path_used = "ocr_retry"
                res2 = converter_ocr.convert(pdf_path, max_num_pages=MAX_NUM_PAGES, max_file_size=MAX_FILE_SIZE_BYTES)
                md2 = sanitize_text(res2.document.export_to_markdown())
                qc2 = run_qc(md2)
                md, qc = md2, qc2
            except Exception as e_ocr:
                if is_memory_alloc_error(e_ocr):
                    path_used = "text_no_ocr__ocr_failed_bad_alloc"
                else:
                    raise
        elif qc.flagged and size_mb > MAX_OCR_MB:
            path_used = "text_no_ocr__ocr_skipped_size_gate"

        header_lines = [f"Source PDF: {pdf_path.name}"]
        if pdf_url:
            header_lines.append(f"Source PDF URL: {pdf_url}")
        if pdf_sha:
            header_lines.append(f"Source PDF SHA256: {pdf_sha}")
        header = "\n".join(header_lines).strip() + "\n\n"

        payload = header + (md or "")
        updated = save_delta(out_path, payload, archive_on_change=True)
        mark_pdf_done(pdf_path)

        new_file = not QC_REPORT.exists()
        with QC_REPORT.open("a", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            if new_file:
                writer.writerow([
                    "pdf_name", "md_name", "updated", "flagged", "reason",
                    "alpha_ratio", "keyword_hits", "path_used", "size_mb",
                    "meta_url_present", "meta_sha_present", "pipe_fp"
                ])
            writer.writerow([
                pdf_path.name, md_name, int(updated), int(qc.flagged), qc.reason,
                f"{qc.alpha_ratio:.4f}", qc.keyword_hits, path_used, f"{size_mb:.2f}",
                int(bool(pdf_url)), int(bool(pdf_sha)), PIPE_FP
            ])

        print(f"OK\t{pdf_path.name}\t{path_used}")
        return 0

    except Exception as e:
        with FAIL_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{pdf_path.name}\t{type(e).__name__}\t{e}\n")
        print(f"FAIL\t{pdf_path.name}\t{e}")
        return 2

# =========================
# DRIVER: subprocess per PDF
# =========================
def driver():
    pdfs = sorted([p for p in PDF_DIR.glob("*.pdf") if p.is_file()])
    print(f"PDFs found: {len(pdfs)}")
    print(f"Pipeline: {PIPELINE_VERSION}")
    print(f"Pipeline FP: {PIPE_FP}")
    print(f"OCR size gate: {MAX_OCR_MB} MB | images_scale: {IMAGES_SCALE_TEXT}")

    for i, pdf in enumerate(pdfs, start=1):
        md_name = f"PDF_EXTRACT_{pdf.stem}.md"
        out_path = OUT_DIR / md_name
        if should_skip_pdf(pdf, out_path):
            print(f"[{i}/{len(pdfs)}] SKIP {pdf.name}")
            continue

        cmd = [sys.executable, __file__, "--one", str(pdf)]
        print(f"[{i}/{len(pdfs)}] RUN  {pdf.name}")
        p = subprocess.run(cmd, capture_output=True, text=True)

        if p.stdout:
            print(p.stdout.strip())
        if p.returncode != 0 and p.stderr:
            print(p.stderr.strip())

        time.sleep(0.05)

# =========================
# ENTRYPOINT
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", type=str, help="Process exactly one PDF (worker mode)")
    args = ap.parse_args()

    if args.one:
        return convert_one_pdf(Path(args.one))
    else:
        driver()
        return 0

if __name__ == "__main__":
    raise SystemExit(main())