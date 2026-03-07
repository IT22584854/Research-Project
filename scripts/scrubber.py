import re
import shutil
import hashlib
from pathlib import Path

# =========================
# 1) PATHS
# =========================
BASE_DIR = Path("data")
RAW_ROOT = BASE_DIR / "0_raw"          # scrub ALL markdowns under here
CLEAN_ROOT = BASE_DIR / "1_cleaned"    # write cleaned files under here (mirrors structure)
HISTORY_ROOT = CLEAN_ROOT / "history"

def ensure_dirs():
    CLEAN_ROOT.mkdir(parents=True, exist_ok=True)
    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# 2) TEXT SANITIZATION
# =========================
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def sanitize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return CONTROL_CHARS_RE.sub("", s)

# =========================
# 3) HASHING / ARCHIVE HELPERS
# =========================
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def archive_old_clean_file(clean_path: Path, rel_path: Path) -> None:
    """Archive old cleaned file into data/1_cleaned/history/<rel_dir>/ with incremented suffix."""
    if not clean_path.exists():
        return

    hist_dir = HISTORY_ROOT / rel_path.parent
    hist_dir.mkdir(parents=True, exist_ok=True)

    stem = clean_path.stem
    suffix = clean_path.suffix

    pattern = re.compile(rf"^{re.escape(stem)}__v(\d{{4}}){re.escape(suffix)}$")
    versions = []
    for f in hist_dir.glob(f"{stem}__v*{suffix}"):
        m = pattern.match(f.name)
        if m:
            versions.append(int(m.group(1)))

    next_v = (max(versions) + 1) if versions else 1
    archived_path = hist_dir / f"{stem}__v{next_v:04d}{suffix}"
    shutil.copy2(clean_path, archived_path)

# =========================
# 4) PII PATTERNS
# =========================
NIC_RE = re.compile(r"\b\d{9}[vVxX]\b|\b\d{12}\b")
SLMC_RE = re.compile(r"\bSLMC/\d{4,6}\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+94\s?|0)(?:\d[\s-]?){8,10}\b")
URL_RE = re.compile(r"\bhttps?://[^\s)>\]]+\b", re.IGNORECASE)

# =========================
# 5) SCRUB FUNCTION
# =========================
def scrub_pii(text: str, redact_urls: bool = False):
    before = text or ""
    text = sanitize_text(before)

    counts = {"NIC": 0, "SLMC": 0, "EMAIL": 0, "PHONE": 0, "URL": 0, "CTRLSTRIP": 0}
    counts["CTRLSTRIP"] = max(0, len(before) - len(text))

    def _sub_count(pattern: re.Pattern, repl: str, label: str, s: str):
        new_s, n = pattern.subn(repl, s)
        counts[label] += n
        return new_s

    text = _sub_count(NIC_RE,   "[REDACTED_NIC]",   "NIC",   text)
    text = _sub_count(SLMC_RE,  "[REDACTED_SLMC]",  "SLMC",  text)
    text = _sub_count(EMAIL_RE, "[REDACTED_EMAIL]", "EMAIL", text)
    text = _sub_count(PHONE_RE, "[REDACTED_PHONE]", "PHONE", text)

    if redact_urls:
        text = _sub_count(URL_RE, "[REDACTED_URL]", "URL", text)

    return text, counts

# =========================
# 6) PROCESS ALL MARKDOWNS (RECURSIVE + MIRROR STRUCTURE)
# =========================
def process_all_markdowns(redact_urls: bool = False):
    ensure_dirs()

    totals = {"NIC": 0, "SLMC": 0, "EMAIL": 0, "PHONE": 0, "URL": 0, "CTRLSTRIP": 0}
    scanned = 0
    updated = 0

    for raw_path in sorted(RAW_ROOT.rglob("*.md")):
        scanned += 1

        rel_path = raw_path.relative_to(RAW_ROOT)  # keeps subfolders (e.g., pdf_md/xxx.md)
        clean_path = CLEAN_ROOT / rel_path
        clean_path.parent.mkdir(parents=True, exist_ok=True)

        content = raw_path.read_text(encoding="utf-8", errors="ignore")
        cleaned, counts = scrub_pii(content, redact_urls=redact_urls)

        old_clean = clean_path.read_text(encoding="utf-8", errors="ignore") if clean_path.exists() else ""
        if sha256_text(old_clean) == sha256_text(cleaned):
            continue

        if clean_path.exists():
            archive_old_clean_file(clean_path, rel_path)

        clean_path.write_text(cleaned, encoding="utf-8")
        updated += 1

        for k in totals:
            totals[k] += counts.get(k, 0)

        print(f"✅ [CLEANED] {rel_path.as_posix()} "
              f"(NIC:{counts['NIC']} SLMC:{counts['SLMC']} EMAIL:{counts['EMAIL']} "
              f"PHONE:{counts['PHONE']} URL:{counts['URL']})")

    print("\n=========================")
    print("PII Scrubbing complete ✅")
    print("=========================")
    print(f"Files scanned : {scanned}")
    print(f"Files updated : {updated}")
    print("\n--- Redaction totals (only updated files) ---")
    for k, v in totals.items():
        print(f"{k:10s}: {v}")

if __name__ == "__main__":
    process_all_markdowns(redact_urls=False)