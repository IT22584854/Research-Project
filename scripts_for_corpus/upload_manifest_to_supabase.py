import os
import json
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from supabase import create_client, Client

# =========================
# CONFIG
# =========================
MANIFEST_PATH = Path("data/2_processed/manifest.jsonl")
TABLE_NAME = "sl_med_corpus"
BATCH_SIZE = 200
FAIL_FAST = False

# Keep ONLY columns that actually exist in Supabase
ALLOWED_COLUMNS = {
    "doc_id",
    "source_key",
    "source_type",
    "source_url",
    "source_pdf",
    "site",
    "site_root_url",
    "raw_path",
    "clean_path",
    "refined_path",
    "clean_rel",
    "retrieved_at",
    "bytes",
    "title",
    "keywords",
    "language",
    "language_primary",
    "language_top",
    "language_profile",
    "markdown",
}

# If you add these columns to Supabase later, you can include them too:
# ALLOWED_COLUMNS.add("source_pdf_url")


# =========================
# ENV LOADING
# =========================
def load_env_from_project_root() -> None:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)

    if not os.getenv("SUPABASE_URL") and (Path.cwd() / ".env").exists():
        load_dotenv(dotenv_path=Path.cwd() / ".env")


def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url:
        raise RuntimeError(
            "SUPABASE_URL is missing. Put it in your .env (project root) as:\n"
            'SUPABASE_URL="https://xxxx.supabase.co"\n'
        )
    if not key:
        raise RuntimeError(
            "SUPABASE_SERVICE_ROLE_KEY is missing. Put it in your .env as:\n"
            'SUPABASE_SERVICE_ROLE_KEY="your_service_role_key"\n'
        )

    return create_client(url, key)


# =========================
# MANIFEST LOADING
# =========================
def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON on line {line_no}: {e}") from e


def chunked(items: List[Dict[str, Any]], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# =========================
# CLEANUP / COLUMN FILTER
# =========================
def normalize_keywords(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]
    elif not isinstance(value, list):
        raise ValueError(f"Invalid keywords type: {type(value).__name__}")

    cleaned = []
    seen = set()

    for kw in value:
        if kw is None:
            continue
        kw = str(kw).strip()
        if not kw:
            continue
        if kw not in seen:
            cleaned.append(kw)
            seen.add(kw)

    return cleaned


def prepare_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if not row.get("doc_id"):
        raise ValueError("Row missing doc_id")

    cleaned_row = {}

    for key in ALLOWED_COLUMNS:
        if key in row:
            cleaned_row[key] = row[key]

    cleaned_row["keywords"] = normalize_keywords(row.get("keywords", []))

    return cleaned_row


# =========================
# UPLOAD
# =========================
def upload_manifest(
    supabase: Client,
    manifest_path: Path,
    table_name: str,
    batch_size: int = 200,
):
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: List[Dict[str, Any]] = []
    skipped = 0

    for line_no, row in iter_jsonl(manifest_path):
        try:
            rows.append(prepare_row(row))
        except Exception as e:
            skipped += 1
            msg = f"Skipping bad row at line {line_no}: {e}"
            if FAIL_FAST:
                raise RuntimeError(msg) from e
            print("[WARN]", msg)

    if not rows:
        print("[WARN] No rows found to upload.")
        return

    print(f"[INFO] Loaded {len(rows)} records from {manifest_path}")
    print(f"[INFO] Uploading to Supabase table: {table_name}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Skipped during cleanup: {skipped}")
    print(f"[INFO] Upload columns: {sorted(ALLOWED_COLUMNS)}")

    total = len(rows)
    uploaded = 0
    failed = 0

    for batch_idx, batch in enumerate(chunked(rows, batch_size), start=1):
        try:
            resp = supabase.table(table_name).upsert(batch, on_conflict="doc_id").execute()

            if hasattr(resp, "error") and resp.error:
                raise RuntimeError(resp.error)

            uploaded += len(batch)
            print(f"[OK] Batch {batch_idx}: uploaded {len(batch)} ({uploaded}/{total})")

        except Exception as e:
            failed += len(batch)
            print(f"[ERROR] Batch {batch_idx} failed ({len(batch)} rows): {e}")

            if FAIL_FAST:
                raise

    print("\n====================")
    print("UPLOAD COMPLETE")
    print("====================")
    print(f"Total rows loaded : {total}")
    print(f"Uploaded          : {uploaded}")
    print(f"Failed            : {failed}")
    print(f"Skipped cleanup   : {skipped}")


# =========================
# MAIN
# =========================
def main():
    load_env_from_project_root()
    supabase = get_supabase_client()

    print("SUPABASE_URL loaded:", bool(os.getenv("SUPABASE_URL")))
    print("SERVICE KEY loaded :", bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY")))

    upload_manifest(
        supabase=supabase,
        manifest_path=MANIFEST_PATH,
        table_name=TABLE_NAME,
        batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    main()