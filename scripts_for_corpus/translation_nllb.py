import os
import re
import sys
from datetime import datetime, timezone

import torch
from dotenv import load_dotenv
from supabase import create_client
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================================================
# CONFIG
# =========================================================
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"

ENGLISH_TARGET = "eng_Latn"
SINHALA_TARGET = "sin_Sinh"
TAMIL_TARGET = "tam_Taml"

SOURCE_TABLE = "sl_med_corpus"
TARGET_TABLE = "sl_med_translations"

DOC_BATCH_SIZE = 64
CHUNK_MAX_CHARS = 1200
MIN_TEXT_LENGTH = 40
MAX_DOC_CHARS = 50000

TRANSLATION_BATCH_SIZE = 8
INSERT_BATCH_SIZE = 100

SUPPORTED_SOURCE_LANGS = ["si", "ta", "mixed"]

# =========================================================
# ENV + SUPABASE
# =========================================================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================================================
# PYTHON / TORCH DEBUG
# =========================================================
print("=" * 60)
print("PYTHON / TORCH DEBUG")
print("PYTHON EXE:", sys.executable)
print("TORCH VERSION:", torch.__version__)
print("TORCH CUDA VERSION:", torch.version.cuda)
print("CUDA AVAILABLE:", torch.cuda.is_available())
print("=" * 60)

# =========================================================
# MODEL LOAD
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
    use_safetensors=False,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    attn_implementation="sdpa" if device == "cuda" else "eager",
).to(device)

model.eval()

ENGLISH_BOS_ID = tokenizer.convert_tokens_to_ids(ENGLISH_TARGET)
SINHALA_BOS_ID = tokenizer.convert_tokens_to_ids(SINHALA_TARGET)
TAMIL_BOS_ID = tokenizer.convert_tokens_to_ids(TAMIL_TARGET)

print("=" * 60)
print("STARTUP CUDA DEBUG")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {device}")
print(f"Model parameter device: {next(model.parameters()).device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Allocated MB after model load: {torch.cuda.memory_allocated(0) / 1024**2:.2f}")
    print(f"Reserved MB after model load: {torch.cuda.memory_reserved(0) / 1024**2:.2f}")
print("=" * 60)

# =========================================================
# SCRIPT DETECTION
# =========================================================
SINHALA_RE = re.compile(r"[\u0D80-\u0DFF]")
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
LATIN_RE = re.compile(r"[A-Za-z]")

def detect_text_type(text: str) -> str:
    if not text:
        return "unknown"

    has_si = bool(SINHALA_RE.search(text))
    has_ta = bool(TAMIL_RE.search(text))
    has_lat = bool(LATIN_RE.search(text))

    if has_si and not has_ta and not has_lat:
        return "si"
    if has_ta and not has_si and not has_lat:
        return "ta"
    if has_lat and not has_si and not has_ta:
        return "en"
    if has_si and has_lat and not has_ta:
        return "mixed_si"
    if has_ta and has_lat and not has_si:
        return "mixed_ta"
    if has_si and has_ta:
        return "mixed_multi"

    return "unknown"


def choose_source_token(source_lang: str, text: str) -> str:
    """
    Pick the NLLB source token for this chunk.
    """
    source_lang = (source_lang or "").strip().lower()
    detected = detect_text_type(text)

    if source_lang == "si":
        return SINHALA_TARGET

    if source_lang == "ta":
        return TAMIL_TARGET

    if source_lang == "mixed":
        if detected in {"si", "mixed_si"}:
            return SINHALA_TARGET
        if detected in {"ta", "mixed_ta"}:
            return TAMIL_TARGET
        return ENGLISH_TARGET

    return ENGLISH_TARGET


# =========================================================
# BATCH TRANSLATION
# =========================================================
def translate_batch_nllb(
    texts: list[str],
    forced_bos_token_id: int,
    target_lang_label: str,
    source_lang: str,
    input_max_length: int = 384,
    output_max_new_tokens: int = 192,
) -> list[str]:
    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return []

    tokenizer.src_lang = source_lang

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=input_max_length,
    ).to(device)

    print("-" * 60)
    print(
        f"translate_batch_nllb called | "
        f"source_lang={source_lang} | "
        f"target_lang={target_lang_label} | "
        f"batch_size={len(cleaned)}"
    )
    print(f"Input device: {inputs['input_ids'].device}")
    print(f"Model parameter device: {next(model.parameters()).device}")

    if torch.cuda.is_available():
        print(f"Allocated MB before generate: {torch.cuda.memory_allocated(0) / 1024**2:.2f}")
        print(f"Reserved MB before generate: {torch.cuda.memory_reserved(0) / 1024**2:.2f}")

    with torch.inference_mode():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=output_max_new_tokens,
            num_beams=1,
        )

    print(f"Output tensor device: {translated_tokens.device}")

    if torch.cuda.is_available():
        print(f"Allocated MB after generate: {torch.cuda.memory_allocated(0) / 1024**2:.2f}")
        print(f"Reserved MB after generate: {torch.cuda.memory_reserved(0) / 1024**2:.2f}")

    outputs = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return [o.strip() for o in outputs]


def safe_translate_batch(
    batch_rows: list[dict],
    forced_bos_token_id: int,
    target_lang_label: str,
):
    out = {}

    if not batch_rows:
        return out

    texts = [r["source_text"] for r in batch_rows]
    source_lang = batch_rows[0]["_source_token"]

    try:
        outputs = translate_batch_nllb(
            texts=texts,
            forced_bos_token_id=forced_bos_token_id,
            target_lang_label=target_lang_label,
            source_lang=source_lang,
        )
        for row, translated in zip(batch_rows, outputs):
            out[(row["doc_id"], row["chunk_index"])] = translated
        return out

    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise

        print(f"OOM on {target_lang_label} batch. Falling back to single-chunk mode.")
        if device == "cuda":
            torch.cuda.empty_cache()

        for row in batch_rows:
            try:
                translated = translate_batch_nllb(
                    [row["source_text"]],
                    forced_bos_token_id=forced_bos_token_id,
                    target_lang_label=target_lang_label,
                    source_lang=row["_source_token"],
                )[0]
                out[(row["doc_id"], row["chunk_index"])] = translated
            except Exception as inner_e:
                print(f"Failed fallback chunk {row['doc_id']}::{row['chunk_index']}: {inner_e}")
                out[(row["doc_id"], row["chunk_index"])] = None

        return out


def batched(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

# =========================================================
# TEXT CLEANING
# =========================================================
def lightly_normalize_markdown(md: str) -> str:
    if not md:
        return ""

    text = md
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"^[ \t]*#{1,6}[ \t]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*>[ \t]?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*[-*+][ \t]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# =========================================================
# CHUNKING
# =========================================================
def split_long_text(text: str, max_chars: int):
    text = text.strip()
    n = len(text)
    start = 0

    while start < n:
        end = min(start + max_chars, n)

        if end < n:
            split_at = text.rfind(". ", start, end)
            if split_at == -1:
                split_at = text.rfind(" ", start, end)
            if split_at == -1 or split_at <= start:
                split_at = end
        else:
            split_at = end

        chunk = text[start:split_at].strip()
        if chunk:
            yield chunk

        start = split_at


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS):
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""

            for piece in split_long_text(para, max_chars):
                chunks.append(piece)
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = para

    if current:
        chunks.append(current.strip())

    return chunks

# =========================================================
# SUPABASE HELPERS
# =========================================================
def fetch_rows(start: int, end: int):
    result = (
        supabase.table(SOURCE_TABLE)
        .select("doc_id, markdown, language")
        .in_("language", SUPPORTED_SOURCE_LANGS)
        .range(start, end)
        .execute()
    )
    return result.data or []


def fetch_existing_rows_for_docs(doc_ids: list[str]) -> dict[str, dict[int, dict]]:
    if not doc_ids:
        return {}

    result = (
        supabase.table(TARGET_TABLE)
        .select("doc_id, chunk_index, translation_en, translation_si, translation_ta, status")
        .in_("doc_id", doc_ids)
        .execute()
    )

    mapping: dict[str, dict[int, dict]] = {}
    for row in (result.data or []):
        doc_id = row.get("doc_id")
        chunk_index = row.get("chunk_index")
        if doc_id is None or chunk_index is None:
            continue
        mapping.setdefault(doc_id, {})[chunk_index] = row

    return mapping


def insert_translations(records: list[dict]):
    if not records:
        return

    supabase.table(TARGET_TABLE).upsert(
        records,
        on_conflict="doc_id,chunk_index"
    ).execute()


def flush_pending_inserts(pending_inserts: list[dict]):
    if not pending_inserts:
        return

    batches = list(batched(pending_inserts, INSERT_BATCH_SIZE))

    for i, batch in enumerate(batches, start=1):
        try:
            insert_translations(batch)
            print(f"Flushed insert batch {i}/{len(batches)} | rows={len(batch)}")
        except Exception as e:
            print(f"Failed insert batch {i}/{len(batches)}: {e}")

    pending_inserts.clear()

# =========================================================
# MAIN PIPELINE
# =========================================================
def chunk_needs_processing(existing_row: dict | None) -> bool:
    if not existing_row:
        return True

    en_ok = existing_row.get("translation_en") is not None and str(existing_row.get("translation_en")).strip() != ""
    si_ok = existing_row.get("translation_si") is not None and str(existing_row.get("translation_si")).strip() != ""
    ta_ok = existing_row.get("translation_ta") is not None and str(existing_row.get("translation_ta")).strip() != ""

    return not (en_ok and si_ok and ta_ok)


def process_document(
    row: dict,
    existing_chunks: dict[int, dict] | None = None,
    pending_inserts: list[dict] | None = None,
):
    if pending_inserts is None:
        pending_inserts = []

    doc_id = row["doc_id"]
    markdown = row.get("markdown", "")
    source_lang = row.get("language", "unknown")

    cleaned = lightly_normalize_markdown(markdown)

    if len(cleaned) < MIN_TEXT_LENGTH:
        print(f"Skipping {doc_id}: too short after cleaning")
        return

    if len(cleaned) > MAX_DOC_CHARS:
        print(f"Skipping {doc_id}: document too large after cleaning ({len(cleaned)} chars)")
        error_record = {
            "doc_id": doc_id,
            "chunk_index": -1,
            "source_lang": source_lang,
            "source_text": None,
            "translation_en": None,
            "translation_si": None,
            "translation_ta": None,
            "status": "failed",
            "error_message": f"Document too large after cleaning ({len(cleaned)} chars)",
            "translated_at": None,
        }
        pending_inserts.append(error_record)
        if len(pending_inserts) >= INSERT_BATCH_SIZE:
            flush_pending_inserts(pending_inserts)
        return

    try:
        chunks = chunk_text(cleaned, CHUNK_MAX_CHARS)
    except MemoryError:
        print(f"Skipping {doc_id}: MemoryError during chunking")
        error_record = {
            "doc_id": doc_id,
            "chunk_index": -1,
            "source_lang": source_lang,
            "source_text": None,
            "translation_en": None,
            "translation_si": None,
            "translation_ta": None,
            "status": "failed",
            "error_message": "MemoryError during chunking",
            "translated_at": None,
        }
        pending_inserts.append(error_record)
        if len(pending_inserts) >= INSERT_BATCH_SIZE:
            flush_pending_inserts(pending_inserts)
        return

    if not chunks:
        print(f"Skipping {doc_id}: no usable chunks")
        return

    if existing_chunks is None:
        existing_chunks = {}

    pending = []
    for idx, chunk in enumerate(chunks):
        existing_row = existing_chunks.get(idx)
        if chunk_needs_processing(existing_row):
            pending.append((idx, chunk))

    pending.sort(key=lambda x: len(x[1]))

    if not pending:
        print(f"Skipping {doc_id}: all chunks already have all three translations")
        return

    print(f"Processing {doc_id} | total_chunks={len(chunks)} | pending={len(pending)} | source_lang={source_lang}")

    prepared_rows = []
    for idx, chunk_text_item in pending:
        prepared_rows.append({
            "doc_id": doc_id,
            "chunk_index": idx,
            "source_lang": source_lang,
            "source_text": chunk_text_item,
            "_source_token": choose_source_token(source_lang, chunk_text_item),
        })

    token_groups = {}
    for r in prepared_rows:
        token_groups.setdefault(r["_source_token"], []).append(r)

    for source_token, rows_for_token in token_groups.items():
        print(f"Source token group {source_token} | chunks={len(rows_for_token)}")

        for batch in batched(rows_for_token, TRANSLATION_BATCH_SIZE):
            batch_indexes = [r["chunk_index"] for r in batch]
            print(f"Translating batch for {doc_id}: chunks {batch_indexes}")

            try:
                en_map = {}
                si_map = {}
                ta_map = {}

                # EN
                need_en_rows = []
                for r in batch:
                    existing_row = existing_chunks.get(r["chunk_index"])
                    if existing_row and existing_row.get("translation_en") not in (None, ""):
                        en_map[(r["doc_id"], r["chunk_index"])] = existing_row["translation_en"]
                    else:
                        need_en_rows.append(r)

                if need_en_rows:
                    en_map.update(
                        safe_translate_batch(
                            need_en_rows,
                            forced_bos_token_id=ENGLISH_BOS_ID,
                            target_lang_label=ENGLISH_TARGET,
                        )
                    )

                # SI
                need_si_rows = []
                for r in batch:
                    existing_row = existing_chunks.get(r["chunk_index"])
                    if existing_row and existing_row.get("translation_si") not in (None, ""):
                        si_map[(r["doc_id"], r["chunk_index"])] = existing_row["translation_si"]
                    elif r["source_lang"] == "si":
                        si_map[(r["doc_id"], r["chunk_index"])] = r["source_text"]
                    else:
                        need_si_rows.append(r)

                if need_si_rows:
                    si_map.update(
                        safe_translate_batch(
                            need_si_rows,
                            forced_bos_token_id=SINHALA_BOS_ID,
                            target_lang_label=SINHALA_TARGET,
                        )
                    )

                # TA
                need_ta_rows = []
                for r in batch:
                    existing_row = existing_chunks.get(r["chunk_index"])
                    if existing_row and existing_row.get("translation_ta") not in (None, ""):
                        ta_map[(r["doc_id"], r["chunk_index"])] = existing_row["translation_ta"]
                    elif r["source_lang"] == "ta":
                        ta_map[(r["doc_id"], r["chunk_index"])] = r["source_text"]
                    else:
                        need_ta_rows.append(r)

                if need_ta_rows:
                    ta_map.update(
                        safe_translate_batch(
                            need_ta_rows,
                            forced_bos_token_id=TAMIL_BOS_ID,
                            target_lang_label=TAMIL_TARGET,
                        )
                    )

                now_iso = datetime.now(timezone.utc).isoformat()

                for r in batch:
                    key = (r["doc_id"], r["chunk_index"])
                    pending_inserts.append({
                        "doc_id": r["doc_id"],
                        "chunk_index": r["chunk_index"],
                        "source_lang": r["source_lang"],
                        "source_text": r["source_text"],
                        "translation_en": en_map.get(key),
                        "translation_si": si_map.get(key),
                        "translation_ta": ta_map.get(key),
                        "status": "done",
                        "error_message": None,
                        "translated_at": now_iso,
                    })

                print(f"Prepared batch for insert: {doc_id} chunks {batch_indexes}")

                if len(pending_inserts) >= INSERT_BATCH_SIZE:
                    flush_pending_inserts(pending_inserts)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM on batch {batch_indexes} for {doc_id}. Falling back to single-chunk mode.")
                    if device == "cuda":
                        torch.cuda.empty_cache()

                    for r in batch:
                        try:
                            key = (r["doc_id"], r["chunk_index"])

                            existing_row = existing_chunks.get(r["chunk_index"])

                            if existing_row and existing_row.get("translation_en") not in (None, ""):
                                en_val = existing_row["translation_en"]
                            else:
                                en_val = translate_batch_nllb(
                                    [r["source_text"]],
                                    forced_bos_token_id=ENGLISH_BOS_ID,
                                    target_lang_label=ENGLISH_TARGET,
                                    source_lang=r["_source_token"],
                                )[0]

                            if existing_row and existing_row.get("translation_si") not in (None, ""):
                                si_val = existing_row["translation_si"]
                            elif r["source_lang"] == "si":
                                si_val = r["source_text"]
                            else:
                                si_val = translate_batch_nllb(
                                    [r["source_text"]],
                                    forced_bos_token_id=SINHALA_BOS_ID,
                                    target_lang_label=SINHALA_TARGET,
                                    source_lang=r["_source_token"],
                                )[0]

                            if existing_row and existing_row.get("translation_ta") not in (None, ""):
                                ta_val = existing_row["translation_ta"]
                            elif r["source_lang"] == "ta":
                                ta_val = r["source_text"]
                            else:
                                ta_val = translate_batch_nllb(
                                    [r["source_text"]],
                                    forced_bos_token_id=TAMIL_BOS_ID,
                                    target_lang_label=TAMIL_TARGET,
                                    source_lang=r["_source_token"],
                                )[0]

                            pending_inserts.append({
                                "doc_id": r["doc_id"],
                                "chunk_index": r["chunk_index"],
                                "source_lang": r["source_lang"],
                                "source_text": r["source_text"],
                                "translation_en": en_val,
                                "translation_si": si_val,
                                "translation_ta": ta_val,
                                "status": "done",
                                "error_message": None,
                                "translated_at": datetime.now(timezone.utc).isoformat(),
                            })

                            if len(pending_inserts) >= INSERT_BATCH_SIZE:
                                flush_pending_inserts(pending_inserts)

                            print(f"Prepared single fallback chunk {doc_id}::{r['chunk_index']}")

                        except Exception as inner_e:
                            pending_inserts.append({
                                "doc_id": r["doc_id"],
                                "chunk_index": r["chunk_index"],
                                "source_lang": r["source_lang"],
                                "source_text": r["source_text"],
                                "translation_en": None,
                                "translation_si": r["source_text"] if r["source_lang"] == "si" else None,
                                "translation_ta": r["source_text"] if r["source_lang"] == "ta" else None,
                                "status": "failed",
                                "error_message": str(inner_e)[:1000],
                                "translated_at": None,
                            })

                            if len(pending_inserts) >= INSERT_BATCH_SIZE:
                                flush_pending_inserts(pending_inserts)

                            print(f"Failed fallback chunk {doc_id}::{r['chunk_index']}: {inner_e}")
                else:
                    print(f"Runtime error on batch {batch_indexes}: {e}")
                    for r in batch:
                        pending_inserts.append({
                            "doc_id": r["doc_id"],
                            "chunk_index": r["chunk_index"],
                            "source_lang": r["source_lang"],
                            "source_text": r["source_text"],
                            "translation_en": None,
                            "translation_si": r["source_text"] if r["source_lang"] == "si" else None,
                            "translation_ta": r["source_text"] if r["source_lang"] == "ta" else None,
                            "status": "failed",
                            "error_message": str(e)[:1000],
                            "translated_at": None,
                        })

                    if len(pending_inserts) >= INSERT_BATCH_SIZE:
                        flush_pending_inserts(pending_inserts)

            except Exception as e:
                print(f"Failed batch {batch_indexes} for {doc_id}: {e}")
                for r in batch:
                    pending_inserts.append({
                        "doc_id": r["doc_id"],
                        "chunk_index": r["chunk_index"],
                        "source_lang": r["source_lang"],
                        "source_text": r["source_text"],
                        "translation_en": None,
                        "translation_si": r["source_text"] if r["source_lang"] == "si" else None,
                        "translation_ta": r["source_text"] if r["source_lang"] == "ta" else None,
                        "status": "failed",
                        "error_message": str(e)[:1000],
                        "translated_at": None,
                    })

                if len(pending_inserts) >= INSERT_BATCH_SIZE:
                    flush_pending_inserts(pending_inserts)


def main():
    start = 0
    processed_doc_count = 0
    pending_inserts: list[dict] = []

    while True:
        end = start + DOC_BATCH_SIZE - 1
        rows = fetch_rows(start, end)

        if not rows:
            flush_pending_inserts(pending_inserts)
            print(f"\nDone. Total documents seen in this run: {processed_doc_count}")
            break

        print(f"\nFetched rows {start} to {end} | count={len(rows)}")

        doc_ids = [row["doc_id"] for row in rows if row.get("doc_id")]
        existing_map = fetch_existing_rows_for_docs(doc_ids)

        for i, row in enumerate(rows, start=1):
            processed_doc_count += 1
            doc_id = row.get("doc_id")
            print(
                f"\n=== DOC #{processed_doc_count} | "
                f"batch item {i}/{len(rows)} | doc_id={doc_id} ==="
            )
            process_document(
                row,
                existing_chunks=existing_map.get(doc_id, {}),
                pending_inserts=pending_inserts,
            )

        flush_pending_inserts(pending_inserts)
        start += DOC_BATCH_SIZE


if __name__ == "__main__":
    main()
