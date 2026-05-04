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

SINHALA_TARGET = "sin_Sinh"
TAMIL_TARGET = "tam_Taml"
SOURCE_LANG = "eng_Latn"

SOURCE_TABLE = "sl_med_corpus"
TARGET_TABLE = "sl_med_translations"

DOC_BATCH_SIZE = 64
CHUNK_MAX_CHARS = 1200
MIN_TEXT_LENGTH = 40
MAX_DOC_CHARS = 50000

TRANSLATION_BATCH_SIZE = 8
INSERT_BATCH_SIZE = 100

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
    attn_implementation = "sdpa"
).to(device)

model.eval()

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
# BATCH TRANSLATION
# =========================================================
def translate_batch_nllb(
    texts: list[str],
    forced_bos_token_id: int,
    target_lang_label: str,
    source_lang: str = SOURCE_LANG,
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
        f"target_lang={target_lang_label} | batch_size={len(cleaned)}"
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
        .eq("language", "en")
        .range(start, end)
        .execute()
    )
    return result.data or []


def fetch_existing_indexes_for_docs(doc_ids: list[str]) -> dict[str, set[int]]:
    if not doc_ids:
        return {}

    result = (
        supabase.table(TARGET_TABLE)
        .select("doc_id, chunk_index")
        .in_("doc_id", doc_ids)
        .execute()
    )

    mapping: dict[str, set[int]] = {}
    for row in (result.data or []):
        doc_id = row.get("doc_id")
        chunk_index = row.get("chunk_index")
        if doc_id is None or chunk_index is None:
            continue
        mapping.setdefault(doc_id, set()).add(chunk_index)

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
def process_document(
    row: dict,
    done_indexes: set[int] | None = None,
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

    if done_indexes is None:
        done_indexes = set()

    pending = [(idx, chunk) for idx, chunk in enumerate(chunks) if idx not in done_indexes]
    pending.sort(key=lambda x:len(x[1]))

    if not pending:
        print(f"Skipping {doc_id}: all chunks already translated")
        return

    print(f"Processing {doc_id} | total_chunks={len(chunks)} | pending={len(pending)}")

    for batch in batched(pending, TRANSLATION_BATCH_SIZE):
        batch_indexes = [idx for idx, _ in batch]
        batch_texts = [text for _, text in batch]

        try:
            print(f"Translating batch for {doc_id}: chunks {batch_indexes}")

            si_outputs = translate_batch_nllb(
                batch_texts,
                forced_bos_token_id=SINHALA_BOS_ID,
                target_lang_label=SINHALA_TARGET,
            )
            ta_outputs = translate_batch_nllb(
                batch_texts,
                forced_bos_token_id=TAMIL_BOS_ID,
                target_lang_label=TAMIL_TARGET,
            )

            now_iso = datetime.now(timezone.utc).isoformat()

            for i, chunk_index in enumerate(batch_indexes):
                pending_inserts.append({
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "source_lang": source_lang,
                    "source_text": batch_texts[i],
                    "translation_si": si_outputs[i],
                    "translation_ta": ta_outputs[i],
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

                for chunk_index, chunk_text_item in batch:
                    try:
                        si = translate_batch_nllb(
                            [chunk_text_item],
                            forced_bos_token_id=SINHALA_BOS_ID,
                            target_lang_label=SINHALA_TARGET,
                        )[0]
                        ta = translate_batch_nllb(
                            [chunk_text_item],
                            forced_bos_token_id=TAMIL_BOS_ID,
                            target_lang_label=TAMIL_TARGET,
                        )[0]

                        pending_inserts.append({
                            "doc_id": doc_id,
                            "chunk_index": chunk_index,
                            "source_lang": source_lang,
                            "source_text": chunk_text_item,
                            "translation_si": si,
                            "translation_ta": ta,
                            "status": "done",
                            "error_message": None,
                            "translated_at": datetime.now(timezone.utc).isoformat(),
                        })

                        if len(pending_inserts) >= INSERT_BATCH_SIZE:
                            flush_pending_inserts(pending_inserts)

                        print(f"Prepared single fallback chunk {doc_id}::{chunk_index}")

                    except Exception as inner_e:
                        pending_inserts.append({
                            "doc_id": doc_id,
                            "chunk_index": chunk_index,
                            "source_lang": source_lang,
                            "source_text": chunk_text_item,
                            "translation_si": None,
                            "translation_ta": None,
                            "status": "failed",
                            "error_message": str(inner_e)[:1000],
                            "translated_at": None,
                        })

                        if len(pending_inserts) >= INSERT_BATCH_SIZE:
                            flush_pending_inserts(pending_inserts)

                        print(f"Failed fallback chunk {doc_id}::{chunk_index}: {inner_e}")
            else:
                print(f"Runtime error on batch {batch_indexes}: {e}")
                for chunk_index, chunk_text_item in batch:
                    pending_inserts.append({
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "source_lang": source_lang,
                        "source_text": chunk_text_item,
                        "translation_si": None,
                        "translation_ta": None,
                        "status": "failed",
                        "error_message": str(e)[:1000],
                        "translated_at": None,
                    })

                if len(pending_inserts) >= INSERT_BATCH_SIZE:
                    flush_pending_inserts(pending_inserts)

        except Exception as e:
            print(f"Failed batch {batch_indexes} for {doc_id}: {e}")
            for chunk_index, chunk_text_item in batch:
                pending_inserts.append({
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "source_lang": source_lang,
                    "source_text": chunk_text_item,
                    "translation_si": None,
                    "translation_ta": None,
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
        existing_map = fetch_existing_indexes_for_docs(doc_ids)

        for i, row in enumerate(rows, start=1):
            processed_doc_count += 1
            doc_id = row.get("doc_id")
            print(
                f"\n=== DOC #{processed_doc_count} | "
                f"batch item {i}/{len(rows)} | doc_id={doc_id} ==="
            )
            process_document(
                row,
                done_indexes=existing_map.get(doc_id, set()),
                pending_inserts=pending_inserts,
            )

        flush_pending_inserts(pending_inserts)
        start += DOC_BATCH_SIZE


if __name__ == "__main__":

    main()