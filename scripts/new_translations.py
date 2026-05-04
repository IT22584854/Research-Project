import os
import re
import gc
import html
from typing import List, Dict

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = r"D:\SL_Medical_Corpus\scripts\translation_input_clean.csv"
OUTPUT_CSV = r"D:\SL_Medical_Corpus\scripts\translation_nllb_1p3b.csv"

DOC_ID_COL = "doc_id"
TEXT_COL = "clean_text"

# official distilled 1.3B NLLB model
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"

SRC_LANG = "eng_Latn"
TARGETS: Dict[str, str] = {
    "translation_si": "sin_Sinh",
    "translation_ta": "tam_Taml",
}

# chunking / generation
CHUNK_SENT_COUNT = 3
MAX_INPUT_LENGTH = 384
MAX_NEW_TOKENS = 512
BATCH_SIZE = 4
NUM_BEAMS = 4

# saving
CHECKPOINT_EVERY_DOCS = 5

# loading
USE_4BIT = True

# optional filter
ONLY_ENGLISH = True  # uses language_primary == 'en' if that column exists


# =========================================================
# GENERAL HELPERS
# =========================================================
def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clean_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter for English medical/public-health text.
    Keeps headings/short title-like lines intact.
    """
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    units = []

    for para in paragraphs:
        # Keep short title-like lines as their own unit
        if len(para.split()) <= 12 and "\n" not in para:
            units.append(para)
            continue

        # Split on sentence endings followed by likely sentence starts
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", para)
        parts = [p.strip() for p in parts if p.strip()]

        if parts:
            units.extend(parts)
        else:
            units.append(para)

    return units


def chunk_sentences(sentences: List[str], chunk_size: int = 3) -> List[str]:
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def batched(items: List[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# =========================================================
# MODEL LOADING
# =========================================================
print("=" * 60)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)

model = None

if USE_4BIT and torch.cuda.is_available():
    try:
        print("Trying 4-bit quantized load...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
        )
        print("Loaded model in 4-bit.")
    except Exception as e:
        print("4-bit load failed:", repr(e))
        model = None

if model is None:
    print("Falling back to standard load...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")

model.eval()

try:
    print("Model device map:", getattr(model, "hf_device_map", "single-device model"))
except Exception:
    pass


# =========================================================
# TRANSLATION FUNCTIONS
# =========================================================
@torch.inference_mode()
def translate_chunks(chunks: List[str], target_lang: str) -> List[str]:
    translated = []
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

    for batch in batched(chunks, BATCH_SIZE):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )

        # only move tensors manually if model is not sharded with device_map
        if not hasattr(model, "hf_device_map"):
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

        generated = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            do_sample=False,
        )

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )
        translated.extend([x.strip() for x in decoded])
        free_memory()

    return translated


def translate_document(text: str, target_lang: str) -> str:
    text = clean_text(text)
    if not text:
        return ""

    sentences = split_sentences(text)
    if not sentences:
        return ""

    chunks = chunk_sentences(sentences, chunk_size=CHUNK_SENT_COUNT)
    translated_chunks = translate_chunks(chunks, target_lang)

    # keep paragraph-ish spacing between chunks
    return "\n\n".join([c for c in translated_chunks if c.strip()]).strip()


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_CSV)

if TEXT_COL not in df.columns:
    raise ValueError(f"Missing required column: {TEXT_COL}")

if DOC_ID_COL not in df.columns:
    raise ValueError(f"Missing required column: {DOC_ID_COL}")

df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

if ONLY_ENGLISH and "language_primary" in df.columns:
    df = df[df["language_primary"].fillna("").str.lower() == "en"].copy()

df = df[df[TEXT_COL].str.strip() != ""].copy()
df.reset_index(drop=True, inplace=True)

print("Docs loaded for translation:", len(df))

for out_col in TARGETS.keys():
    if out_col not in df.columns:
        df[out_col] = ""


# =========================================================
# RESUME SUPPORT
# =========================================================
existing_rows = []
done_ids = set()

if os.path.exists(OUTPUT_CSV):
    print(f"Existing output found: {OUTPUT_CSV}")
    existing_df = pd.read_csv(OUTPUT_CSV)

    if DOC_ID_COL in existing_df.columns:
        done_ids = set(existing_df[DOC_ID_COL].astype(str))
        existing_rows = existing_df.to_dict("records")
        print("Already translated docs:", len(done_ids))

df_remaining = df[~df[DOC_ID_COL].astype(str).isin(done_ids)].copy()
df_remaining.reset_index(drop=True, inplace=True)

print("Remaining docs to translate:", len(df_remaining))


# =========================================================
# RUN TRANSLATION
# =========================================================
all_rows = existing_rows.copy()

for i, row in df_remaining.iterrows():
    doc_id = str(row[DOC_ID_COL])
    source_text = row[TEXT_COL]

    print("-" * 60)
    print(f"[{i+1}/{len(df_remaining)}] doc_id={doc_id}")
    print(f"Source length (chars): {len(source_text)}")

    result = row.to_dict()

    for out_col, tgt_lang in TARGETS.items():
        try:
            print(f"  Translating -> {out_col} ({tgt_lang})")
            result[out_col] = translate_document(source_text, tgt_lang)
            print(f"  Done -> {out_col}, output chars: {len(result[out_col])}")
        except RuntimeError as e:
            print(f"  RuntimeError in {out_col}: {repr(e)}")
            result[out_col] = ""
            free_memory()
        except Exception as e:
            print(f"  Error in {out_col}: {repr(e)}")
            result[out_col] = ""

    all_rows.append(result)

    should_save = (
        (len(all_rows) % CHECKPOINT_EVERY_DOCS == 0)
        or (i == len(df_remaining) - 1)
    )

    if should_save:
        out_df = pd.DataFrame(all_rows)
        out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"Checkpoint saved -> {OUTPUT_CSV}")

print("=" * 60)
print("Translation complete.")
print("Saved to:", OUTPUT_CSV)
print("=" * 60)