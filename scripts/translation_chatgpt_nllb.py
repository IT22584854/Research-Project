import os
import time
import pandas as pd
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_CSV = r"C:\Users\Charunya\Desktop\medical_translation_chunks.csv"
OUTPUT_CSV = r"C:\Users\Charunya\Desktop\medical_translation_chunks_TEST100_with_chatgpt_nllb.csv"

SOURCE_COL = "source_text"
TARGET_LANG_COL = "target_lang"
GEMINI_COL = "translated_text"

CHATGPT_OUTPUT_COL = "chatgpt_translation"
NLLB_OUTPUT_COL = "nllb_translation"

CHATGPT_MODEL = "gpt-4o-mini"
MAX_WORKERS = 6

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_BATCH_SIZE = 8

TEST_UNIQUE_CHUNKS = 100

NLLB_LANG_CODES = {
    "sinhala": "sin_Sinh",
    "si": "sin_Sinh",
    "tamil": "tam_Taml",
    "ta": "tam_Taml",
}

TARGET_LANG_NAMES = {
    "sinhala": "Sinhala",
    "si": "Sinhala",
    "tamil": "Tamil",
    "ta": "Tamil",
}


# =========================================================
# HELPERS
# =========================================================

def normalize_lang(lang):
    return str(lang).strip().lower()


def save_progress(df):
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


def select_first_100_unique_chunks(df):
    first_chunks = (
        df[["doc_id", "chunk_index"]]
        .drop_duplicates()
        .head(TEST_UNIQUE_CHUNKS)
    )

    selected = df.merge(first_chunks, on=["doc_id", "chunk_index"], how="inner")

    print(f"[INFO] Selected rows: {len(selected)}")
    print(
        "[INFO] Selected unique chunks:",
        selected[["doc_id", "chunk_index"]].drop_duplicates().shape[0]
    )

    return selected


# =========================================================
# CHATGPT TRANSLATION
# =========================================================

client = OpenAI()


def build_chatgpt_prompt(source_text, target_lang):
    target_name = TARGET_LANG_NAMES.get(normalize_lang(target_lang), target_lang)

    return f"""
Translate the following medical/public-health text into {target_name}.

Rules:
- Use only the source text.
- Do not use or copy any existing translation.
- Do not add explanations.
- Do not add new facts.
- Preserve medical meaning accurately.
- Return only the translation.

Source text:
{source_text}
""".strip()


def translate_chatgpt(row):
    source_text = str(row[SOURCE_COL]).strip()
    target_lang = normalize_lang(row[TARGET_LANG_COL])

    if not source_text:
        return ""

    prompt = build_chatgpt_prompt(source_text, target_lang)

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=CHATGPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful medical translation assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            wait = 2 ** attempt
            print(f"[WARN] ChatGPT failed on attempt {attempt + 1}/5. Waiting {wait}s. Error: {e}")
            time.sleep(wait)

    return ""


def run_chatgpt_parallel(df):
    if CHATGPT_OUTPUT_COL not in df.columns:
        df[CHATGPT_OUTPUT_COL] = ""

    jobs = []

    for idx, row in df.iterrows():
        existing = str(row.get(CHATGPT_OUTPUT_COL, "")).strip()

        if existing and existing.lower() != "nan":
            continue

        jobs.append((idx, row))

    print(f"[INFO] ChatGPT jobs: {len(jobs)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(translate_chatgpt, row): idx
            for idx, row in jobs
        }

        completed = 0

        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="ChatGPT"):
            idx = future_to_idx[future]

            try:
                df.at[idx, CHATGPT_OUTPUT_COL] = future.result()
            except Exception as e:
                print(f"[ERROR] ChatGPT row {idx} failed: {e}")
                df.at[idx, CHATGPT_OUTPUT_COL] = ""

            completed += 1

            if completed % 20 == 0:
                save_progress(df)
                print(f"[SAVE] ChatGPT progress saved after {completed} rows.")

    save_progress(df)
    return df


# =========================================================
# NLLB GPU TRANSLATION
# =========================================================

def load_nllb():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading NLLB model: {NLLB_MODEL}")
    print(f"[INFO] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)

    model.to(device)
    model.eval()

    return tokenizer, model, device


def translate_nllb_one(text, target_lang, tokenizer, model, device):
    target_lang = normalize_lang(target_lang)
    target_code = NLLB_LANG_CODES.get(target_lang)

    if not target_code:
        return ""

    tokenizer.src_lang = "eng_Latn"

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.inference_mode():
        output_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
            max_new_tokens=512,
            num_beams=4
        )

    translation = tokenizer.batch_decode(
        output_tokens,
        skip_special_tokens=True
    )[0]

    return translation.strip()


def run_nllb_gpu(df):
    if NLLB_OUTPUT_COL not in df.columns:
        df[NLLB_OUTPUT_COL] = ""

    tokenizer, model, device = load_nllb()

    jobs = []

    for idx, row in df.iterrows():
        existing = str(row.get(NLLB_OUTPUT_COL, "")).strip()

        if existing and existing.lower() != "nan":
            continue

        jobs.append(idx)

    print(f"[INFO] NLLB jobs: {len(jobs)}")

    for start in tqdm(range(0, len(jobs), NLLB_BATCH_SIZE), desc="NLLB"):
        batch_indices = jobs[start:start + NLLB_BATCH_SIZE]

        for idx in batch_indices:
            text = df.at[idx, SOURCE_COL]
            target_lang = df.at[idx, TARGET_LANG_COL]

            try:
                df.at[idx, NLLB_OUTPUT_COL] = translate_nllb_one(
                    text=text,
                    target_lang=target_lang,
                    tokenizer=tokenizer,
                    model=model,
                    device=device
                )
            except Exception as e:
                print(f"[ERROR] NLLB row {idx} failed: {e}")
                df.at[idx, NLLB_OUTPUT_COL] = ""

        if start % (NLLB_BATCH_SIZE * 5) == 0:
            save_progress(df)
            print("[SAVE] NLLB progress saved.")

    save_progress(df)
    return df


# =========================================================
# MAIN
# =========================================================

def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    print("[INFO] Columns found:")
    print(df.columns.tolist())

    required_cols = [
        SOURCE_COL,
        TARGET_LANG_COL,
        GEMINI_COL,
        "doc_id",
        "chunk_index"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = select_first_100_unique_chunks(df)

    print("[INFO] Existing Gemini translations are kept but ignored during generation.")

    df = run_chatgpt_parallel(df)
    df = run_nllb_gpu(df)

    save_progress(df)

    print("\n[DONE] Test translation complete.")
    print(f"[DONE] Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()