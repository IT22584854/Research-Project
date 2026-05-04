import pandas as pd
import numpy as np
import re
import math
from collections import Counter

# =========================================================
# FILE PATHS
# =========================================================
GEMINI_FILE = r"C:\Users\Charunya\Desktop\medical_translation_chunks.csv"
CHATGPT_FILE = r"C:\Users\Charunya\Desktop\medical_translation_chunks_TEST100_with_chatgpt_glossary.csv"
NLLB_FILE = r"C:\Users\Charunya\Downloads\translations.jsonl"

OUTPUT_STATS = r"C:\Users\Charunya\Desktop\translation_model_stats_comparison.csv"
OUTPUT_MANUAL_SAMPLE = r"C:\Users\Charunya\Desktop\manual_translation_review_sample.csv"

MATTR_WINDOW = 100
SAMPLE_SIZE = 20


# =========================================================
# BASIC METRIC FUNCTIONS
# =========================================================
def tokenize(text):
    return re.findall(r"\w+", str(text).lower())


def calculate_mattr(tokens, window_size=100):
    if not tokens:
        return 0

    if len(tokens) < window_size:
        return len(set(tokens)) / len(tokens)

    scores = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        scores.append(len(set(window)) / window_size)

    return float(np.mean(scores))


def calculate_entropy(tokens):
    if not tokens:
        return 0

    counts = Counter(tokens)
    total = len(tokens)

    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )


def english_leakage(tokens):
    if not tokens:
        return 0

    english_tokens = [t for t in tokens if re.match(r"^[a-z0-9]+$", t)]
    return (len(english_tokens) / len(tokens)) * 100


def run_stats(df, model_name, language, text_col):
    text = " ".join(df[text_col].fillna("").astype(str))
    tokens = tokenize(text)

    if not tokens:
        return None

    return {
        "Model": model_name,
        "Language": language,
        "TTR": round(len(set(tokens)) / len(tokens), 4),
        "MATTR": round(calculate_mattr(tokens, MATTR_WINDOW), 4),
        "English_Leakage_CMI_%": round(english_leakage(tokens), 2),
        "Wordform_Entropy": round(calculate_entropy(tokens), 4),
        "Avg_Row_Token_Length": round(df[text_col].fillna("").astype(str).str.split().str.len().mean(), 2),
    }


# =========================================================
# LOADERS
# =========================================================
def load_gemini():
    """
    Handles Gemini file in long format:
    expected columns: translated_text, target_lang
    """
    df = pd.read_csv(GEMINI_FILE, encoding="utf-8-sig")

    print("\nGemini columns:")
    print(df.columns.tolist())

    required = ["translated_text", "target_lang"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Gemini file missing column: {col}")

    rows = []

    for lang_value in df["target_lang"].dropna().unique():
        lang_df = df[df["target_lang"] == lang_value].copy()

        lang_name = str(lang_value).strip().lower()
        if "sin" in lang_name or "si" == lang_name:
            language = "Sinhala"
        elif "tam" in lang_name or "ta" == lang_name:
            language = "Tamil"
        else:
            language = lang_value

        stats = run_stats(lang_df, "Gemini 2.5 Flash", language, "translated_text")
        if stats:
            rows.append(stats)

    return rows, df


def find_first_existing_col(df, possible_cols, file_label):
    for col in possible_cols:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find any expected column for {file_label}. "
        f"Tried: {possible_cols}. Found: {df.columns.tolist()}"
    )

def load_chatgpt_long_csv():
    df = pd.read_csv(CHATGPT_FILE, encoding="utf-8-sig")

    print("\nChatGPT columns:")
    print(df.columns.tolist())

    required = ["chatgpt_glossary_translation", "target_lang"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"ChatGPT file missing column: {col}")

    rows = []

    for lang_value in df["target_lang"].dropna().unique():
        lang_df = df[df["target_lang"] == lang_value].copy()

        lang_name = str(lang_value).strip().lower()
        if "sin" in lang_name or lang_name == "si" or lang_name == "sinhala":
            language = "Sinhala"
        elif "tam" in lang_name or lang_name == "ta" or lang_name == "tamil":
            language = "Tamil"
        else:
            language = lang_value

        stats = run_stats(
            lang_df,
            "ChatGPT + Glossary",
            language,
            "chatgpt_glossary_translation"
        )

        if stats:
            rows.append(stats)

    return rows, df

def load_nllb_jsonl():
    """
    Handles NLLB JSONL with translation_si and translation_ta columns.
    """
    df = pd.read_json(NLLB_FILE, lines=True, encoding="utf-8-sig")

    print("\nNLLB columns:")
    print(df.columns.tolist())

    required = ["translation_si", "translation_ta"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"NLLB JSONL missing column: {col}")

    rows = []

    stats_si = run_stats(df, "NLLB-200 distilled 1.3B", "Sinhala", "translation_si")
    if stats_si:
        rows.append(stats_si)

    stats_ta = run_stats(df, "NLLB-200 distilled 1.3B", "Tamil", "translation_ta")
    if stats_ta:
        rows.append(stats_ta)

    return rows, df


# =========================================================
# MANUAL REVIEW SAMPLE
# =========================================================
def create_manual_review_sample(chatgpt_df, chatgpt_si_col, chatgpt_ta_col, nllb_df):
    """
    Creates a side-by-side sample for manual scoring.
    Uses ChatGPT CSV and NLLB JSONL because both are expected to have doc_id/chunk_index.
    Gemini is not included here unless its file also has doc_id/chunk_index in a compatible format.
    """
    required_keys = ["doc_id", "chunk_index"]

    for col in required_keys:
        if col not in chatgpt_df.columns or col not in nllb_df.columns:
            print("\n[WARN] Could not create manual sample because doc_id/chunk_index is missing.")
            return

    chat_cols = ["doc_id", "chunk_index"]
    if "source_text" in chatgpt_df.columns:
        chat_cols.append("source_text")

    chat_cols.extend([chatgpt_si_col, chatgpt_ta_col])

    chat_part = chatgpt_df[chat_cols].copy()
    chat_part = chat_part.rename(columns={
        chatgpt_si_col: "chatgpt_translation_si",
        chatgpt_ta_col: "chatgpt_translation_ta",
    })

    nllb_part = nllb_df[
        ["doc_id", "chunk_index", "translation_si", "translation_ta"]
    ].copy()

    nllb_part = nllb_part.rename(columns={
        "translation_si": "nllb_translation_si",
        "translation_ta": "nllb_translation_ta",
    })

    merged = chat_part.merge(
        nllb_part,
        on=["doc_id", "chunk_index"],
        how="inner",
    )

    if merged.empty:
        print("\n[WARN] Manual review sample is empty. doc_id/chunk_index may not match.")
        return

    sample = merged.sample(
        n=min(SAMPLE_SIZE, len(merged)),
        random_state=42,
    )

    sample["Meaning_Gemini"] = ""
    sample["Meaning_ChatGPT"] = ""
    sample["Meaning_NLLB"] = ""

    sample["Medical_Accuracy_Gemini"] = ""
    sample["Medical_Accuracy_ChatGPT"] = ""
    sample["Medical_Accuracy_NLLB"] = ""

    sample["Fluency_Gemini"] = ""
    sample["Fluency_ChatGPT"] = ""
    sample["Fluency_NLLB"] = ""

    sample.to_csv(OUTPUT_MANUAL_SAMPLE, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] Manual review sample saved to: {OUTPUT_MANUAL_SAMPLE}")


# =========================================================
# MAIN
# =========================================================
def main():
    all_stats = []

    gemini_stats, gemini_df = load_gemini()
    all_stats.extend(gemini_stats)

    chatgpt_stats, chatgpt_df = load_chatgpt_long_csv()
    all_stats.extend(chatgpt_stats)

    nllb_stats, nllb_df = load_nllb_jsonl()
    all_stats.extend(nllb_stats)

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(OUTPUT_STATS, index=False, encoding="utf-8-sig")

    print("\n[DONE] Stats comparison saved to:")
    print(OUTPUT_STATS)
    print("\nFinal comparison:")
    print(stats_df.to_string(index=False))

    #create_manual_review_sample(
      #  chatgpt_df=chatgpt_df,
      #  chatgpt_si_col=chatgpt_si_col,
      #  chatgpt_ta_col=chatgpt_ta_col,
      #  nllb_df=nllb_df,
    #)


if __name__ == "__main__":
    main()