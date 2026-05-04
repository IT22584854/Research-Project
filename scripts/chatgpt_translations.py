import os
import re
import json
import time
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = r"C:\Users\Charunya\Desktop\medical_translation_chunks_TEST100_with_chatgpt_nllb.csv"
GLOSSARY_FILE = r"C:\Users\Charunya\Desktop\medical_glossary.json"
OUTPUT_CSV = r"C:\Users\Charunya\Desktop\medical_translation_chunks_TEST100_with_chatgpt_glossary.csv"

SOURCE_COL = "source_text"
TARGET_LANG_COL = "target_lang"
DOC_ID_COL = "doc_id"
CHUNK_INDEX_COL = "chunk_index"

OUTPUT_COL = "chatgpt_glossary_translation"

MODEL_NAME = "gpt-4o-mini"

MAX_GLOSSARY_TERMS_PER_CHUNK = 80
MAX_WORKERS = 6
MAX_RETRIES = 5
SAVE_EVERY = 20

client = OpenAI()

# =========================================================
# AMBIGUOUS ACRONYM RULES
# =========================================================
AMBIGUOUS_ACRONYMS = {
    "ASD": ["Autism Spectrum Disorder", "Atrial Septal Defect"],
    "MS": ["Multiple Sclerosis", "Mitral Stenosis"],
    "CP": ["Cerebral Palsy", "Chest Pain"],
    "RA": ["Rheumatoid Arthritis", "Right Atrium"],
    "BP": ["Blood Pressure", "Bipolar Disorder"],
    "PT": ["Physical Therapy", "Prothrombin Time", "Patient"],
    "DM": ["Diabetes Mellitus", "Dermatomyositis"],
    "MR": ["Mitral Regurgitation", "Mental Retardation", "Medical Record"],
}

ACRONYM_CONTEXT_HINTS = {
    "ASD": {
        "Autism Spectrum Disorder": ["autism", "developmental", "neurodevelopmental", "behavior", "spectrum"],
        "Atrial Septal Defect": ["heart", "cardiac", "congenital", "septal", "atrial"],
    },
    "MS": {
        "Multiple Sclerosis": ["neurolog", "demyelin", "cns", "relapse"],
        "Mitral Stenosis": ["valve", "cardiac", "murmur", "mitral"],
    },
    "CP": {
        "Cerebral Palsy": ["motor disorder", "spastic", "neurolog"],
        "Chest Pain": ["pain", "chest", "angina", "cardiac"],
    },
    "RA": {
        "Rheumatoid Arthritis": ["joint", "arthritis", "autoimmune", "inflammation"],
        "Right Atrium": ["heart", "cardiac", "atrium", "echocardiogram"],
    },
    "BP": {
        "Blood Pressure": ["hypertension", "mmhg", "systolic", "diastolic"],
        "Bipolar Disorder": ["psychiatric", "mood", "mania", "depression"],
    },
    "PT": {
        "Physical Therapy": ["rehabilitation", "exercise", "mobility"],
        "Prothrombin Time": ["coagulation", "inr", "clotting"],
        "Patient": ["the pt", "pt presented", "pt was"],
    },
    "DM": {
        "Diabetes Mellitus": ["glucose", "insulin", "hyperglycemia"],
        "Dermatomyositis": ["muscle weakness", "rash", "myositis"],
    },
    "MR": {
        "Mitral Regurgitation": ["valve", "cardiac", "mitral"],
        "Mental Retardation": ["intellectual disability", "developmental"],
        "Medical Record": ["record", "chart", "documentation"],
    },
}

# =========================================================
# HELPERS
# =========================================================
def load_glossary(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_lang(lang: str) -> str:
    lang = str(lang).strip().lower()

    if lang in ["si", "sinhala", "sinhalese"]:
        return "sinhala"

    if lang in ["ta", "tamil"]:
        return "tamil"

    return lang


def resolve_ambiguous_acronyms(text: str) -> Dict[str, str]:
    text_lower = str(text).lower()
    resolved = {}

    for acronym, expansions in AMBIGUOUS_ACRONYMS.items():
        if not re.search(rf"\b{re.escape(acronym.lower())}\b", text_lower):
            continue

        scores = {}

        for expansion in expansions:
            hints = ACRONYM_CONTEXT_HINTS.get(acronym, {}).get(expansion, [])
            score = sum(1 for hint in hints if hint.lower() in text_lower)

            if score > 0:
                scores[expansion] = score

        if scores:
            best = max(scores.items(), key=lambda x: x[1])

            if list(scores.values()).count(best[1]) == 1:
                resolved[acronym] = best[0]

    return resolved


def find_relevant_glossary_terms(text: str, glossary: Dict, max_terms: int) -> Tuple[Dict, Dict]:
    text_lower = str(text).lower()
    matches = []
    resolved = resolve_ambiguous_acronyms(text)

    for term in glossary.keys():
        term_lower = str(term).lower()

        if len(term) <= 6 or str(term).isupper():
            if re.search(rf"\b{re.escape(term_lower)}\b", text_lower):
                matches.append(term)
        elif term_lower in text_lower:
            matches.append(term)

    matches = sorted(set(matches), key=str.lower)[:max_terms]

    return {term: glossary[term] for term in matches}, resolved


def build_translation_prompt(source_text: str, target_lang: str, glossary_block: str, acronym_block: str) -> str:
    target_label = target_lang.capitalize()

    return f"""
You are an expert medical translator.

### Step 1: Context Analysis
Analyze the source text to determine the specific medical specialty (e.g., Neurology, Cardiology, Pediatrics, etc.). Ensure all terminology and acronyms are translated according to that specific field's standards.

### Step 2: Translation Task
Translate the text into {target_label} following these rules:
1. Consistency: If an acronym (like ASD, MS, or PT) has multiple meanings, use the one that matches the medical specialty identified in Step 1.
2. Glossary: Use the provided glossary terms exactly.
3. No Omissions: Translate every sentence and heading. Do not summarize.
4. Technical Terms: For highly specialized terms without a direct {target_label} equivalent, provide a transliteration in {target_label} script followed by the English term in parentheses.
5. Output: Return ONLY the translated text.

### Reference Data:
Resolved Acronym Suggestions (Verify against context):
{acronym_block}

Matched Glossary:
{glossary_block}

### Source Text to Translate:
{source_text}
""".strip()


def build_prompt_for_row(row, glossary: Dict) -> str:
    source_text = str(row[SOURCE_COL]).strip()
    target_lang = normalize_lang(row[TARGET_LANG_COL])

    matched_terms, resolved_acronyms = find_relevant_glossary_terms(
        source_text,
        glossary,
        MAX_GLOSSARY_TERMS_PER_CHUNK,
    )

    lang_key = "si" if target_lang == "sinhala" else "ta"

    glossary_lines = [
        f"{english_term} -> {translations.get(lang_key, '')}"
        for english_term, translations in matched_terms.items()
        if translations.get(lang_key, "")
    ]

    acronym_lines = [
        f"{acronym} = {meaning}"
        for acronym, meaning in resolved_acronyms.items()
    ]

    glossary_block = "\n".join(glossary_lines) if glossary_lines else "No matched glossary terms."
    acronym_block = "\n".join(acronym_lines) if acronym_lines else "No resolved acronym suggestions."

    return build_translation_prompt(
        source_text=source_text,
        target_lang=target_lang,
        glossary_block=glossary_block,
        acronym_block=acronym_block,
    )


# =========================================================
# CHATGPT TRANSLATION
# =========================================================
def translate_chatgpt_with_gemini_prompt(row, glossary: Dict) -> str:
    source_text = str(row[SOURCE_COL]).strip()

    if not source_text:
        return ""

    prompt = build_prompt_for_row(row, glossary)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medical translator. Follow glossary terms exactly and return only the translated text.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
            )

            output = response.choices[0].message.content

            return output.strip() if output else ""

        except Exception as e:
            wait = 2 ** (attempt - 1)
            print(f"[WARN] Failed attempt {attempt}/{MAX_RETRIES}. Waiting {wait}s. Error: {e}")
            time.sleep(wait)

    return ""


def save_progress(df: pd.DataFrame):
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


def run_parallel(df: pd.DataFrame, glossary: Dict) -> pd.DataFrame:
    if OUTPUT_COL not in df.columns:
        df[OUTPUT_COL] = ""

    jobs = []

    for idx, row in df.iterrows():
        existing = str(row.get(OUTPUT_COL, "")).strip()

        if existing and existing.lower() != "nan":
            continue

        jobs.append((idx, row))

    print(f"[INFO] ChatGPT glossary-aware jobs: {len(jobs)}")

    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(translate_chatgpt_with_gemini_prompt, row, glossary): idx
            for idx, row in jobs
        }

        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="ChatGPT glossary"):
            idx = future_to_idx[future]

            try:
                df.at[idx, OUTPUT_COL] = future.result()
            except Exception as e:
                print(f"[ERROR] Row {idx} failed: {e}")
                df.at[idx, OUTPUT_COL] = ""

            completed += 1

            if completed % SAVE_EVERY == 0:
                save_progress(df)
                print(f"[SAVE] Progress saved after {completed} rows.")

    save_progress(df)
    return df


# =========================================================
# MAIN
# =========================================================
def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    glossary = load_glossary(GLOSSARY_FILE)

    print("[INFO] Columns found:")
    print(df.columns.tolist())

    required_cols = [
        SOURCE_COL,
        TARGET_LANG_COL,
        DOC_ID_COL,
        CHUNK_INDEX_COL,
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = run_parallel(df, glossary)

    print("\n[DONE] ChatGPT translation with Gemini-style glossary prompt completed.")
    print(f"[DONE] Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()