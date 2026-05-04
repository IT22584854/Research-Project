import os
import re
import json
import time
from typing import List, Dict, Any, Tuple

import pandas as pd
from google import genai

# =========================
# CONFIG
# =========================
INPUT_CSV = r"C:\Users\Charunya\Desktop\top_100_filtered_medical_docs.csv"
GLOSSARY_FILE = r"C:\Users\Charunya\Desktop\medical_glossary.json"
OUTPUT_CSV = r"C:\Users\Charunya\Desktop\medical_translation_chunks.csv"

MODEL_NAME = "gemini-2.5-flash"
TEXT_COLUMN = "clean_text"
DOC_ID_COLUMN = "doc_id"

MAX_CHARS_PER_CHUNK = 3000
OVERLAP_CHARS = 300
MAX_GLOSSARY_TERMS_PER_CHUNK = 80
BATCH_SIZE = 10
CSV_FLUSH_EVERY = 25
SLEEP_BETWEEN_CALLS = 1.0
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 3
LIMIT_ROWS = None  # Set to None for full run

client = genai.Client(
    vertexai=True,
    project="nenagovsl",
    location="us-central1",
    http_options={'retry_options': {'attempts': 5}}
)

# =========================
# AMBIGUOUS ACRONYM RULES
# =========================
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
        "Atrial Septal Defect": ["heart", "cardiac", "congenital", "septal", "atrial"]
    },
    "MS": {
        "Multiple Sclerosis": ["neurolog", "demyelin", "cns", "relapse"],
        "Mitral Stenosis": ["valve", "cardiac", "murmur", "mitral"]
    },
    "CP": {
        "Cerebral Palsy": ["motor disorder", "spastic", "neurolog"],
        "Chest Pain": ["pain", "chest", "angina", "cardiac"]
    },
    "RA": {
        "Rheumatoid Arthritis": ["joint", "arthritis", "autoimmune", "inflammation"],
        "Right Atrium": ["heart", "cardiac", "atrium", "echocardiogram"]
    },
    "BP": {
        "Blood Pressure": ["hypertension", "mmhg", "systolic", "diastolic"],
        "Bipolar Disorder": ["psychiatric", "mood", "mania", "depression"]
    },
    "PT": {
        "Physical Therapy": ["rehabilitation", "exercise", "mobility"],
        "Prothrombin Time": ["coagulation", "inr", "clotting"],
        "Patient": ["the pt", "pt presented", "pt was"]
    },
    "DM": {
        "Diabetes Mellitus": ["glucose", "insulin", "hyperglycemia"],
        "Dermatomyositis": ["muscle weakness", "rash", "myositis"]
    },
    "MR": {
        "Mitral Regurgitation": ["valve", "cardiac", "mitral"],
        "Mental Retardation": ["intellectual disability", "developmental"],
        "Medical Record": ["record", "chart", "documentation"]
    }
}

# =========================
# CORE FUNCTIONS
# =========================

def read_csv_with_fallback(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            df.columns = df.columns.str.strip()
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read CSV.")

def load_glossary(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, max_chars: int = 3000, overlap_chars: int = 300) -> List[str]:
    text = normalize_whitespace(text)
    if not text: return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}"
        else:
            chunks.append(current)
            # Take overlap from end of current chunk
            overlap = current[-overlap_chars:].strip()
            current = f"{overlap}\n\n{para}" if overlap else para
    
    if current: chunks.append(current)
    return chunks

def resolve_ambiguous_acronyms(text: str) -> Dict[str, str]:
    text_lower = text.lower()
    resolved = {}
    for acronym, expansions in AMBIGUOUS_ACRONYMS.items():
        if not re.search(rf"\b{re.escape(acronym.lower())}\b", text_lower):
            continue
        scores = {}
        for exp in expansions:
            score = sum(1 for hint in ACRONYM_CONTEXT_HINTS.get(acronym, {}).get(exp, []) if hint.lower() in text_lower)
            if score > 0: scores[exp] = score
        if scores:
            best = max(scores.items(), key=lambda x: x[1])
            if list(scores.values()).count(best[1]) == 1:
                resolved[acronym] = best[0]
    return resolved

def find_relevant_glossary_terms(text: str, glossary: Dict, max_terms: int) -> Tuple[Dict, Dict]:
    text_lower = text.lower()
    matches = []
    resolved = resolve_ambiguous_acronyms(text)
    for term in glossary.keys():
        if len(term) <= 6 or term.isupper():
            if re.search(rf"\b{re.escape(term.lower())}\b", text_lower): matches.append(term)
        elif term.lower() in text_lower:
            matches.append(term)
    matches = sorted(set(matches), key=str.lower)[:max_terms]
    return {t: glossary[t] for t in matches}, resolved

def build_translation_prompt(source_text: str, target_lang: str, glossary_block: str, acronym_block: str) -> str:
    target_label = target_lang.capitalize()
    return f"""
You are an expert medical translator. 

### Step 1: Context Analysis
Analyze the source text to determine the specific medical specialty (e.g., Neurology, Cardiology, Pediatrics, etc.). Ensure all terminology and acronyms are translated according to that specific field's standards.

### Step 2: Translation Task
Translate the text into {target_label} following these rules:
1. **Consistency:** If an acronym (like ASD, MS, or PT) has multiple meanings, use the one that matches the medical specialty identified in Step 1.
2. **Glossary:** Use the provided glossary terms exactly.
3. **No Omissions:** Translate every sentence and heading. Do not summarize.
4. **Technical Terms:** For highly specialized terms without a direct {target_label} equivalent, provide a transliteration in {target_label} script followed by the English term in parentheses.
5. **Output:** Return ONLY the translated text.

### Reference Data:
Resolved Acronym Suggestions (Verify against context):
{acronym_block}

Matched Glossary:
{glossary_block}

### Source Text to Translate:
{source_text}
""".strip()

def translate_chunk_with_retry(source_text: str, target_lang: str, glossary: Dict) -> Dict[str, Any]:
    matched_terms, resolved_acronyms = find_relevant_glossary_terms(source_text, glossary, MAX_GLOSSARY_TERMS_PER_CHUNK)
    
    lang_key = "si" if target_lang == "sinhala" else "ta"
    g_lines = [f"{k} -> {v.get(lang_key, '')}" for k, v in matched_terms.items()]
    a_lines = [f"{k} = {v}" for k, v in resolved_acronyms.items()]
    
    prompt = build_translation_prompt(source_text, target_lang, "\n".join(g_lines), "\n".join(a_lines))
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            if not response.text:
                # Check for safety blocks
                if response.candidates[0].finish_reason:
                    return {"translated_text": "", "status": "blocked", "error_message": f"Safety Finish Reason: {response.candidates[0].finish_reason}"}
                raise ValueError("Empty response")
            return {"translated_text": response.text.strip(), "status": "success", "attempt_count": attempt}
        except Exception as e:
            if attempt == MAX_RETRIES: return {"translated_text": "", "status": "failed", "error_message": str(e)}
            time.sleep(RETRY_SLEEP_SECONDS)

# =========================
# MAIN EXECUTION
# =========================
def main():
    df = read_csv_with_fallback(INPUT_CSV)
    if LIMIT_ROWS: df = df.head(LIMIT_ROWS)
    glossary = load_glossary(GLOSSARY_FILE)
    
    results = []
    for idx, row in df.iterrows():
        doc_id = str(row[DOC_ID_COLUMN])
        chunks = chunk_text(str(row[TEXT_COLUMN]))
        
        for c_idx, chunk in enumerate(chunks):
            for lang in ["sinhala", "tamil"]:
                print(f"Doc {doc_id} | Chunk {c_idx} | {lang}...")
                res = translate_chunk_with_retry(chunk, lang, glossary)
                res.update({"doc_id": doc_id, "chunk_index": c_idx, "target_lang": lang, "source_text": chunk})
                results.append(res)
                
                if len(results) >= CSV_FLUSH_EVERY:
                    pd.DataFrame(results).to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False, encoding="utf-8-sig")
                    results = []
                time.sleep(SLEEP_BETWEEN_CALLS)

    if results: pd.DataFrame(results).to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False, encoding="utf-8-sig")
    print("Done!")

if __name__ == "__main__":
    main()
