import pandas as pd
import json
import os
import re
import time
from google import genai

# --- CONFIGURATION ---
INPUT_CSV = r"C:\Users\Charunya\Desktop\final_unique_medical_corpus.csv"
GLOSSARY_FILE = r"C:\Users\Charunya\Desktop\medical_glossary.json"
TEXT_COLUMN = "clean_text"
MODEL_NAME = "gemini-3-flash-preview"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def llm_call(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text.strip()


def extract_json_block(text: str):
    """
    Tries to extract the first JSON object from model output.
    Useful if the model wraps JSON in markdown fences.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text.strip()).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("No valid JSON found in model response.")


def get_unique_terms(df, column_name):
    """
    Extract unique terminology by sampling chunks of the CSV.
    Returns a sorted list of English terms.
    """
    all_raw_terms = set()

    step = 20
    for i in range(0, len(df), step):
        sample_text = " ".join(df[column_name].iloc[i:i + step].fillna("").astype(str))

        extraction_prompt = f"""
You are extracting terminology for a medical glossary.

Task:
Extract every unique English term in these categories only:
- medical conditions
- diseases
- symptoms
- medication names
- drug classes
- surgical procedures
- clinical procedures
- anatomical terms
- diagnostic tests

Rules:
- Return ONLY valid JSON
- Do not include explanations
- Do not include duplicate terms
- Keep terms in English
- Preserve standard medical capitalization where appropriate
- Ignore hospital names, person names, addresses, dates, page headers, boilerplate, URLs, and document formatting noise

Output format:
{{
  "terms": ["term1", "term2", "term3"]
}}

Text:
{sample_text}
"""

        print(f"Extracting terms from rows {i} to {min(i + step - 1, len(df) - 1)}...")

        try:
            response = llm_call(extraction_prompt)
            parsed = extract_json_block(response)
            terms = parsed.get("terms", [])

            for term in terms:
                if isinstance(term, str) and term.strip():
                    all_raw_terms.add(term.strip())
        except Exception as e:
            print(f"Extraction failed for rows {i}-{i + step}: {e}")

        time.sleep(1)

    return sorted(all_raw_terms, key=str.lower)


def create_master_glossary(term_list):
    """
    Translate the unique English term list into Sinhala and Tamil glossary entries.
    """
    glossary = {}

    batch_size = 50
    for i in range(0, len(term_list), batch_size):
        batch = term_list[i:i + batch_size]

        translation_prompt = f"""
You are creating a trilingual medical glossary.

Task:
Translate each English medical term into:
- Sinhala (si)
- Tamil (ta)

Rules:
- Return ONLY valid JSON
- Keep the English term exactly as given as the key
- For medication names, procedure names, and highly technical biomedical terms:
  use a careful phonetic transliteration in Sinhala/Tamil, and optionally include the English in parentheses only if needed for clarity
- For common conditions, symptoms, anatomy, and diagnostic concepts:
  use the most natural and medically appropriate Sinhala/Tamil equivalent
- Do not omit any term
- Do not add extra terms
- Do not add notes or explanations

Output format:
{{
  "English Term 1": {{"si": "Sinhala translation", "ta": "Tamil translation"}},
  "English Term 2": {{"si": "Sinhala translation", "ta": "Tamil translation"}}
}}

Terms:
{json.dumps(batch, ensure_ascii=False)}
"""

        print(f"Translating glossary batch {i // batch_size + 1} / {(len(term_list) + batch_size - 1) // batch_size}...")

        try:
            response = llm_call(translation_prompt)
            batch_glossary = extract_json_block(response)

            for eng_term, translations in batch_glossary.items():
                if (
                    isinstance(translations, dict)
                    and "si" in translations
                    and "ta" in translations
                ):
                    glossary[eng_term] = {
                        "si": str(translations["si"]).strip(),
                        "ta": str(translations["ta"]).strip(),
                    }
                else:
                    print(f"Skipping malformed entry for: {eng_term}")

        except Exception as e:
            print(f"Error parsing translation batch starting at index {i}: {e}")

        time.sleep(1)

    return glossary


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not found. Available columns: {list(df.columns)}")

    unique_english_terms = get_unique_terms(df, TEXT_COLUMN)
    print(f"Found {len(unique_english_terms)} unique medical terms.")

    master_glossary = create_master_glossary(unique_english_terms)

    with open(GLOSSARY_FILE, "w", encoding="utf-8") as f:
        json.dump(master_glossary, f, ensure_ascii=False, indent=4)

    print(f"Glossary saved to {GLOSSARY_FILE}")
