import json
import os
import re
import time
from difflib import SequenceMatcher
from google import genai

# ---------------- CONFIG ----------------
INPUT_GLOSSARY = r"C:\Users\Charunya\Desktop\medical_glossary.json"

CLEAN_GLOSSARY = r"C:\Users\Charunya\Desktop\medical_glossary_clean.json"
DUPLICATE_REPORT = r"C:\Users\Charunya\Desktop\medical_glossary_duplicate_report.json"
USER_FORMS_GLOSSARY = r"C:\Users\Charunya\Desktop\medical_glossary_singlish_tamilish.json"

MODEL_NAME = "gemini-3-flash-preview"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ---------------- HELPERS ----------------

def normalize_term(term: str) -> str:
    term = str(term).strip().lower()
    term = re.sub(r"[-_/]", " ", term)
    term = re.sub(r"[^\w\s]", "", term)
    term = re.sub(r"\s+", " ", term)
    return term.strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_term(a), normalize_term(b)).ratio()


def llm_call(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not response.text:
        raise ValueError("Empty response from Gemini")

    return response.text.strip()


def extract_json_block(text: str):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("No valid JSON found in model response.")


def clean_list(values):
    cleaned = []

    if not isinstance(values, list):
        return cleaned

    seen = set()

    for value in values:
        if not isinstance(value, str):
            continue

        item = value.strip()
        if not item:
            continue

        key = normalize_term(item)

        if key not in seen:
            seen.add(key)
            cleaned.append(item)

    return cleaned


# ---------------- STEP 1: CLEAN DUPLICATES ----------------

def choose_best_term(terms):
    """
    Chooses the best canonical English term.
    Example:
    acute kidney failure -> Acute kidney failure
    """
    terms = [t for t in terms if isinstance(t, str) and t.strip()]

    def score(term):
        words = term.split()
        title_like = sum(1 for w in words if w[:1].isupper())
        all_lower = term.islower()
        return (
            all_lower,
            -title_like,
            len(term)
        )

    return sorted(terms, key=score)[0]


def clean_glossary(glossary, fuzzy_threshold=0.94):
    """
    Removes:
    - exact normalized duplicates
    - near spelling/capitalization duplicates

    Increase fuzzy_threshold to 0.97 if it removes too much.
    Decrease to 0.90 if it misses spelling duplicates.
    """
    groups = []

    for term, translations in glossary.items():
        if not isinstance(term, str) or not term.strip():
            continue

        placed = False

        for group in groups:
            representative = group["terms"][0]

            if normalize_term(term) == normalize_term(representative):
                group["terms"].append(term)
                placed = True
                break

            if similarity(term, representative) >= fuzzy_threshold:
                group["terms"].append(term)
                placed = True
                break

        if not placed:
            groups.append({
                "terms": [term],
                "translations": translations
            })

    cleaned = {}
    duplicate_report = {}

    for group in groups:
        canonical = choose_best_term(group["terms"])

        translations = glossary.get(canonical, group["translations"])

        cleaned[canonical] = {
            "si": str(translations.get("si", "")).strip() if isinstance(translations, dict) else "",
            "ta": str(translations.get("ta", "")).strip() if isinstance(translations, dict) else ""
        }

        if len(group["terms"]) > 1:
            duplicate_report[canonical] = group["terms"]

    return cleaned, duplicate_report


# ---------------- STEP 2: GENERATE SINGLISH + TAMILISH ----------------

def generate_singlish_tamilish_forms(cleaned_glossary):
    user_forms = {}

    items = list(cleaned_glossary.items())
    batch_size = 30

    total_batches = (len(items) + batch_size - 1) // batch_size

    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]

        batch_payload = {
            term: translations
            for term, translations in batch_items
        }

        prompt = f"""
You are creating a search-friendly medical terminology expansion glossary for Sri Lankan users.

Task:
For each English medical term, generate likely informal romanized Sinhala and romanized Tamil forms.

Include:
1. singlish_forms:
   - Sinhala medical term written using English letters
   - Common Sri Lankan romanized Sinhala spellings
   - Must mean the same concept as the English term
   - Do not create full search queries
   - Do not add words like symptoms, treatment, medicine, doctor, hospital, test, causes, or pain unless they are part of the original term

2. tamilish_forms:
   - Tamil medical term written using English letters
   - Common Sri Lankan romanized Tamil spellings
   - Must mean the same concept as the English term
   - Do not create full search queries
   - Do not add words like symptoms, treatment, medicine, doctor, hospital, test, causes, or pain unless they are part of the original term

Rules:
- Return ONLY valid JSON.
- Keep the English term exactly as the key.
- Do not remove any input term.
- Do not add new English terms.
- Do not create advice, diagnosis, treatment, symptom, or medicine query phrases.
- Create terminology variants only.
- Each list should contain 2 to 6 useful variants.
- Avoid duplicates within the same list.
- Use lowercase English letters for romanized forms where possible.
- Preserve the original Sinhala and Tamil translations in the output.

Input glossary:
{json.dumps(batch_payload, ensure_ascii=False, indent=2)}

Output format:
{{
  "English Term": {{
    "si": "Sinhala translation",
    "ta": "Tamil translation",
    "singlish_forms": ["..."],
    "tamilish_forms": ["..."]
  }}
}}
"""

        print(f"Generating batch {i // batch_size + 1} / {total_batches}")

        try:
            response = llm_call(prompt)
            parsed = extract_json_block(response)

            for term, data in parsed.items():
                if term not in cleaned_glossary:
                    continue

                if not isinstance(data, dict):
                    continue

                user_forms[term] = {
                    "si": cleaned_glossary[term]["si"],
                    "ta": cleaned_glossary[term]["ta"],
                    "singlish_forms": clean_list(data.get("singlish_forms", [])),
                    "tamilish_forms": clean_list(data.get("tamilish_forms", [])),
                }

        except Exception as e:
            print(f"Failed batch starting at index {i}: {e}")

            for term, translations in batch_payload.items():
                user_forms[term] = {
                    "si": translations["si"],
                    "ta": translations["ta"],
                    "singlish_forms": [],
                    "tamilish_forms": [],
                }

        time.sleep(1)

    return user_forms


# ---------------- MAIN ----------------

if __name__ == "__main__":
    with open(INPUT_GLOSSARY, "r", encoding="utf-8") as f:
        glossary = json.load(f)

    print(f"Original glossary entries: {len(glossary)}")

    cleaned_glossary, duplicate_report = clean_glossary(
        glossary,
        fuzzy_threshold=0.94
    )

    print(f"Cleaned glossary entries: {len(cleaned_glossary)}")
    print(f"Duplicate groups found: {len(duplicate_report)}")

    with open(CLEAN_GLOSSARY, "w", encoding="utf-8") as f:
        json.dump(cleaned_glossary, f, ensure_ascii=False, indent=4)

    with open(DUPLICATE_REPORT, "w", encoding="utf-8") as f:
        json.dump(duplicate_report, f, ensure_ascii=False, indent=4)

    print(f"Clean glossary saved to: {CLEAN_GLOSSARY}")
    print(f"Duplicate report saved to: {DUPLICATE_REPORT}")

    singlish_tamilish_glossary = generate_singlish_tamilish_forms(cleaned_glossary)

    with open(USER_FORMS_GLOSSARY, "w", encoding="utf-8") as f:
        json.dump(singlish_tamilish_glossary, f, ensure_ascii=False, indent=4)

    print(f"Singlish/Tamilish glossary saved to: {USER_FORMS_GLOSSARY}")
