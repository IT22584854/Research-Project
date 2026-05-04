import re
import json
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd


# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = r"C:\Users\Charunya\Desktop\medical_translation_chunks.csv"
GLOSSARY_JSON = r"C:\Users\Charunya\Desktop\medical_glossary.json"

OUTPUT_ROW_SCORES_CSV = r"C:\Users\Charunya\Desktop\chunked_eval_row_scores.csv"
OUTPUT_DOC_SUMMARY_CSV = r"C:\Users\Charunya\Desktop\chunked_eval_doc_summary.csv"
OUTPUT_LANG_SUMMARY_CSV = r"C:\Users\Charunya\Desktop\chunked_eval_language_summary.csv"
OUTPUT_FLAGGED_CSV = r"C:\Users\Charunya\Desktop\chunked_eval_flagged_rows.csv"

SOURCE_COL = "source_text"
TARGET_COL = "translated_text"
DOC_COL = "doc_id"
CHUNK_COL = "chunk_index"
LANG_COL = "target_lang"

KNOWN_OVERLAP_CHARS = 300

# fallback heuristic list if glossary match is unavailable
MEDICAL_TERMS = [
    "disease", "infection", "infectious", "symptom", "symptoms", "diagnosis",
    "treatment", "therapy", "prevention", "vaccine", "vaccination", "virus",
    "bacteria", "clinical", "patient", "patients", "blood", "heart", "lung",
    "lungs", "kidney", "brain", "fever", "pain", "risk", "risks", "dose",
    "dosage", "medicine", "medical", "hospital", "epidemiology", "epidemic",
    "pandemic", "screening", "hypertension", "diabetes", "stroke", "cancer",
    "malaria", "dengue", "tuberculosis", "covid", "influenza"
]

CONNECTOR_WORDS = {
    "and", "or", "but", "because", "while", "when", "which", "that", "of",
    "in", "on", "for", "with", "to", "from", "by", "during", "through",
    "into", "as", "if", "than", "then", "also"
}

# glossary key guesses
SOURCE_TERM_KEYS = [
    "source_term", "english", "en", "term", "source", "source_text", "label"
]
SI_KEYS = [
    "si", "sinhala", "preferred_si", "translation_si", "target_si"
]
TA_KEYS = [
    "ta", "tamil", "preferred_ta", "translation_ta", "target_ta"
]
ALT_SI_KEYS = [
    "alt_si", "alternate_si", "alternates_si", "variants_si", "synonyms_si"
]
ALT_TA_KEYS = [
    "alt_ta", "alternate_ta", "alternates_ta", "variants_ta", "synonyms_ta"
]


# =========================================================
# HELPERS
# =========================================================
def clean_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_match(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def char_ratio(src: str, trg: str) -> float:
    s = max(len(src), 1)
    return len(trg) / s


def sequence_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def starts_like_heading(text: str) -> bool:
    text = text.lstrip()
    if not text:
        return False
    patterns = [
        r"^#{1,6}\s",
        r"^\d+[\.\)]\s",
        r"^[\-\*\u2022]\s",
        r"^[A-Z][A-Za-z0-9 ,:/\-\(\)]{0,80}$",
    ]
    return any(re.match(p, text) for p in patterns)


def ends_sentence(text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    return text[-1] in ".?!:;।"


def begins_mid_sentence(text: str) -> bool:
    text = text.lstrip()
    if not text:
        return False
    first_token_match = re.match(r"^([^\s,;:.!?]+)", text)
    first_token = first_token_match.group(1).lower() if first_token_match else ""

    starts_lower = text[:1].islower()
    starts_connector = first_token in CONNECTOR_WORDS
    starts_punct = bool(re.match(r"^[,;:\)\]\-]", text))
    return (starts_lower or starts_connector or starts_punct) and not starts_like_heading(text)


def count_medical_terms(text: str) -> int:
    text_l = text.lower()
    total = 0
    for term in MEDICAL_TERMS:
        total += len(re.findall(rf"\b{re.escape(term)}\b", text_l))
    return total


def extract_medical_terms(text: str):
    text_l = text.lower()
    found = set()
    for term in MEDICAL_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text_l):
            found.add(term)
    return found


def longest_suffix_prefix_overlap(a: str, b: str, min_overlap: int = 60, max_overlap: int = 450) -> int:
    a_n = normalize_for_match(a)
    b_n = normalize_for_match(b)
    max_len = min(len(a_n), len(b_n), max_overlap)
    for k in range(max_len, min_overlap - 1, -1):
        if a_n[-k:] == b_n[:k]:
            return k
    return 0


def repeated_phrase_score(prev_text: str, next_text: str, window: int = 450) -> float:
    prev_tail = normalize_for_match(prev_text)[-window:]
    next_head = normalize_for_match(next_text)[:window]
    if not prev_tail or not next_head:
        return 0.0
    return sequence_sim(prev_tail, next_head)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# =========================================================
# GLOSSARY FUNCTIONS
# =========================================================
def first_existing_key(d: dict, key_candidates):
    for k in key_candidates:
        if k in d and d[k] not in [None, "", []]:
            return d[k]
    return None


def to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        # split pipe/semicolon/comma only if present
        if "|" in value:
            return [x.strip() for x in value.split("|") if x.strip()]
        if ";" in value:
            return [x.strip() for x in value.split(";") if x.strip()]
        return [value]
    return [str(value).strip()]


def load_glossary(glossary_path: str):
    """
    Supports either:
    1. dict-of-dicts:
       {
         "hypertension": {"si": "...", "ta": "..."},
         "blood clot": {"preferred_si": "...", "preferred_ta": "..."}
       }

    2. list-of-records:
       [
         {"source_term": "hypertension", "si": "...", "ta": "..."},
         ...
       ]
    """
    path = Path(glossary_path)
    if not path.exists():
        print(f"[WARN] Glossary file not found: {glossary_path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    glossary = {}

    if isinstance(data, dict):
        # dict-of-dicts OR dict-of-strings
        for raw_key, raw_value in data.items():
            source_term = str(raw_key).strip().lower()
            if not source_term:
                continue

            entry = {
                "source_term": source_term,
                "si_terms": [],
                "ta_terms": [],
            }

            if isinstance(raw_value, dict):
                si_main = first_existing_key(raw_value, SI_KEYS)
                ta_main = first_existing_key(raw_value, TA_KEYS)
                si_alt = first_existing_key(raw_value, ALT_SI_KEYS)
                ta_alt = first_existing_key(raw_value, ALT_TA_KEYS)

                entry["si_terms"] = to_list(si_main) + to_list(si_alt)
                entry["ta_terms"] = to_list(ta_main) + to_list(ta_alt)
            else:
                # if value is plain string, not useful for bilingual match
                entry["si_terms"] = []
                entry["ta_terms"] = []

            glossary[source_term] = entry

    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue

            source_term = first_existing_key(row, SOURCE_TERM_KEYS)
            if not source_term:
                continue

            source_term = str(source_term).strip().lower()
            if not source_term:
                continue

            si_main = first_existing_key(row, SI_KEYS)
            ta_main = first_existing_key(row, TA_KEYS)
            si_alt = first_existing_key(row, ALT_SI_KEYS)
            ta_alt = first_existing_key(row, ALT_TA_KEYS)

            glossary[source_term] = {
                "source_term": source_term,
                "si_terms": to_list(si_main) + to_list(si_alt),
                "ta_terms": to_list(ta_main) + to_list(ta_alt),
            }

    else:
        print("[WARN] Unsupported glossary JSON structure. Proceeding without glossary.")
        return {}

    # normalize and deduplicate target terms
    for term, entry in glossary.items():
        entry["si_terms"] = list(dict.fromkeys([normalize_for_match(x).lower() for x in entry["si_terms"] if x]))
        entry["ta_terms"] = list(dict.fromkeys([normalize_for_match(x).lower() for x in entry["ta_terms"] if x]))

    print(f"[INFO] Loaded glossary entries: {len(glossary)}")
    return glossary


def find_relevant_glossary_terms(text: str, glossary: dict, max_terms: int = 80):
    text_lower = normalize_for_match(text).lower()
    matches = []

    for term in glossary.keys():
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, text_lower):
            matches.append(term)

    # longer terms first to prioritize specific phrases
    matches = sorted(matches, key=len, reverse=True)
    return matches[:max_terms]


def glossary_coverage_details(src: str, trg: str, target_lang: str, glossary: dict):
    """
    Returns:
    - matched_source_terms_count
    - preserved_terms_count
    - matched_terms
    - missing_terms
    """
    if not glossary:
        return 0, 0, [], []

    relevant_terms = find_relevant_glossary_terms(src, glossary)
    if not relevant_terms:
        return 0, 0, [], []

    trg_norm = normalize_for_match(trg).lower()

    matched_terms = []
    missing_terms = []

    for source_term in relevant_terms:
        entry = glossary.get(source_term, {})
        if str(target_lang).lower() in ["si", "sinhala"]:
            expected_terms = entry.get("si_terms", [])
        elif str(target_lang).lower() in ["ta", "tamil"]:
            expected_terms = entry.get("ta_terms", [])
        else:
            expected_terms = []

        # if no expected target term exists, skip this glossary item from scoring
        if not expected_terms:
            continue

        found = False
        for expected in expected_terms:
            if expected and expected in trg_norm:
                found = True
                break

        if found:
            matched_terms.append(source_term)
        else:
            missing_terms.append(source_term)

    total_scored = len(matched_terms) + len(missing_terms)
    preserved = len(matched_terms)

    return total_scored, preserved, matched_terms, missing_terms


# =========================================================
# SCORING FUNCTIONS
# =========================================================
def score_completion(src: str, trg: str) -> float:
    src_wc = word_count(src)
    trg_wc = word_count(trg)
    if src_wc == 0:
        return 0.0

    ratio = trg_wc / src_wc

    if 0.55 <= ratio <= 1.8:
        base = 1.0
    elif 0.40 <= ratio < 0.55 or 1.8 < ratio <= 2.2:
        base = 0.75
    elif 0.25 <= ratio < 0.40 or 2.2 < ratio <= 2.8:
        base = 0.45
    else:
        base = 0.15

    if trg_wc < max(4, int(src_wc * 0.35)):
        base -= 0.25

    return clamp01(base)


def score_grammar(trg: str) -> float:
    if not trg.strip():
        return 0.0

    score = 1.0

    if re.search(r"[.]{3,}|[,]{3,}|[?]{2,}|[!]{2,}", trg):
        score -= 0.15

    if re.search(r"\s{3,}", trg):
        score -= 0.10

    noise_chars = len(re.findall(r"[_\|\{\}\[\]<>~`]", trg))
    density = noise_chars / max(len(trg), 1)
    if density > 0.03:
        score -= 0.25
    elif density > 0.01:
        score -= 0.12

    if re.search(r"[,;:\-\(\[]\s*$", trg):
        score -= 0.18

    if word_count(trg) < 3:
        score -= 0.35

    return clamp01(score)


def score_terminology_heuristic(src: str, trg: str) -> float:
    src_terms = extract_medical_terms(src)
    if not src_terms:
        return 1.0

    trg_l = trg.lower()
    preserved_signals = 0

    for term in src_terms:
        if term in trg_l:
            preserved_signals += 1

    ratio = preserved_signals / len(src_terms)

    if ratio >= 0.75:
        return 1.0
    if ratio >= 0.40:
        return 0.8
    if ratio >= 0.15:
        return 0.6

    return 0.5


def score_terminology(src: str, trg: str, target_lang: str, glossary: dict):
    """
    Uses glossary if possible.
    Falls back to heuristic medical term preservation if no glossary terms matched.
    """
    total_scored, preserved, matched_terms, missing_terms = glossary_coverage_details(
        src, trg, target_lang, glossary
    )

    if total_scored > 0:
        ratio = preserved / total_scored

        if ratio >= 0.95:
            score = 1.0
        elif ratio >= 0.80:
            score = 0.9
        elif ratio >= 0.60:
            score = 0.75
        elif ratio >= 0.40:
            score = 0.6
        else:
            score = 0.35

        return score, total_scored, preserved, matched_terms, missing_terms, "glossary"

    # fallback
    fallback_score = score_terminology_heuristic(src, trg)
    return fallback_score, 0, 0, [], [], "heuristic"


def score_context_preservation(src: str, trg: str) -> float:
    if not src.strip() or not trg.strip():
        return 0.0

    src_num = re.findall(r"\b\d+(?:\.\d+)?\b", src)
    trg_num = re.findall(r"\b\d+(?:\.\d+)?\b", trg)

    number_score = 1.0
    if src_num:
        overlap = sum(1 for n in src_num if n in trg_num)
        number_score = overlap / len(src_num)

    src_lines = [x.strip() for x in src.splitlines() if x.strip()]
    trg_lines = [x.strip() for x in trg.splitlines() if x.strip()]
    line_score = 1.0
    if src_lines:
        ratio = len(trg_lines) / max(len(src_lines), 1)
        if 0.5 <= ratio <= 2.0:
            line_score = 1.0
        elif 0.3 <= ratio <= 2.5:
            line_score = 0.8
        else:
            line_score = 0.55

    length_score = score_completion(src, trg)

    return clamp01((0.45 * number_score) + (0.20 * line_score) + (0.35 * length_score))


def score_chunking_effectiveness(prev_src: str, src: str, next_src: str) -> float:
    score = 1.0

    if begins_mid_sentence(src):
        score -= 0.18

    if src and not ends_sentence(src) and next_src:
        score -= 0.12

    if prev_src:
        overlap_len = longest_suffix_prefix_overlap(prev_src, src)
        if overlap_len == 0:
            sim = repeated_phrase_score(prev_src, src)
            if sim < 0.18:
                score -= 0.20
        else:
            score += 0.02

    if word_count(src) < 8:
        score -= 0.20

    return clamp01(score)


def overall_chunk_score(row) -> float:
    weights = {
        "completion_score": 0.24,
        "grammar_score": 0.18,
        "terminology_score": 0.18,
        "context_score": 0.24,
        "chunking_score": 0.16,
    }
    total = 0.0
    for k, w in weights.items():
        total += row[k] * w
    return round(total, 4)


def rating_5(score_01: float) -> float:
    return round(score_01 * 5, 2)


# =========================================================
# MAIN
# =========================================================
def main():
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    glossary = load_glossary(GLOSSARY_JSON)

    df = pd.read_csv(input_path)

    required = {DOC_COL, CHUNK_COL, LANG_COL, SOURCE_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().isin(["success", "completed", "ok"])].copy()

    df[CHUNK_COL] = pd.to_numeric(df[CHUNK_COL], errors="coerce")
    df = df.dropna(subset=[DOC_COL, CHUNK_COL, LANG_COL]).copy()
    df[CHUNK_COL] = df[CHUNK_COL].astype(int)

    df[SOURCE_COL] = df[SOURCE_COL].fillna("").map(clean_text)
    df[TARGET_COL] = df[TARGET_COL].fillna("").map(clean_text)

    df = df.sort_values([DOC_COL, LANG_COL, CHUNK_COL]).reset_index(drop=True)

    rows = []

    grouped = df.groupby([DOC_COL, LANG_COL], sort=False)
    for (doc_id, lang), group in grouped:
        group = group.sort_values(CHUNK_COL).reset_index(drop=True)

        for i, r in group.iterrows():
            prev_src = group.loc[i - 1, SOURCE_COL] if i > 0 else ""
            next_src = group.loc[i + 1, SOURCE_COL] if i < len(group) - 1 else ""

            src = r[SOURCE_COL]
            trg = r[TARGET_COL]

            completion = score_completion(src, trg)
            grammar = score_grammar(trg)

            terminology, glossary_terms_scored, glossary_terms_preserved, glossary_matched_terms, glossary_missing_terms, terminology_mode = score_terminology(
                src, trg, lang, glossary
            )

            context = score_context_preservation(src, trg)
            chunking = score_chunking_effectiveness(prev_src, src, next_src)

            overlap_with_prev = longest_suffix_prefix_overlap(prev_src, src) if prev_src else 0
            overlap_sim_with_prev = repeated_phrase_score(prev_src, src) if prev_src else 0.0

            out = dict(r)
            out["source_word_count"] = word_count(src)
            out["target_word_count"] = word_count(trg)
            out["char_ratio"] = round(char_ratio(src, trg), 4)
            out["medical_terms_in_source"] = count_medical_terms(src)
            out["begins_mid_sentence_flag"] = begins_mid_sentence(src)
            out["ends_incomplete_flag"] = bool(src) and not ends_sentence(src)
            out["overlap_chars_with_prev"] = overlap_with_prev
            out["overlap_similarity_with_prev"] = round(overlap_sim_with_prev, 4)

            out["terminology_mode"] = terminology_mode
            out["glossary_terms_scored"] = glossary_terms_scored
            out["glossary_terms_preserved"] = glossary_terms_preserved
            out["glossary_matched_terms"] = " | ".join(glossary_matched_terms)
            out["glossary_missing_terms"] = " | ".join(glossary_missing_terms)

            out["completion_score"] = round(completion, 4)
            out["grammar_score"] = round(grammar, 4)
            out["terminology_score"] = round(terminology, 4)
            out["context_score"] = round(context, 4)
            out["chunking_score"] = round(chunking, 4)
            out["overall_score"] = overall_chunk_score({
                "completion_score": completion,
                "grammar_score": grammar,
                "terminology_score": terminology,
                "context_score": context,
                "chunking_score": chunking,
            })
            out["overall_rating_5"] = rating_5(out["overall_score"])

            reasons = []
            if completion < 0.6:
                reasons.append("possible_incomplete_translation")
            if grammar < 0.65:
                reasons.append("possible_grammar_or_format_issue")
            if terminology < 0.65:
                reasons.append("possible_medical_term_loss")
            if context < 0.65:
                reasons.append("possible_context_loss")
            if chunking < 0.65:
                reasons.append("weak_chunk_boundary")
            out["flag_reasons"] = ";".join(reasons)

            rows.append(out)

    scored_df = pd.DataFrame(rows)

    doc_summary = (
        scored_df.groupby([DOC_COL, LANG_COL], as_index=False)[
            ["completion_score", "grammar_score", "terminology_score", "context_score", "chunking_score", "overall_score"]
        ]
        .mean()
    )
    doc_summary["overall_rating_5"] = doc_summary["overall_score"].map(rating_5)

    lang_summary = (
        scored_df.groupby([LANG_COL], as_index=False)[
            ["completion_score", "grammar_score", "terminology_score", "context_score", "chunking_score", "overall_score"]
        ]
        .mean()
    )
    lang_summary["overall_rating_5"] = lang_summary["overall_score"].map(rating_5)

    flagged = scored_df[scored_df["flag_reasons"].astype(str).str.len() > 0].copy()

    scored_df.to_csv(OUTPUT_ROW_SCORES_CSV, index=False, encoding="utf-8-sig")
    doc_summary.to_csv(OUTPUT_DOC_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    lang_summary.to_csv(OUTPUT_LANG_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    flagged.to_csv(OUTPUT_FLAGGED_CSV, index=False, encoding="utf-8-sig")

    print("Saved:")
    print(f"  Row scores:    {OUTPUT_ROW_SCORES_CSV}")
    print(f"  Doc summary:   {OUTPUT_DOC_SUMMARY_CSV}")
    print(f"  Lang summary:  {OUTPUT_LANG_SUMMARY_CSV}")
    print(f"  Flagged rows:  {OUTPUT_FLAGGED_CSV}")
    print()
    print("Language summary:")
    print(lang_summary.to_string(index=False))


if __name__ == "__main__":
    main()