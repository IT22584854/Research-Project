import json
import pandas as pd

# =========================================================
# CONFIGURATION
# =========================================================
DATA_PATH = r"C:\Users\Charunya\Desktop\medical_translation_chunks_TEST100_with_chatgpt_nllb.csv"
GLOSSARY_PATH = r"C:\Users\Charunya\Desktop\medical_glossary.json"
OUTPUT_CSV = r"C:\Users\Charunya\Desktop\glossary_precision_recall_f1_results.csv"

SOURCE_COL = "source_text"
LANG_COL = "target_lang"

MODELS = {
    "Gemini 2.5 Flash": "translated_text",
    "ChatGPT": "chatgpt_translation",
    "NLLB": "nllb_translation",
}


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
    glossary = json.load(f)


# =========================================================
# HELPERS
# =========================================================
def normalize_lang(lang):
    lang = str(lang).strip().lower()

    if lang in ["si", "sinhala", "sinhalese"]:
        return "sinhala"

    if lang in ["ta", "tamil"]:
        return "tamil"

    return lang


def get_expected_terms(source_text, target_lang):
    """
    Finds glossary terms that appear in the English source text and returns
    their expected Sinhala or Tamil translations.
    """
    source_text = str(source_text).lower()
    target_lang = normalize_lang(target_lang)

    expected = []

    for english_term, translations in glossary.items():
        english_term_lower = str(english_term).lower()

        if english_term_lower in source_text:
            if target_lang == "sinhala":
                term = translations.get("si", "")
            elif target_lang == "tamil":
                term = translations.get("ta", "")
            else:
                term = ""

            if term:
                expected.append(str(term).strip())

    return expected


def get_all_possible_terms_for_lang(target_lang):
    """
    Returns all glossary terms for the selected target language.
    Used for detecting extra glossary terms as false positives.
    """
    target_lang = normalize_lang(target_lang)
    terms = []

    for _, translations in glossary.items():
        if target_lang == "sinhala":
            term = translations.get("si", "")
        elif target_lang == "tamil":
            term = translations.get("ta", "")
        else:
            term = ""

        if term:
            terms.append(str(term).strip())

    return terms


def evaluate_model(model_name, translation_col):
    """
    Computes glossary-based Precision, Recall, and F1 for one model.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    evaluated_rows = 0
    rows_with_expected_terms = 0

    for _, row in df.iterrows():
        source_text = row.get(SOURCE_COL, "")
        target_lang = normalize_lang(row.get(LANG_COL, ""))
        translation = str(row.get(translation_col, ""))

        if target_lang not in ["sinhala", "tamil"]:
            continue

        evaluated_rows += 1

        expected_terms = get_expected_terms(source_text, target_lang)

        if expected_terms:
            rows_with_expected_terms += 1

        expected_set = set(expected_terms)
        possible_terms = get_all_possible_terms_for_lang(target_lang)

        # True positives and false negatives
        for term in expected_set:
            if term in translation:
                total_tp += 1
            else:
                total_fn += 1

        # False positives: glossary terms found in translation but not expected from source
        for possible_term in possible_terms:
            if possible_term in translation and possible_term not in expected_set:
                total_fp += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "Model": model_name,
        "Evaluated_Rows": evaluated_rows,
        "Rows_With_Expected_Glossary_Terms": rows_with_expected_terms,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
    }


# =========================================================
# MAIN
# =========================================================
def main():
    print("Columns found:")
    print(df.columns.tolist())

    missing_cols = [col for col in [SOURCE_COL, LANG_COL] if col not in df.columns]
    for model_name, col in MODELS.items():
        if col not in df.columns:
            missing_cols.append(col)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    results = []

    for model_name, translation_col in MODELS.items():
        print(f"\n[INFO] Evaluating {model_name} using column: {translation_col}")
        result = evaluate_model(model_name, translation_col)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n[DONE] Glossary-based evaluation saved to:")
    print(OUTPUT_CSV)
    print("\nFinal results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()