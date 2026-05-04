import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============================================================
# CONFIGURATION
# ============================================================

PRIMARY_CORPUS = r"C:\Users\Charunya\Desktop\final_unique_medical_corpus.csv"
FALLBACK_CORPUS = r"C:\Users\Charunya\Downloads\sl_med_corpus_rows (4).csv"
TEST_JSONL = r"C:\Users\Charunya\Downloads\test.jsonl.txt"

OUTPUT_CSV = r"C:\Users\Charunya\Desktop\english_test_qa_llm_judge_results.csv"
SUMMARY_CSV = r"C:\Users\Charunya\Desktop\english_test_qa_llm_judge_summary.csv"

DOC_ID_COL = "doc_id"
PRIMARY_TEXT_COL = "clean_text"
FALLBACK_TEXT_COL = "markdown"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JUDGE_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MAX_WORKERS = 5
LIMIT = None  # Set to 20 for quick testing

REFUSAL_TEXT = "The provided document does not contain this information."

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in your environment.")

client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# LOAD FILES
# ============================================================

def read_csv_with_fallback(path):
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read CSV: {path}")


def load_test_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


# ============================================================
# DOCUMENT CACHE
# ============================================================

def build_doc_lookup(primary_df, fallback_df):
    """
    Builds fast dictionary lookups so repeated doc_id searches are quick.
    """
    primary_lookup = {}
    fallback_lookup = {}

    if DOC_ID_COL not in primary_df.columns:
        raise ValueError(f"Primary corpus missing column: {DOC_ID_COL}")

    if PRIMARY_TEXT_COL not in primary_df.columns:
        raise ValueError(f"Primary corpus missing column: {PRIMARY_TEXT_COL}")

    if DOC_ID_COL not in fallback_df.columns:
        raise ValueError(f"Fallback corpus missing column: {DOC_ID_COL}")

    if FALLBACK_TEXT_COL not in fallback_df.columns:
        raise ValueError(f"Fallback corpus missing column: {FALLBACK_TEXT_COL}")

    primary_df[DOC_ID_COL] = primary_df[DOC_ID_COL].astype(str)
    fallback_df[DOC_ID_COL] = fallback_df[DOC_ID_COL].astype(str)

    for _, row in primary_df.iterrows():
        doc_id = str(row[DOC_ID_COL])
        text = str(row.get(PRIMARY_TEXT_COL, "")).strip()

        if text and text.lower() != "nan":
            primary_lookup[doc_id] = text

    for _, row in fallback_df.iterrows():
        doc_id = str(row[DOC_ID_COL])
        text = str(row.get(FALLBACK_TEXT_COL, "")).strip()

        if text and text.lower() != "nan":
            fallback_lookup[doc_id] = text

    return primary_lookup, fallback_lookup


def build_context_map(test_records, primary_lookup, fallback_lookup):
    """
    Creates a cache of source_record_id -> document context.
    First uses primary clean_text, then fallback markdown.
    """
    context_map = {}
    missing_doc_ids = []

    for record in test_records:
        doc_id = str(record.get("source_record_id", ""))

        if doc_id in context_map:
            continue

        context = primary_lookup.get(doc_id, "")

        if not context:
            context = fallback_lookup.get(doc_id, "")

        if not context:
            missing_doc_ids.append(doc_id)

        context_map[doc_id] = context

    return context_map, missing_doc_ids


# ============================================================
# QA EXTRACTION
# ============================================================

def extract_qa_pairs(record):
    """
    Extracts one or more QA pairs from ShareGPT-style records.
    Supports single-turn and multi-turn records.
    """
    example_type = record.get("example_type", "")
    conversations = record.get("conversations", [])

    qa_pairs = []

    if example_type in ["single_qa_answerable", "single_qa_refusal"]:
        if len(conversations) >= 2:
            qa_pairs.append({
                "question": conversations[0].get("value", ""),
                "answer": conversations[1].get("value", ""),
                "turn_index": 0,
            })

        return qa_pairs

    if example_type == "multi_turn":
        turn_index = 0

        for i in range(0, len(conversations) - 1, 2):
            user_turn = conversations[i]
            assistant_turn = conversations[i + 1]

            if user_turn.get("from") == "human" and assistant_turn.get("from") == "gpt":
                qa_pairs.append({
                    "question": user_turn.get("value", ""),
                    "answer": assistant_turn.get("value", ""),
                    "turn_index": turn_index,
                })
                turn_index += 1

        return qa_pairs

    return qa_pairs


def build_judge_items(test_records, context_map):
    """
    Converts dataset records into individual judge items.
    Multi-turn records become multiple QA items.
    """
    items = []

    for example_index, record in enumerate(test_records):
        doc_id = str(record.get("source_record_id", ""))
        context = context_map.get(doc_id, "")
        example_type = record.get("example_type", "")

        qa_pairs = extract_qa_pairs(record)

        for qa in qa_pairs:
            items.append({
                "example_index": example_index,
                "turn_index": qa["turn_index"],
                "source_record_id": doc_id,
                "source_row_index": record.get("source_row_index", ""),
                "example_type": example_type,
                "question": qa["question"],
                "answer": qa["answer"],
                "context": context,
                "context_found": bool(context),
                "context_length": len(context),
            })

    return items


# ============================================================
# LLM-AS-JUDGE
# ============================================================

def build_judge_prompt(item):
    return f"""
You are evaluating a document-grounded English QA dataset.

Judge the QUESTION and ANSWER using only the DOCUMENT CONTEXT.

Return JSON only.

Scores:
- groundedness: 0, 1, or 2
  2 = fully supported by the document
  1 = partially supported or vague
  0 = unsupported or hallucinated

- answer_correctness: 0, 1, or 2
  2 = correct and complete
  1 = partially correct
  0 = incorrect

- question_quality: 0, 1, or 2
  2 = clear, useful, answerable/refusable as intended
  1 = somewhat unclear or weak
  0 = poor or unrelated

- answer_quality: 0, 1, or 2
  2 = concise, clear, useful
  1 = acceptable but weak
  0 = poor

Also return:
- hallucination: true or false
- refusal_correct: true, false, or null

For refusal examples, the answer should be exactly:
"{REFUSAL_TEXT}"

Example type:
{item["example_type"]}

DOCUMENT CONTEXT:
{item["context"]}

QUESTION:
{item["question"]}

ANSWER:
{item["answer"]}

Return JSON with this schema:
{{
  "groundedness": 0,
  "answer_correctness": 0,
  "question_quality": 0,
  "answer_quality": 0,
  "hallucination": false,
  "refusal_correct": null,
  "reason": "short explanation"
}}
""".strip()


def judge_one(item):
    """
    Sends one QA item to the LLM judge.
    """
    if not item["context_found"]:
        return {
            **item,
            "groundedness": None,
            "answer_correctness": None,
            "question_quality": None,
            "answer_quality": None,
            "hallucination": None,
            "refusal_correct": None,
            "reason": "Context not found for source_record_id.",
        }

    prompt = build_judge_prompt(item)

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict evaluator for document-grounded QA datasets. Return valid JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        text = response.choices[0].message.content.strip()

        try:
            result = json.loads(text)
        except Exception:
            result = {
                "groundedness": None,
                "answer_correctness": None,
                "question_quality": None,
                "answer_quality": None,
                "hallucination": None,
                "refusal_correct": None,
                "reason": f"Invalid judge JSON: {text[:300]}",
            }

        return {**item, **result}

    except Exception as e:
        return {
            **item,
            "groundedness": None,
            "answer_correctness": None,
            "question_quality": None,
            "answer_quality": None,
            "hallucination": None,
            "refusal_correct": None,
            "reason": f"Judge API error: {str(e)}",
        }


# ============================================================
# SUMMARY
# ============================================================

def create_summary(results_df):
    valid_df = results_df.dropna(subset=["groundedness"])

    if valid_df.empty:
        return pd.DataFrame([{
            "total_items": len(results_df),
            "valid_judged_items": 0,
            "missing_context_items": (~results_df["context_found"]).sum(),
            "mean_groundedness": None,
            "mean_answer_correctness": None,
            "mean_question_quality": None,
            "mean_answer_quality": None,
            "hallucination_rate": None,
            "fully_grounded_rate": None,
            "correct_answer_rate": None,
            "refusal_correct_rate": None,
        }])

    refusal_df = valid_df[valid_df["example_type"] == "single_qa_refusal"]

    return pd.DataFrame([{
        "total_items": len(results_df),
        "valid_judged_items": len(valid_df),
        "missing_context_items": (~results_df["context_found"]).sum(),
        "mean_groundedness": valid_df["groundedness"].mean(),
        "mean_answer_correctness": valid_df["answer_correctness"].mean(),
        "mean_question_quality": valid_df["question_quality"].mean(),
        "mean_answer_quality": valid_df["answer_quality"].mean(),
        "hallucination_rate": valid_df["hallucination"].fillna(False).mean(),
        "fully_grounded_rate": (valid_df["groundedness"] == 2).mean(),
        "correct_answer_rate": (valid_df["answer_correctness"] == 2).mean(),
        "refusal_correct_rate": refusal_df["refusal_correct"].fillna(False).mean() if not refusal_df.empty else None,
    }])


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--primary_corpus", default=PRIMARY_CORPUS)
    parser.add_argument("--fallback_corpus", default=FALLBACK_CORPUS)
    parser.add_argument("--test_jsonl", default=TEST_JSONL)
    parser.add_argument("--output_csv", default=OUTPUT_CSV)
    parser.add_argument("--summary_csv", default=SUMMARY_CSV)
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=LIMIT)

    args = parser.parse_args()

    print("[INFO] Loading corpus files...")
    primary_df = read_csv_with_fallback(args.primary_corpus)
    fallback_df = read_csv_with_fallback(args.fallback_corpus)

    print("[INFO] Loading test records...")
    test_records = load_test_jsonl(args.test_jsonl)

    if args.limit:
        test_records = test_records[:args.limit]

    print(f"[INFO] Test records loaded: {len(test_records)}")

    print("[INFO] Building document lookup...")
    primary_lookup, fallback_lookup = build_doc_lookup(primary_df, fallback_df)

    print("[INFO] Building context cache...")
    context_map, missing_doc_ids = build_context_map(
        test_records=test_records,
        primary_lookup=primary_lookup,
        fallback_lookup=fallback_lookup,
    )

    print(f"[INFO] Unique documents needed: {len(context_map)}")
    print(f"[INFO] Missing document contexts: {len(missing_doc_ids)}")

    if missing_doc_ids:
        missing_path = args.output_csv.replace(".csv", "_missing_doc_ids.txt")

        with open(missing_path, "w", encoding="utf-8") as f:
            for doc_id in missing_doc_ids:
                f.write(str(doc_id) + "\n")

        print(f"[WARN] Missing doc IDs saved to: {missing_path}")

    judge_items = build_judge_items(test_records, context_map)

    print(f"[INFO] Total QA turns to judge: {len(judge_items)}")
    print(f"[INFO] Judge model: {JUDGE_MODEL}")
    print(f"[INFO] Parallel workers: {args.max_workers}")

    results = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(judge_one, item) for item in judge_items]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging QA"):
            results.append(future.result())

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    summary_df = create_summary(results_df)
    summary_df.to_csv(args.summary_csv, index=False, encoding="utf-8-sig")

    print("\n[DONE] Evaluation complete.")
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print(f"\nDetailed results saved to: {args.output_csv}")
    print(f"Summary saved to: {args.summary_csv}")


if __name__ == "__main__":
    main()