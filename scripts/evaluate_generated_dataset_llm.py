import os
import json
import re
import math
import random
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

DEFAULT_DATASET_PATH = r"D:\SL_Medical_Corpus\generated_dataset_full\sharegpt_dataset.json"
DEFAULT_SOURCE_CSV = "C:\\Users\\Charunya\\Downloads\\sl_med_corpus_rows (1).csv"
DEFAULT_OUTPUT_DIR = r"D:\SL_Medical_Corpus\data\10_instruction_eval"
DEFAULT_MAX_WORKERS = 5
DEFAULT_MAX_RETRIES = 3
DEFAULT_SAMPLE_PER_GROUP = 20
DEFAULT_RANDOM_SEED = 42

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment.")

if not OPENAI_MODEL:
    raise ValueError("OPENAI_MODEL is not set in the environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

REFUSAL_TEXT = "The provided document does not contain this information."

# ============================================================
# JUDGE PROMPT
# ============================================================

JUDGE_SYSTEM_PROMPT = """
You are evaluating a synthetic instruction-tuning example for a document-grounded medical dataset.

Your task:
Decide whether the generated example is grounded in the source document and whether it is high quality.

Evaluation principles:
- Use only the SOURCE DOCUMENT as evidence.
- Do not use outside medical knowledge.
- Treat unsupported additions as hallucinations.
- A good question should be realistic, meaningful, and understandable.
- A good answer should be relevant, concise, and supported by the source.
- If the answer should have refused because the source lacks the needed information, mark should_refuse = true.
- For non-English source documents (e.g. Sinhala, Tamil, mixed), judge whether the English question/answer still accurately reflects the source content.
- Do not penalize paraphrasing if it remains faithful to the source.

Return only valid JSON matching the schema.
""".strip()

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "groundedness": {
            "type": "string",
            "enum": ["fully_grounded", "partially_grounded", "not_grounded"]
        },
        "question_quality": {
            "type": "string",
            "enum": ["good", "borderline", "poor"]
        },
        "answer_quality": {
            "type": "string",
            "enum": ["good", "borderline", "poor"]
        },
        "should_refuse": {"type": "boolean"},
        "language_handling": {
            "type": "string",
            "enum": ["good", "uncertain", "poor"]
        },
        "hallucination": {"type": "boolean"},
        "nonsense_or_unrealistic": {"type": "boolean"},
        "notes": {"type": "string"}
    },
    "required": [
        "groundedness",
        "question_quality",
        "answer_quality",
        "should_refuse",
        "language_handling",
        "hallucination",
        "nonsense_or_unrealistic",
        "notes"
    ]
}

# ============================================================
# HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_source_lookup(csv_path: str, id_column: str, text_column: str, language_column: Optional[str]) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(csv_path)

    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found in CSV.")

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in CSV.")

    if language_column and language_column not in df.columns:
        raise ValueError(f"Language column '{language_column}' not found in CSV.")

    lookup = {}
    for _, row in df.iterrows():
        record_id = str(row[id_column])
        doc_text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        lang = str(row[language_column]) if language_column and pd.notna(row[language_column]) else "unknown"

        lookup[record_id] = {
            "document_text": doc_text,
            "language": lang
        }

    return lookup


def load_generated_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_valid_example(item: Dict[str, Any]) -> bool:
    if "conversations" not in item or not isinstance(item["conversations"], list):
        return False

    conv = item["conversations"]
    if len(conv) < 2:
        return False

    for turn in conv:
        if turn.get("from") not in {"human", "gpt"}:
            return False
        if not normalize_text(turn.get("value", "")):
            return False

    return True


def conversation_key(item: Dict[str, Any]) -> str:
    parts = []
    for turn in item["conversations"]:
        parts.append(f"{turn['from']}::{normalize_text(turn['value']).lower()}")
    return " || ".join(parts)


def structural_clean(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    original_count = len(dataset)

    valid = [x for x in dataset if is_valid_example(x)]
    invalid_removed = original_count - len(valid)

    seen = set()
    deduped = []
    duplicate_removed = 0

    for item in valid:
        key = conversation_key(item)
        if key in seen:
            duplicate_removed += 1
            continue
        seen.add(key)
        deduped.append(item)

    return {
        "cleaned_dataset": deduped,
        "cleaning_summary": {
            "original_examples": original_count,
            "after_validity_filter": len(valid),
            "invalid_removed": invalid_removed,
            "after_exact_deduplication": len(deduped),
            "duplicate_removed": duplicate_removed
        }
    }


def extract_question_answer(item: Dict[str, Any]) -> Dict[str, str]:
    conv = item["conversations"]
    question = conv[0]["value"] if conv else ""
    answer = conv[-1]["value"] if conv else ""
    return {"question": question, "answer": answer}


def judge_example(
    item: Dict[str, Any],
    source_lookup: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    record_id = str(item.get("source_record_id", ""))
    source_info = source_lookup.get(record_id, {"document_text": "", "language": "unknown"})
    doc_text = source_info["document_text"]
    doc_language = source_info["language"]

    qa = extract_question_answer(item)

    user_prompt = f"""
Evaluate this generated example against its source document.

SOURCE DOCUMENT LANGUAGE:
{doc_language}

EXAMPLE TYPE:
{item.get("example_type", "unknown")}

SOURCE RECORD ID:
{record_id}

SOURCE DOCUMENT:
{doc_text}

GENERATED EXAMPLE:
{json.dumps(item["conversations"], ensure_ascii=False, indent=2)}

Focus on:
- groundedness to the source
- hallucination / unsupported claims
- question realism / nonsense
- whether the answer should have refused
- whether multilingual handling seems faithful when the source is Sinhala / Tamil / mixed and the output is English
""".strip()

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "instruction_eval_judgment",
                "schema": JUDGE_SCHEMA,
                "strict": True
            }
        }
    )

    judgment = json.loads(response.output_text)

    return {
        "source_record_id": record_id,
        "source_row_index": item.get("source_row_index"),
        "example_type": item.get("example_type"),
        "doc_language": doc_language,
        "question": qa["question"],
        "answer": qa["answer"],
        "judgment": judgment,
        "conversations": item["conversations"]
    }


def process_with_retries(item, source_lookup, max_retries):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return {
                "success": True,
                "result": judge_example(item, source_lookup),
                "error": None
            }
        except Exception as e:
            last_error = f"Attempt {attempt}/{max_retries}: {str(e)}"

    return {
        "success": False,
        "result": None,
        "error": last_error
    }


def rate_to_score(label: str, mapping: Dict[str, int]) -> int:
    return mapping.get(label, -1)


def compute_summary(judged_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not judged_rows:
        return {}

    groundedness_counts = Counter()
    question_quality_counts = Counter()
    answer_quality_counts = Counter()
    language_handling_counts = Counter()
    hallucination_counts = Counter()
    nonsense_counts = Counter()
    should_refuse_counts = Counter()
    example_type_counts = Counter()
    language_counts = Counter()

    per_language = defaultdict(list)
    per_type = defaultdict(list)

    grounded_map = {
        "fully_grounded": 2,
        "partially_grounded": 1,
        "not_grounded": 0
    }
    quality_map = {
        "good": 2,
        "borderline": 1,
        "poor": 0
    }
    lang_map = {
        "good": 2,
        "uncertain": 1,
        "poor": 0
    }

    for row in judged_rows:
        j = row["judgment"]
        lang = row["doc_language"]
        ex_type = row["example_type"]

        groundedness_counts[j["groundedness"]] += 1
        question_quality_counts[j["question_quality"]] += 1
        answer_quality_counts[j["answer_quality"]] += 1
        language_handling_counts[j["language_handling"]] += 1
        hallucination_counts[str(j["hallucination"])] += 1
        nonsense_counts[str(j["nonsense_or_unrealistic"])] += 1
        should_refuse_counts[str(j["should_refuse"])] += 1
        example_type_counts[ex_type] += 1
        language_counts[lang] += 1

        per_language[lang].append(row)
        per_type[ex_type].append(row)

    def avg_score(rows, field, mapping):
        vals = [rate_to_score(r["judgment"][field], mapping) for r in rows]
        vals = [v for v in vals if v >= 0]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "total_judged_examples": len(judged_rows),
        "groundedness_counts": dict(groundedness_counts),
        "question_quality_counts": dict(question_quality_counts),
        "answer_quality_counts": dict(answer_quality_counts),
        "language_handling_counts": dict(language_handling_counts),
        "hallucination_counts": dict(hallucination_counts),
        "nonsense_or_unrealistic_counts": dict(nonsense_counts),
        "should_refuse_counts": dict(should_refuse_counts),
        "example_type_counts": dict(example_type_counts),
        "doc_language_counts": dict(language_counts),

        "rates": {
            "fully_grounded_rate": round(groundedness_counts["fully_grounded"] / len(judged_rows), 4),
            "partially_grounded_rate": round(groundedness_counts["partially_grounded"] / len(judged_rows), 4),
            "not_grounded_rate": round(groundedness_counts["not_grounded"] / len(judged_rows), 4),
            "hallucination_rate": round(hallucination_counts["True"] / len(judged_rows), 4),
            "nonsense_or_unrealistic_rate": round(nonsense_counts["True"] / len(judged_rows), 4),
            "should_refuse_rate": round(should_refuse_counts["True"] / len(judged_rows), 4),
        },

        "average_scores_overall": {
            "groundedness_score": avg_score(judged_rows, "groundedness", grounded_map),
            "question_quality_score": avg_score(judged_rows, "question_quality", quality_map),
            "answer_quality_score": avg_score(judged_rows, "answer_quality", quality_map),
            "language_handling_score": avg_score(judged_rows, "language_handling", lang_map),
        },

        "per_language_summary": {},
        "per_example_type_summary": {}
    }

    for lang, rows in per_language.items():
        summary["per_language_summary"][lang] = {
            "count": len(rows),
            "fully_grounded_rate": round(sum(r["judgment"]["groundedness"] == "fully_grounded" for r in rows) / len(rows), 4),
            "hallucination_rate": round(sum(r["judgment"]["hallucination"] for r in rows) / len(rows), 4),
            "should_refuse_rate": round(sum(r["judgment"]["should_refuse"] for r in rows) / len(rows), 4),
            "nonsense_rate": round(sum(r["judgment"]["nonsense_or_unrealistic"] for r in rows) / len(rows), 4),
            "groundedness_score": avg_score(rows, "groundedness", grounded_map),
            "question_quality_score": avg_score(rows, "question_quality", quality_map),
            "answer_quality_score": avg_score(rows, "answer_quality", quality_map),
            "language_handling_score": avg_score(rows, "language_handling", lang_map),
        }

    for ex_type, rows in per_type.items():
        summary["per_example_type_summary"][ex_type] = {
            "count": len(rows),
            "fully_grounded_rate": round(sum(r["judgment"]["groundedness"] == "fully_grounded" for r in rows) / len(rows), 4),
            "hallucination_rate": round(sum(r["judgment"]["hallucination"] for r in rows) / len(rows), 4),
            "should_refuse_rate": round(sum(r["judgment"]["should_refuse"] for r in rows) / len(rows), 4),
            "nonsense_rate": round(sum(r["judgment"]["nonsense_or_unrealistic"] for r in rows) / len(rows), 4),
        }

    return summary


def filter_high_quality_rows(judged_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Conservative filter:
    keep only rows that are fully grounded, non-hallucinatory, not nonsense,
    and at least borderline/good on both question and answer quality.
    """
    kept = []

    for row in judged_rows:
        j = row["judgment"]

        if j["groundedness"] != "fully_grounded":
            continue
        if j["hallucination"]:
            continue
        if j["nonsense_or_unrealistic"]:
            continue
        if j["question_quality"] == "poor":
            continue
        if j["answer_quality"] == "poor":
            continue
        if row["example_type"] == "single_qa_answerable" and j["should_refuse"]:
            continue
        if row["doc_language"] in {"si", "ta", "mixed"} and j["language_handling"] == "poor":
            continue

        kept.append(row)

    return kept


def convert_judged_rows_back_to_training_format(judged_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted = []
    for row in judged_rows:
        converted.append({
            "example_type": row["example_type"],
            "source_record_id": row["source_record_id"],
            "source_row_index": row["source_row_index"],
            "conversations": row["conversations"]
        })
    return converted


def export_review_sample(judged_rows: List[Dict[str, Any]], output_path: str, sample_per_group: int, seed: int):
    rng = random.Random(seed)
    buckets = defaultdict(list)

    for row in judged_rows:
        key = (row["doc_language"], row["example_type"])
        buckets[key].append(row)

    sample_rows = []
    for key, rows in buckets.items():
        rng.shuffle(rows)
        sample_rows.extend(rows[:sample_per_group])

    flat_rows = []
    for row in sample_rows:
        flat_rows.append({
            "source_record_id": row["source_record_id"],
            "doc_language": row["doc_language"],
            "example_type": row["example_type"],
            "question": row["question"],
            "answer": row["answer"],
            "groundedness": row["judgment"]["groundedness"],
            "question_quality": row["judgment"]["question_quality"],
            "answer_quality": row["judgment"]["answer_quality"],
            "should_refuse": row["judgment"]["should_refuse"],
            "language_handling": row["judgment"]["language_handling"],
            "hallucination": row["judgment"]["hallucination"],
            "nonsense_or_unrealistic": row["judgment"]["nonsense_or_unrealistic"],
            "notes": row["judgment"]["notes"]
        })

    pd.DataFrame(flat_rows).to_csv(output_path, index=False, encoding="utf-8-sig")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated instruction dataset with LLM-based groundedness judging.")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--source_csv", type=str, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--id_column", type=str, default="doc_id")
    parser.add_argument("--text_column", type=str, default="markdown")
    parser.add_argument("--language_column", type=str, default="language")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--sample_per_group", type=int, default=DEFAULT_SAMPLE_PER_GROUP)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a pilot run.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading generated dataset...")
    dataset = load_generated_dataset(args.dataset_path)

    print("Running structural cleaning...")
    cleaned = structural_clean(dataset)
    cleaned_dataset = cleaned["cleaned_dataset"]

    if args.limit is not None:
        cleaned_dataset = cleaned_dataset[:args.limit]

    print("Loading source lookup...")
    source_lookup = load_source_lookup(
        csv_path=args.source_csv,
        id_column=args.id_column,
        text_column=args.text_column,
        language_column=args.language_column
    )

    print(f"Judging {len(cleaned_dataset)} examples...")
    judged_rows = []
    failed_rows = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_with_retries, item, source_lookup, args.max_retries) for item in cleaned_dataset]

        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM judging"):
            result = future.result()
            if result["success"]:
                judged_rows.append(result["result"])
            else:
                failed_rows.append({"error": result["error"]})

    summary = compute_summary(judged_rows)
    high_quality_rows = filter_high_quality_rows(judged_rows)
    cleaned_training_dataset = convert_judged_rows_back_to_training_format(high_quality_rows)

    # paths
    cleaning_summary_path = os.path.join(args.output_dir, "cleaning_summary.json")
    judged_rows_path = os.path.join(args.output_dir, "judged_rows.json")
    failed_rows_path = os.path.join(args.output_dir, "judge_failures.json")
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    cleaned_dataset_path = os.path.join(args.output_dir, "instruction_dataset_evaluated_clean.json")
    review_sample_path = os.path.join(args.output_dir, "human_review_sample.csv")

    with open(cleaning_summary_path, "w", encoding="utf-8") as f:
        json.dump(cleaned["cleaning_summary"], f, indent=2, ensure_ascii=False)

    with open(judged_rows_path, "w", encoding="utf-8") as f:
        json.dump(judged_rows, f, indent=2, ensure_ascii=False)

    with open(failed_rows_path, "w", encoding="utf-8") as f:
        json.dump(failed_rows, f, indent=2, ensure_ascii=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(cleaned_dataset_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_training_dataset, f, indent=2, ensure_ascii=False)

    export_review_sample(
        judged_rows=judged_rows,
        output_path=review_sample_path,
        sample_per_group=args.sample_per_group,
        seed=args.seed
    )

    print("\nDone.")
    print("\nSaved files:")
    print(cleaning_summary_path)
    print(judged_rows_path)
    print(failed_rows_path)
    print(summary_path)
    print(cleaned_dataset_path)
    print(review_sample_path)

    print("\nHigh-level summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()