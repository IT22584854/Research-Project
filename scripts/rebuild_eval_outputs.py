import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

MERGED_JUDGED_PATH = r"D:\SL_Medical_Corpus\data\10_instruction_eval\judged_rows_merged.json"
OUT_DIR = Path(r"D:\SL_Medical_Corpus\data\10_instruction_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def rate_to_score(label, mapping):
    return mapping.get(label, -1)

def compute_summary(judged_rows):
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

    grounded_map = {"fully_grounded": 2, "partially_grounded": 1, "not_grounded": 0}
    quality_map = {"good": 2, "borderline": 1, "poor": 0}
    lang_map = {"good": 2, "uncertain": 1, "poor": 0}

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
            "fully_grounded_rate": round(groundedness_counts["fully_grounded"] / len(judged_rows), 4) if judged_rows else 0.0,
            "partially_grounded_rate": round(groundedness_counts["partially_grounded"] / len(judged_rows), 4) if judged_rows else 0.0,
            "not_grounded_rate": round(groundedness_counts["not_grounded"] / len(judged_rows), 4) if judged_rows else 0.0,
            "hallucination_rate": round(hallucination_counts["True"] / len(judged_rows), 4) if judged_rows else 0.0,
            "nonsense_or_unrealistic_rate": round(nonsense_counts["True"] / len(judged_rows), 4) if judged_rows else 0.0,
            "should_refuse_rate": round(should_refuse_counts["True"] / len(judged_rows), 4) if judged_rows else 0.0,
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

def filter_high_quality_rows(judged_rows):
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

        kept.append({
            "example_type": row["example_type"],
            "source_record_id": row["source_record_id"],
            "source_row_index": row["source_row_index"],
            "conversations": row["conversations"]
        })
    return kept

def export_review_sample(judged_rows, output_path, sample_per_group=20, seed=42):
    rng = random.Random(seed)
    buckets = defaultdict(list)

    for row in judged_rows:
        key = (row["doc_language"], row["example_type"])
        buckets[key].append(row)

    sample_rows = []
    for _, rows in buckets.items():
        rng.shuffle(rows)
        sample_rows.extend(rows[:sample_per_group])

    flat = []
    for row in sample_rows:
        flat.append({
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

    pd.DataFrame(flat).to_csv(output_path, index=False, encoding="utf-8-sig")

def main():
    with open(MERGED_JUDGED_PATH, "r", encoding="utf-8") as f:
        judged_rows = json.load(f)

    summary = compute_summary(judged_rows)
    cleaned_dataset = filter_high_quality_rows(judged_rows)

    summary_path = OUT_DIR / "evaluation_summary.json"
    cleaned_path = OUT_DIR / "instruction_dataset_evaluated_clean.json"
    review_path = OUT_DIR / "human_review_sample.csv"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(cleaned_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_dataset, f, indent=2, ensure_ascii=False)

    export_review_sample(judged_rows, review_path)

    print(f"Judged rows: {len(judged_rows)}")
    print(f"Cleaned dataset: {len(cleaned_dataset)}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {cleaned_path}")
    print(f"Saved: {review_path}")

if __name__ == "__main__":
    main()