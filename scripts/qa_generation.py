import os
import json
import math
import random
import argparse
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

DEFAULT_TOTAL_EXAMPLES = 5600
DEFAULT_SINGLE_RATIO = 0.70
DEFAULT_MULTI_RATIO = 0.25
DEFAULT_REFUSAL_RATIO = 0.05
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_WORKERS = 5
DEFAULT_CSV_PATH = r"C:\Users\Charunya\Downloads\sl_med_corpus_rows (1).csv"

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment.")

if not OPENAI_MODEL:
    raise ValueError("OPENAI_MODEL is not set in the environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

REFUSAL_TEXT = "The provided document does not contain this information."

# ============================================================
# PROMPTS
# ============================================================

BASE_SYSTEM_PROMPT = """
You are generating document-grounded conversational training data from the DOCUMENT CONTEXT.

You must follow these rules exactly:
- Use only information explicitly supported by the DOCUMENT CONTEXT.
- Do not use external knowledge.
- Do not invent missing details.
- Keep all questions and answers in English.
- Make the examples realistic, diverse, and conversational.
- Answers must be concise, grounded, and synthesized from the document.
- Avoid copying long spans from the document.
- Avoid duplicate questions or repetitive patterns.
- Prefer practical and natural user phrasing.

For answerable QA:
- Write questions that CAN be answered directly from the document.
- The answer must stay fully grounded in the document.

For refusal QA:
- Write realistic questions that a user might ask but that CANNOT be answered from the document.
- The answer must be exactly:
  "The provided document does not contain this information."

For multi-turn dialogues:
- The conversation must stay grounded in the document.
- The second user turn must be a natural follow-up to the previous assistant answer.
- Include clarification, procedural detail, exceptions, or conditional reasoning only if supported by the document.
- Do not invent missing information.
""".strip()


def build_user_prompt(
    document_text: str,
    answerable_single_count: int,
    refusal_single_count: int,
    include_multi_turn: bool
) -> str:
    return f"""
Generate training data from the DOCUMENT CONTEXT using the exact counts below.

Required output:
- answerable_direct_qa: exactly {answerable_single_count} items
- refusal_direct_qa: exactly {refusal_single_count} items
- multi_turn_dialogues: exactly {1 if include_multi_turn else 0} items

Additional rules:
- For answerable_direct_qa, every answer must be supported by the document.
- For refusal_direct_qa, every answer must be exactly:
  "{REFUSAL_TEXT}"
- For multi_turn_dialogues, each dialogue must have exactly 4 turns total in this role order:
  user, assistant, user, assistant
- If a requested category count is 0, return an empty array for that category.
- Return only valid JSON matching the required schema.

DOCUMENT CONTEXT:
{document_text}
""".strip()


# ============================================================
# JSON SCHEMA
# ============================================================

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answerable_direct_qa": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"}
                },
                "required": ["question", "answer"]
            }
        },
        "refusal_direct_qa": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"}
                },
                "required": ["question", "answer"]
            }
        },
        "multi_turn_dialogues": {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["user", "assistant"]
                        },
                        "content": {"type": "string"}
                    },
                    "required": ["role", "content"]
                }
            }
        }
    },
    "required": [
        "answerable_direct_qa",
        "refusal_direct_qa",
        "multi_turn_dialogues"
    ]
}


# ============================================================
# HELPERS
# ============================================================

def load_csv_records(
    csv_path: str,
    text_column: str,
    id_column: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in CSV columns: {list(df.columns)}")

    df = df.copy()
    df[text_column] = df[text_column].fillna("").astype(str).str.strip()
    df = df[df[text_column] != ""].reset_index(drop=True)

    if id_column and id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found in CSV columns: {list(df.columns)}")

    if limit is not None:
        df = df.head(limit).reset_index(drop=True)

    records = []
    for idx, row in df.iterrows():
        record_id = str(row[id_column]) if id_column else str(idx)
        records.append({
            "row_index": idx,
            "record_id": record_id,
            "text": row[text_column]
        })

    return records


def compute_targets(
    total_examples: int,
    single_ratio: float,
    multi_ratio: float,
    refusal_ratio: float
) -> Dict[str, int]:
    if not math.isclose(single_ratio + multi_ratio + refusal_ratio, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Ratios must sum to 1.0")

    multi_target = round(total_examples * multi_ratio)
    refusal_target = round(total_examples * refusal_ratio)
    answerable_target = total_examples - multi_target - refusal_target

    return {
        "answerable_target": answerable_target,
        "multi_target": multi_target,
        "refusal_target": refusal_target,
        "single_total_target": answerable_target + refusal_target,
        "total_examples": total_examples
    }


def auto_scale_total_examples(
    requested_total_examples: int,
    n_docs: int,
    multi_ratio: float
) -> int:
    """
    At most 1 multi-turn dialogue per document.
    So multi_target = round(total * multi_ratio) must be <= n_docs.
    With 25% multi ratio, max feasible total is about 4 * n_docs.
    """
    if n_docs <= 0:
        raise ValueError("No documents available.")

    max_feasible = math.floor(n_docs / multi_ratio)
    adjusted_total = min(requested_total_examples, max_feasible)

    # Keep it at least n_docs so every doc can still receive at least one single item.
    adjusted_total = max(adjusted_total, n_docs)
    return adjusted_total


def distribute_single_turns(
    records: List[Dict[str, Any]],
    single_total_target: int,
    refusal_target: int,
    seed: int
) -> List[Dict[str, Any]]:
    n = len(records)
    if n == 0:
        raise ValueError("No records available after CSV loading/filtering.")

    if single_total_target < n:
        raise ValueError(
            "single_total_target is smaller than the number of documents. "
            "This would leave some documents with zero single-turn examples."
        )

    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)

    base = single_total_target // n
    remainder = single_total_target % n

    plan = []
    for rec in records:
        plan.append({
            **rec,
            "answerable_single_count": 0,
            "refusal_single_count": 0,
            "include_multi_turn": False,
            "single_total_count": 0
        })

    for idx in range(n):
        plan[idx]["single_total_count"] = base

    for pos in range(remainder):
        idx = order[pos]
        plan[idx]["single_total_count"] += 1

    eligible_groups = {}
    for idx, item in enumerate(plan):
        eligible_groups.setdefault(item["single_total_count"], []).append(idx)

    sorted_counts = sorted(eligible_groups.keys(), reverse=True)
    refusal_assigned = 0

    for count in sorted_counts:
        group = eligible_groups[count]
        rng.shuffle(group)
        for idx in group:
            if refusal_assigned >= refusal_target:
                break
            if plan[idx]["single_total_count"] >= 2:
                plan[idx]["refusal_single_count"] = 1
                refusal_assigned += 1
        if refusal_assigned >= refusal_target:
            break

    if refusal_assigned != refusal_target:
        raise ValueError(
            f"Could not assign the required number of refusals. "
            f"Needed {refusal_target}, assigned {refusal_assigned}."
        )

    for item in plan:
        item["answerable_single_count"] = item["single_total_count"] - item["refusal_single_count"]

    return plan


def distribute_multi_turns(
    plan: List[Dict[str, Any]],
    multi_target: int,
    seed: int
) -> List[Dict[str, Any]]:
    n = len(plan)
    if multi_target > n:
        raise ValueError(
            f"multi_target ({multi_target}) exceeds number of documents ({n}). "
            f"This script supports at most one multi-turn dialogue per document."
        )

    rng = random.Random(seed + 1)
    order = list(range(n))
    rng.shuffle(order)

    for idx in order[:multi_target]:
        plan[idx]["include_multi_turn"] = True

    return plan


def validate_turn_order(dialogue: List[Dict[str, str]]) -> bool:
    if len(dialogue) != 4:
        return False
    expected = ["user", "assistant", "user", "assistant"]
    return [turn.get("role") for turn in dialogue] == expected


def validate_generated_output(
    data: Dict[str, Any],
    answerable_single_count: int,
    refusal_single_count: int,
    include_multi_turn: bool
) -> None:
    if len(data["answerable_direct_qa"]) != answerable_single_count:
        raise ValueError("Model returned wrong answerable_direct_qa count.")

    if len(data["refusal_direct_qa"]) != refusal_single_count:
        raise ValueError("Model returned wrong refusal_direct_qa count.")

    expected_multi = 1 if include_multi_turn else 0
    if len(data["multi_turn_dialogues"]) != expected_multi:
        raise ValueError("Model returned wrong multi_turn_dialogues count.")

    for qa in data["refusal_direct_qa"]:
        if qa["answer"] != REFUSAL_TEXT:
            raise ValueError("Refusal answer does not match required text exactly.")

    for dialogue in data["multi_turn_dialogues"]:
        if not validate_turn_order(dialogue):
            raise ValueError("Dialogue turn order is invalid.")


def call_model_for_record(
    document_text: str,
    answerable_single_count: int,
    refusal_single_count: int,
    include_multi_turn: bool
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(
        document_text=document_text,
        answerable_single_count=answerable_single_count,
        refusal_single_count=refusal_single_count,
        include_multi_turn=include_multi_turn
    )

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "doc_grounded_training_data",
                "schema": SCHEMA,
                "strict": True
            }
        }
    )

    data = json.loads(response.output_text)

    validate_generated_output(
        data=data,
        answerable_single_count=answerable_single_count,
        refusal_single_count=refusal_single_count,
        include_multi_turn=include_multi_turn
    )

    return data


def convert_to_sharegpt_examples(
    generated: Dict[str, Any],
    source_record_id: str,
    source_row_index: int
) -> List[Dict[str, Any]]:
    items = []

    for qa in generated.get("answerable_direct_qa", []):
        items.append({
            "example_type": "single_qa_answerable",
            "source_record_id": source_record_id,
            "source_row_index": source_row_index,
            "conversations": [
                {"from": "human", "value": qa["question"]},
                {"from": "gpt", "value": qa["answer"]}
            ]
        })

    for qa in generated.get("refusal_direct_qa", []):
        items.append({
            "example_type": "single_qa_refusal",
            "source_record_id": source_record_id,
            "source_row_index": source_row_index,
            "conversations": [
                {"from": "human", "value": qa["question"]},
                {"from": "gpt", "value": qa["answer"]}
            ]
        })

    for dialogue in generated.get("multi_turn_dialogues", []):
        formatted_dialogue = []
        for turn in dialogue:
            role = "human" if turn["role"] == "user" else "gpt"
            formatted_dialogue.append({
                "from": role,
                "value": turn["content"]
            })

        items.append({
            "example_type": "multi_turn",
            "source_record_id": source_record_id,
            "source_row_index": source_row_index,
            "conversations": formatted_dialogue
        })

    return items


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_single_record(item: Dict[str, Any], max_retries: int) -> Dict[str, Any]:
    record_id = item["record_id"]
    row_index = item["row_index"]
    text = item["text"]

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            generated = call_model_for_record(
                document_text=text,
                answerable_single_count=item["answerable_single_count"],
                refusal_single_count=item["refusal_single_count"],
                include_multi_turn=item["include_multi_turn"]
            )

            examples = convert_to_sharegpt_examples(
                generated=generated,
                source_record_id=record_id,
                source_row_index=row_index
            )

            return {
                "success": True,
                "examples": examples,
                "audit": {
                    "source_record_id": record_id,
                    "source_row_index": row_index,
                    "answerable_single_count": item["answerable_single_count"],
                    "refusal_single_count": item["refusal_single_count"],
                    "include_multi_turn": item["include_multi_turn"],
                    "generated_example_count": len(examples),
                    "status": "success"
                },
                "failed": None
            }

        except Exception as e:
            last_error = f"Attempt {attempt}/{max_retries}: {str(e)}"

    return {
        "success": False,
        "examples": [],
        "audit": None,
        "failed": {
            "source_record_id": record_id,
            "source_row_index": row_index,
            "answerable_single_count": item["answerable_single_count"],
            "refusal_single_count": item["refusal_single_count"],
            "include_multi_turn": item["include_multi_turn"],
            "status": "failed",
            "error": last_error
        }
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate balanced instruction data from a local CSV.")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH, help="Path to local CSV file.")
    parser.add_argument("--text_column", type=str, required=True, help="Name of the text column in the CSV.")
    parser.add_argument("--id_column", type=str, default=None, help="Optional ID column name.")
    parser.add_argument("--output_dir", type=str, default="./generated_dataset", help="Directory for outputs.")
    parser.add_argument("--total_examples", type=int, default=DEFAULT_TOTAL_EXAMPLES, help="Requested total final examples.")
    parser.add_argument("--single_ratio", type=float, default=DEFAULT_SINGLE_RATIO, help="Answerable single QA ratio.")
    parser.add_argument("--multi_ratio", type=float, default=DEFAULT_MULTI_RATIO, help="Multi-turn ratio.")
    parser.add_argument("--refusal_ratio", type=float, default=DEFAULT_REFUSAL_RATIO, help="Refusal ratio.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for dry runs.")
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES, help="Retries per record.")
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel workers for API requests.")
    parser.add_argument(
        "--disable_auto_scale",
        action="store_true",
        help="Disable auto-scaling of total_examples for small document limits."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        if csv_path != DEFAULT_CSV_PATH and os.path.exists(DEFAULT_CSV_PATH):
            print(f"Warning: CSV not found at '{csv_path}'. Falling back to default path '{DEFAULT_CSV_PATH}'.")
            csv_path = DEFAULT_CSV_PATH
        else:
            raise FileNotFoundError(
                f"CSV file not found: '{csv_path}'. "
                f"Provide a valid --csv_path or place the file at '{DEFAULT_CSV_PATH}'."
            )

    records = load_csv_records(
        csv_path=csv_path,
        text_column=args.text_column,
        id_column=args.id_column,
        limit=args.limit
    )

    n_docs = len(records)
    if n_docs == 0:
        raise ValueError("No usable records found in the CSV.")

    requested_total_examples = args.total_examples
    if args.disable_auto_scale:
        adjusted_total_examples = requested_total_examples
    else:
        adjusted_total_examples = auto_scale_total_examples(
            requested_total_examples=requested_total_examples,
            n_docs=n_docs,
            multi_ratio=args.multi_ratio
        )

    targets = compute_targets(
        total_examples=adjusted_total_examples,
        single_ratio=args.single_ratio,
        multi_ratio=args.multi_ratio,
        refusal_ratio=args.refusal_ratio
    )

    print(f"Loaded {n_docs} non-empty records from CSV.")
    print("\nRequested total_examples:", requested_total_examples)
    print("Adjusted total_examples:", adjusted_total_examples)
    print("\nTarget distribution:")
    print(json.dumps(targets, indent=2))

    plan = distribute_single_turns(
        records=records,
        single_total_target=targets["single_total_target"],
        refusal_target=targets["refusal_target"],
        seed=args.seed
    )

    plan = distribute_multi_turns(
        plan=plan,
        multi_target=targets["multi_target"],
        seed=args.seed
    )

    total_answerable = sum(x["answerable_single_count"] for x in plan)
    total_refusal = sum(x["refusal_single_count"] for x in plan)
    total_multi = sum(1 for x in plan if x["include_multi_turn"])

    assert total_answerable == targets["answerable_target"], (total_answerable, targets["answerable_target"])
    assert total_refusal == targets["refusal_target"], (total_refusal, targets["refusal_target"])
    assert total_multi == targets["multi_target"], (total_multi, targets["multi_target"])

    print("\nPlanned distribution verified:")
    print(f"Answerable single QA: {total_answerable}")
    print(f"Refusal QA: {total_refusal}")
    print(f"Multi-turn dialogues: {total_multi}")
    print(f"Total examples: {total_answerable + total_refusal + total_multi}")

    all_examples = []
    audit_rows = []
    failed_rows = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_single_record, item, args.max_retries) for item in plan]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating dataset"):
            result = future.result()

            all_examples.extend(result["examples"])

            if result["audit"] is not None:
                audit_rows.append(result["audit"])

            if result["failed"] is not None:
                failed_rows.append(result["failed"])

    sharegpt_path = os.path.join(args.output_dir, "sharegpt_dataset.json")
    audit_jsonl_path = os.path.join(args.output_dir, "audit.jsonl")
    failed_json_path = os.path.join(args.output_dir, "failed_rows.json")
    plan_json_path = os.path.join(args.output_dir, "generation_plan.json")
    summary_json_path = os.path.join(args.output_dir, "summary.json")

    write_json(sharegpt_path, all_examples)
    write_jsonl(audit_jsonl_path, audit_rows)
    write_json(failed_json_path, failed_rows)
    write_json(plan_json_path, plan)

    summary = {
        "input_records": n_docs,
        "requested_total_examples": requested_total_examples,
        "adjusted_total_examples": adjusted_total_examples,
        "generated_examples": len(all_examples),
        "successful_records": len(audit_rows),
        "failed_records": len(failed_rows),
        "example_type_counts": {
            "single_qa_answerable": sum(1 for x in all_examples if x["example_type"] == "single_qa_answerable"),
            "single_qa_refusal": sum(1 for x in all_examples if x["example_type"] == "single_qa_refusal"),
            "multi_turn": sum(1 for x in all_examples if x["example_type"] == "multi_turn")
        },
        "requested_ratios": {
            "single_ratio": args.single_ratio,
            "multi_ratio": args.multi_ratio,
            "refusal_ratio": args.refusal_ratio
        },
        "targets": targets,
        "output_files": {
            "sharegpt_dataset": sharegpt_path,
            "audit_jsonl": audit_jsonl_path,
            "failed_rows": failed_json_path,
            "generation_plan": plan_json_path,
            "summary": summary_json_path
        }
    }

    write_json(summary_json_path, summary)

    print("\nDone.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()