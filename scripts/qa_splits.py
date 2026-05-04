import json
import csv
import sys
import random
from collections import Counter, defaultdict
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
INPUT_JSON = Path(r"D:\SL_Medical_Corpus\data\10_instruction_eval\instruction_dataset_evaluated_clean.json")
CORPUS_CSV = Path(r"C:\Users\Charunya\Downloads\sl_med_corpus_rows (2).csv")

TRAIN_OUT = Path(r"D:\SL_Medical_Corpus\data\10_instruction_eval\train.jsonl")
EVAL_OUT  = Path(r"D:\SL_Medical_Corpus\data\10_instruction_eval\eval.jsonl")

RANDOM_SEED = 42
MIN_TRAIN_SIZE = 5000

# final QA dataset fields
GROUP_ID_FIELD = "source_record_id"
TYPE_FIELD = "example_type"

# corpus CSV candidates
CSV_DOC_ID_CANDIDATES = ["doc_id", "source_record_id", "id"]
CSV_LANG_CANDIDATES = ["language", "language_label", "language_primary", "source_lang"]

TARGET_BUCKET_DIST = {
    "single_turn": 0.70,
    "multi_turn": 0.25,
    "refusal": 0.05,
}

# Increase CSV field size limit for large text columns
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = max_int // 10


# =========================================================
# LOADERS
# =========================================================
def load_json_records(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of records.")
    return data


def first_existing_field(d, candidates):
    for c in candidates:
        if c in d:
            return c
    return None


def normalize_lang(value):
    if value is None:
        return "unknown"

    v = str(value).strip().lower()
    mapping = {
        "en": "en",
        "eng": "en",
        "english": "en",
        "si": "si",
        "sin": "si",
        "sinhala": "si",
        "sin_sinh": "si",
        "ta": "ta",
        "tam": "ta",
        "tamil": "ta",
        "tam_taml": "ta",
        "mixed": "mixed",
    }
    return mapping.get(v, v if v else "unknown")


def load_doc_language_map(csv_path: Path):
    doc_lang_map = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        header_lookup = {name: True for name in reader.fieldnames}

        doc_id_col = first_existing_field(header_lookup, CSV_DOC_ID_CANDIDATES)
        lang_col = first_existing_field(header_lookup, CSV_LANG_CANDIDATES)

        if not doc_id_col:
            raise ValueError(
                f"Could not find ID column. Tried {CSV_DOC_ID_CANDIDATES}. Found: {reader.fieldnames}"
            )

        if not lang_col:
            raise ValueError(
                f"Could not find language column. Tried {CSV_LANG_CANDIDATES}. Found: {reader.fieldnames}"
            )

        print(f"Using CSV id column      : {doc_id_col}")
        print(f"Using CSV language column: {lang_col}")

        for row in reader:
            doc_id = str(row.get(doc_id_col, "")).strip()
            lang = normalize_lang(row.get(lang_col))
            if doc_id:
                doc_lang_map[doc_id] = lang

    return doc_lang_map


# =========================================================
# LABELING
# =========================================================
def get_group_id(record):
    group_id = record.get(GROUP_ID_FIELD)
    if not group_id:
        raise ValueError(f"Record missing '{GROUP_ID_FIELD}'")
    return str(group_id).strip()


def assign_bucket(record):
    ex_type = str(record.get(TYPE_FIELD, "")).strip().lower()

    if "refusal" in ex_type or "unanswerable" in ex_type:
        return "refusal"

    if "multi" in ex_type:
        return "multi_turn"

    return "single_turn"


def annotate_records(records, doc_lang_map):
    annotated = []
    missing_lang = 0

    for r in records:
        group_id = get_group_id(r)

        lang = doc_lang_map.get(group_id, "unknown")
        if lang == "unknown":
            missing_lang += 1

        annotated.append({
            **r,
            "_group_id": group_id,
            "_lang": lang,
            "_bucket": assign_bucket(r),
        })

    return annotated, missing_lang


# =========================================================
# GROUPING
# =========================================================
def build_group_stats(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["_group_id"]].append(r)

    out = []
    for group_id, items in groups.items():
        out.append({
            "group_id": group_id,
            "records": items,
            "size": len(items),
            "lang_counts": Counter(x["_lang"] for x in items),
            "bucket_counts": Counter(x["_bucket"] for x in items),
        })
    return out


def counter_to_freq(counter, total):
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def l1_distance(freq_a, freq_b, keys):
    return sum(abs(freq_a.get(k, 0.0) - freq_b.get(k, 0.0)) for k in keys)


# =========================================================
# SCORING
# =========================================================
def score_candidate(current_eval_size, current_lang_counts, current_bucket_counts,
                    group, target_eval_size, global_lang_dist,
                    bucket_keys, lang_keys):
    new_size = current_eval_size + group["size"]

    new_lang_counts = current_lang_counts + group["lang_counts"]
    new_bucket_counts = current_bucket_counts + group["bucket_counts"]

    new_lang_freq = counter_to_freq(new_lang_counts, new_size)
    new_bucket_freq = counter_to_freq(new_bucket_counts, new_size)

    size_penalty = abs(target_eval_size - new_size) / max(target_eval_size, 1)
    overshoot_penalty = max(0, new_size - target_eval_size) / max(target_eval_size, 1)

    lang_penalty = l1_distance(new_lang_freq, global_lang_dist, lang_keys)
    bucket_penalty = l1_distance(new_bucket_freq, TARGET_BUCKET_DIST, bucket_keys)

    return (
        2.5 * size_penalty +
        1.5 * overshoot_penalty +
        1.5 * lang_penalty +
        2.0 * bucket_penalty
    )


# =========================================================
# STRICT SPLIT
# =========================================================
def grouped_split_strict_min_train(records, min_train_size=5000, seed=42):
    random.seed(seed)

    total_records = len(records)
    if min_train_size >= total_records:
        raise ValueError(
            f"MIN_TRAIN_SIZE ({min_train_size}) must be smaller than total records ({total_records})."
        )

    target_eval_size = total_records - min_train_size

    global_lang_counts = Counter(r["_lang"] for r in records)
    global_lang_dist = counter_to_freq(global_lang_counts, total_records)

    bucket_keys = ["single_turn", "multi_turn", "refusal"]
    lang_keys = sorted(global_lang_counts.keys())

    groups = build_group_stats(records)
    random.shuffle(groups)
    groups.sort(key=lambda g: g["size"], reverse=True)

    eval_group_ids = set()
    eval_size = 0
    eval_lang_counts = Counter()
    eval_bucket_counts = Counter()

    remaining_groups = groups[:]

    while remaining_groups:
        best_idx = None
        best_score = float("inf")

        remaining_train_if_take_any = total_records - eval_size
        if remaining_train_if_take_any <= min_train_size:
            break

        for i, group in enumerate(remaining_groups):
            hypothetical_eval_size = eval_size + group["size"]
            hypothetical_train_size = total_records - hypothetical_eval_size

            if hypothetical_train_size < min_train_size:
                continue

            score = score_candidate(
                current_eval_size=eval_size,
                current_lang_counts=eval_lang_counts,
                current_bucket_counts=eval_bucket_counts,
                group=group,
                target_eval_size=target_eval_size,
                global_lang_dist=global_lang_dist,
                bucket_keys=bucket_keys,
                lang_keys=lang_keys,
            )

            if score < best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break

        chosen = remaining_groups.pop(best_idx)
        eval_group_ids.add(chosen["group_id"])
        eval_size += chosen["size"]
        eval_lang_counts += chosen["lang_counts"]
        eval_bucket_counts += chosen["bucket_counts"]

    eval_records = [r for r in records if r["_group_id"] in eval_group_ids]
    train_records = [r for r in records if r["_group_id"] not in eval_group_ids]

    if len(train_records) < min_train_size:
        raise RuntimeError(
            f"Split failed: train size is {len(train_records)}, which is below {min_train_size}."
        )

    return train_records, eval_records


# =========================================================
# OUTPUT
# =========================================================
def strip_helper_fields(records):
    cleaned = []
    for r in records:
        rr = dict(r)
        rr.pop("_group_id", None)
        rr.pop("_lang", None)
        rr.pop("_bucket", None)
        cleaned.append(rr)
    return cleaned


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(name, records):
    total = len(records)
    unique_groups = len(set(r["_group_id"] for r in records))
    bucket_counts = Counter(r["_bucket"] for r in records)
    lang_counts = Counter(r["_lang"] for r in records)

    print(f"\n=== {name} ===")
    print(f"Total records: {total}")
    print(f"Unique source_record_id: {unique_groups}")

    print("\nBucket distribution:")
    for k in ["single_turn", "multi_turn", "refusal"]:
        c = bucket_counts.get(k, 0)
        pct = (c / total * 100) if total else 0
        print(f"  {k}: {c} ({pct:.2f}%)")

    print("\nLanguage distribution:")
    for k, c in sorted(lang_counts.items()):
        pct = (c / total * 100) if total else 0
        print(f"  {k}: {c} ({pct:.2f}%)")


def main():
    records = load_json_records(INPUT_JSON)
    doc_lang_map = load_doc_language_map(CORPUS_CSV)

    print(f"Loaded QA records: {len(records)}")
    print(f"Loaded id->language mappings: {len(doc_lang_map)}")

    annotated, missing_lang = annotate_records(records, doc_lang_map)
    print(f"Records with missing language mapping: {missing_lang}")

    train_records, eval_records = grouped_split_strict_min_train(
        annotated,
        min_train_size=MIN_TRAIN_SIZE,
        seed=RANDOM_SEED,
    )

    summarize("FULL DATASET", annotated)
    summarize("TRAIN", train_records)
    summarize("EVAL", eval_records)

    train_ids = set(r["_group_id"] for r in train_records)
    eval_ids = set(r["_group_id"] for r in eval_records)
    overlap = train_ids & eval_ids

    print("\nLeakage check:")
    print(f"Overlapping source_record_id: {len(overlap)}")

    print("\nFinal sizes:")
    print(f"  Train: {len(train_records)}")
    print(f"  Eval : {len(eval_records)}")
    print(f"  Train >= {MIN_TRAIN_SIZE}: {len(train_records) >= MIN_TRAIN_SIZE}")

    save_jsonl(strip_helper_fields(train_records), TRAIN_OUT)
    save_jsonl(strip_helper_fields(eval_records), EVAL_OUT)

    print("\nSaved:")
    print(f"  Train -> {TRAIN_OUT}")
    print(f"  Eval  -> {EVAL_OUT}")


if __name__ == "__main__":
    main()