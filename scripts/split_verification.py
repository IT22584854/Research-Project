import json
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

# =========================================================
# PATHS
# =========================================================
FULL_PATH = Path(r"D:\SL_Medical_Corpus\data\10_instruction_eval\instruction_dataset_evaluated_clean.json")
TRAIN_PATH = Path(r"D:\SL_Medical_Corpus\data\10_instruction_eval\train.jsonl")
TEST_PATH = Path(r"D:\SL_Medical_Corpus\data\10_instruction_eval\eval.jsonl")
CORPUS_CSV = Path(r"C:\Users\Charunya\Downloads\sl_med_corpus_rows (2).csv")

# =========================================================
# CONFIG
# =========================================================
GROUP_FIELD = "source_record_id"
TYPE_FIELD = "example_type"

CSV_ID_CANDIDATES = ["doc_id", "source_record_id", "id"]
CSV_LANG_CANDIDATES = ["language", "language_label", "language_primary", "source_lang"]

# increase CSV field size limit for large markdown/text columns
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
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list.")
    return data


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}, line {line_no}: {e}")
    return rows


def first_existing_field(fieldnames, candidates):
    for c in candidates:
        if c in fieldnames:
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
            raise ValueError("CSV has no headers.")

        id_col = first_existing_field(reader.fieldnames, CSV_ID_CANDIDATES)
        lang_col = first_existing_field(reader.fieldnames, CSV_LANG_CANDIDATES)

        if not id_col:
            raise ValueError(
                f"Could not find ID column. Tried {CSV_ID_CANDIDATES}. Found: {reader.fieldnames}"
            )
        if not lang_col:
            raise ValueError(
                f"Could not find language column. Tried {CSV_LANG_CANDIDATES}. Found: {reader.fieldnames}"
            )

        print(f"Using CSV id column      : {id_col}")
        print(f"Using CSV language column: {lang_col}")

        for row in reader:
            doc_id = str(row.get(id_col, "")).strip()
            lang = normalize_lang(row.get(lang_col))
            if doc_id:
                doc_lang_map[doc_id] = lang

    return doc_lang_map


# =========================================================
# RECORD HELPERS
# =========================================================
def get_group_id(record):
    value = record.get(GROUP_FIELD)
    if not value:
        raise ValueError(f"Record missing '{GROUP_FIELD}'. Keys: {list(record.keys())}")
    return str(value).strip()


def bucket_type(record):
    t = str(record.get(TYPE_FIELD, "")).strip().lower()

    if "refusal" in t or "unanswerable" in t:
        return "refusal"
    if "multi" in t:
        return "multi_turn"
    return "single_turn"


def canonical_record_string(record):
    # stable serialization for exact record comparison
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def annotate_with_language(records, doc_lang_map):
    annotated = []
    missing_lang = 0

    for r in records:
        group_id = get_group_id(r)
        lang = doc_lang_map.get(group_id, "unknown")
        if lang == "unknown":
            missing_lang += 1

        rr = dict(r)
        rr["_lang"] = lang
        rr["_bucket"] = bucket_type(r)
        rr["_group_id"] = group_id
        annotated.append(rr)

    return annotated, missing_lang


# =========================================================
# STATS
# =========================================================
def print_stats(name, records):
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


# =========================================================
# VERIFICATION
# =========================================================
def verify_splits(full_records, train_records, test_records):
    print("\n================ VERIFICATION ================")

    # exact record membership checks
    full_set = set(canonical_record_string(r) for r in full_records)
    train_set = set(canonical_record_string(r) for r in train_records)
    test_set = set(canonical_record_string(r) for r in test_records)

    union_set = train_set | test_set
    intersection_set = train_set & test_set

    missing_records = full_set - union_set
    extra_records = union_set - full_set

    print(f"Full count : {len(full_records)}")
    print(f"Train count: {len(train_records)}")
    print(f"Test count : {len(test_records)}")
    print(f"Sum        : {len(train_records) + len(test_records)}")

    print("\nExact-record checks:")
    print(f"  Missing records from splits: {len(missing_records)}")
    print(f"  Extra records in splits    : {len(extra_records)}")
    print(f"  Duplicate exact rows across train/test: {len(intersection_set)}")

    # document leakage
    train_docs = set(r["_group_id"] for r in train_records)
    test_docs = set(r["_group_id"] for r in test_records)
    doc_overlap = train_docs & test_docs

    print("\nDocument leakage check:")
    print(f"  Overlapping source_record_id values: {len(doc_overlap)}")

    # allocation consistency: every doc_id should belong to only one split
    split_by_doc = defaultdict(set)
    for r in train_records:
        split_by_doc[r["_group_id"]].add("train")
    for r in test_records:
        split_by_doc[r["_group_id"]].add("test")

    mixed_docs = [doc_id for doc_id, splits in split_by_doc.items() if len(splits) > 1]
    print(f"  source_record_id assigned to both splits: {len(mixed_docs)}")

    # pass/fail summary
    checks = {
        "all_records_accounted_for": len(missing_records) == 0 and len(extra_records) == 0,
        "no_exact_row_overlap": len(intersection_set) == 0,
        "no_doc_leakage": len(doc_overlap) == 0,
        "sum_matches_full": len(full_records) == len(train_records) + len(test_records),
    }

    print("\nSummary:")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    if missing_records:
        print("\nSample missing record:")
        sample = next(iter(missing_records))
        print(sample[:1000])

    if extra_records:
        print("\nSample extra record:")
        sample = next(iter(extra_records))
        print(sample[:1000])

    if doc_overlap:
        print("\nSample overlapping source_record_id values:")
        for doc_id in list(sorted(doc_overlap))[:10]:
            print(f"  {doc_id}")

    return checks


# =========================================================
# MAIN
# =========================================================
def main():
    full_raw = load_json(FULL_PATH)
    train_raw = load_jsonl(TRAIN_PATH)
    test_raw = load_jsonl(TEST_PATH)

    doc_lang_map = load_doc_language_map(CORPUS_CSV)

    full_records, full_missing_lang = annotate_with_language(full_raw, doc_lang_map)
    train_records, train_missing_lang = annotate_with_language(train_raw, doc_lang_map)
    test_records, test_missing_lang = annotate_with_language(test_raw, doc_lang_map)

    print(f"\nLanguage mapping coverage:")
    print(f"  Full missing language: {full_missing_lang}")
    print(f"  Train missing language: {train_missing_lang}")
    print(f"  Test missing language: {test_missing_lang}")

    checks = verify_splits(full_records, train_records, test_records)

    print_stats("FULL", full_records)
    print_stats("TRAIN", train_records)
    print_stats("TEST", test_records)

    print("\n================ FINAL STATUS ================")
    if all(checks.values()):
        print("Verification passed.")
    else:
        print("Verification failed. Check the diagnostics above.")


if __name__ == "__main__":
    main()