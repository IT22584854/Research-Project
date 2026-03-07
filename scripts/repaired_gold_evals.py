#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import Counter

OLD_GOLD_EVAL = Path(r"D:\SL_Medical_Corpus\data\7_gold_standard\gold_eval_DOC_DISJOINT.jsonl")
NEW_VALID = Path(r"D:\SL_Medical_Corpus\data\7_gold_standard\repair_work\gold_eval_repair_VALID_v2.jsonl")
OUT_PATH = Path(r"D:\SL_Medical_Corpus\data\7_gold_standard\gold_eval_REPAIRED_99.jsonl")

LEAKED_DOC_IDS = {
    "27ef09f20c972f54c6b8620f27e8776ef07e59c9c7dd53e6376d60c39956e72f",
    "362f63d30590e31e87e38e5bfc139194bf3474b6bab44f32c9bd1ed86fe08e7b",
    "4dc9c6fe3c6e230f21ca314d40fd5a10451850de4f395421ca5b45d7bc919fe3",
    "94dabe1cedc2bfc101e768d0329241c0e7260d66906b00481bf049b12d957ca3",
    "cd5aac1c45cd24a648c645e9c78c42b1e2fdb367bebb0affb4342bee3087fd94",
}

TARGET_COUNTS = {
    "en": 33,
    "si": 33,
    "ta": 33,
}

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    # Keep only non-leaked old gold-eval records
    old_clean = []
    for rec in read_jsonl(OLD_GOLD_EVAL):
        if rec["doc_id"] not in LEAKED_DOC_IDS:
            old_clean.append(rec)

    old_lang_counts = Counter(rec["lang"] for rec in old_clean)
    print("Old clean records kept:", len(old_clean))
    print("Old clean language counts:", dict(old_lang_counts))

    # Figure out how many new ones we need per language
    need = {
        lang: TARGET_COUNTS[lang] - old_lang_counts.get(lang, 0)
        for lang in TARGET_COUNTS
    }
    print("Need from repair set:", need)

    # Select new valid records by language
    selected_new = []
    selected_ids = set()

    for lang in ["en", "si", "ta"]:
        candidates = [r for r in read_jsonl(NEW_VALID) if r.get("lang") == lang]

        taken = 0
        for rec in candidates:
            rid = rec.get("id")
            if rid in selected_ids:
                continue
            selected_new.append(rec)
            selected_ids.add(rid)
            taken += 1
            if taken >= need[lang]:
                break

        if taken < need[lang]:
            raise SystemExit(f"Not enough valid {lang} records. Needed {need[lang]}, got {taken}")

    final_records = old_clean + selected_new

    final_lang_counts = Counter(rec["lang"] for rec in final_records)
    final_task_counts = Counter(rec["task"] for rec in final_records)
    final_doc_ids = {rec["doc_id"] for rec in final_records}

    write_jsonl(OUT_PATH, final_records)

    print("\nWrote ->", OUT_PATH)
    print("Final records:", len(final_records))
    print("Final language counts:", dict(final_lang_counts))
    print("Final task counts:", dict(final_task_counts))
    print("Final unique documents:", len(final_doc_ids))

if __name__ == "__main__":
    main()
