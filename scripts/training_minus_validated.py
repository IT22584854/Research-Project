#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

# 5000-record training set (change filename if yours is different)
TRAIN_5000 = Path("data/4_instruction/train_multiturn.jsonl")

# The 1000 that were selected earlier (source of IDs to remove)
CURATED_1000 = Path("data/6_curation/curation_pool_1000.jsonl")

# Output: remaining records (what you asked for)
OUT_REMAINING = Path("data/4_instruction/train_multiturn_remaining_4000.jsonl")

# Optional output: the ones that got removed
OUT_REMOVED = Path("data/4_instruction/train_multiturn_removed_curated1000.jsonl")

def iter_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# Load IDs to remove
remove_ids = set()
for obj in iter_jsonl(CURATED_1000):
    rid = obj.get("id")
    if rid:
        remove_ids.add(rid)

total = 0
kept = 0
removed = 0

with TRAIN_5000.open("r", encoding="utf-8") as fin, \
     OUT_REMAINING.open("w", encoding="utf-8") as fkeep, \
     OUT_REMOVED.open("w", encoding="utf-8") as frem:

    for line in fin:
        line = line.strip()
        if not line:
            continue
        total += 1
        obj = json.loads(line)
        rid = obj.get("id")

        if rid in remove_ids:
            frem.write(json.dumps(obj, ensure_ascii=False) + "\n")
            removed += 1
        else:
            fkeep.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

print("Total in train:", total)
print("Removed (curated1000):", removed)
print("Remaining written:", kept)
print("Remaining file:", OUT_REMAINING)
print("Removed file (optional):", OUT_REMOVED)