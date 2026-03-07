#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

FILE_PATH = Path("data/4_instruction/train_multiturn.jsonl")

REMOVE_TEXT = "Do not mention tools or call external systems."

updated_lines = []
modified_records = 0
removed_count = 0

with FILE_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        record = json.loads(line)
        changed = False

        for msg in record.get("messages", []):
            if msg.get("role") == "system" and REMOVE_TEXT in msg.get("content", ""):
                msg["content"] = msg["content"].replace(REMOVE_TEXT, "").strip()
                removed_count += 1
                changed = True

        if changed:
            modified_records += 1

        updated_lines.append(json.dumps(record, ensure_ascii=False))

# overwrite original file
with FILE_PATH.open("w", encoding="utf-8") as f:
    for line in updated_lines:
        f.write(line + "\n")

print("Done.")
print("Records modified:", modified_records)
print("Occurrences removed:", removed_count)