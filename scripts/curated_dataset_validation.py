#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import hashlib
from pathlib import Path

IN_PATH = Path("data/6_curation/curation_pool_1000.jsonl")
OUT_PATH = Path("data/6_curation/validation_gpt5/batch_input.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-5"  # GPT-5 flagship (use your exact model id if your org requires a dated one)

SYSTEM = (
    "You are a strict dataset validator for an instruction-tuning JSONL dataset. "
    "Return ONLY JSON. "
    "Decide if the record is VALID or INVALID for training.\n\n"
    "Return:\n"
    '{ "verdict":"VALID|INVALID", "reasons":["..."], "suggested_fix":null|"..." }'
)

def record_id(obj: dict, fallback_line: str) -> str:
    # Prefer an existing id field if you have one; otherwise hash the record
    if isinstance(obj.get("id"), str) and obj["id"].strip():
        return obj["id"].strip()
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
    for i, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Still send it—model can mark invalid; include raw line
            obj = {"_raw_line": line}

        cid = f"val-{record_id(obj, line)}"

        # One batch request line (Responses API)
        req = {
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": MODEL,
                "input": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(obj, ensure_ascii=False)},
                ],
                "temperature": 0,
                # Optional: keep outputs short-ish to control cost
                "max_output_tokens": 300,
            },
        }

        fout.write(json.dumps(req, ensure_ascii=False) + "\n")

print(f"Wrote: {OUT_PATH}")