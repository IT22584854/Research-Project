#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

IN_PATH = Path("data/6_curation/validation_gpt5/curation_pool_1000_invalid.jsonl")
OUT_PATH = Path("data/6_curation/validation_gpt5/repair_batch_input.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-5"
IDK = "I don't know based on the provided text."

REPAIR_SYSTEM = f"""
You repair a 2-turn assistant conversation using ONLY the Context embedded in the user's message.

Hard rules:
1) Use ONLY the provided Context. Do not add any facts not present in Context.
2) If the Context does not contain enough information to answer, reply exactly: "{IDK}"
3) For follow-up requests asking to quote exact line(s), quote ONLY text that appears verbatim in the Context.
4) Keep the language requested by the user (English/Sinhala/Tamil). Preserve the user's style cues (e.g., Machan/Anna etc.) naturally.
5) Do not mention policies, external sources, browsing, databases, or tools.

Return ONLY a single JSON object with keys:
- "a1": corrected assistant reply to turn 1
- "a2": corrected assistant reply to turn 2
- "repair_notes": short reason for the fix (e.g., "fixed unsupported claim", "fixed quote", "set to IDK due to missing info")
"""

def extract_turns(messages):
    # expected:
    # 0 system
    # 1 user (contains Context)
    # 2 assistant (a1)
    # 3 user (follow-up)
    # 4 assistant (a2)
    return (
        messages[0]["content"],
        messages[1]["content"],
        messages[2]["content"],
        messages[3]["content"],
        messages[4]["content"],
    )

with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        line = line.strip()
        if not line:
            continue

        rec = json.loads(line)
        rid = rec["id"]
        msgs = rec.get("messages", [])
        if len(msgs) < 5:
            continue

        sys0, user1, a1_old, user2, a2_old = extract_turns(msgs)
        v = rec.get("_validation_gpt5") or {}

        # Give the model the original convo + validator findings.
        prompt = {
            "record_id": rid,
            "task": rec.get("task"),
            "lang": rec.get("lang"),
            "style": rec.get("style"),
            "turns": {
                "system": sys0,
                "user1": user1,
                "assistant1_old": a1_old,
                "user2": user2,
                "assistant2_old": a2_old,
            },
            "validator_findings": v,
            "required_idk": IDK,
        }

        req = {
            "custom_id": f"repair-{rid}-{i}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": MODEL,
                # Batch model supports minimal/low/medium/high — use minimal to avoid token-eating.
                "reasoning": {"effort": "minimal"},
                "max_output_tokens": 900,
                "text": {"format": {"type": "json_object"}},
                "truncation": "auto",
                "input": [
                    {"role": "system", "content": REPAIR_SYSTEM.strip()},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
            },
        }

        fout.write(json.dumps(req, ensure_ascii=False) + "\n")

print("Repair batch input written to:", OUT_PATH)