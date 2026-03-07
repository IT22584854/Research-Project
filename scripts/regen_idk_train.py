#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

IN_PATH = Path("data/4_instruction/train_multiturn.jsonl")
OUT_PATH = Path("data/6_curation/idk_regen/train_idk_regen_batch_input.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-5"

REGEN_SYSTEM = """
You are regenerating missing/abstained assistant answers in a 2-turn, document-grounded conversation.

You MUST use ONLY the Context contained inside the first user message.
Do NOT invent facts. Do NOT add details not supported by Context.

Goal: Replace "I don't know" style answers with the best grounded answer when possible.

Rules:
- Only say "I don't know" if the answer is truly absent from Context.
- If partially answerable, answer the supported part and state what is missing briefly.
- For tasks that request quotes/evidence: return ONE short exact phrase copied verbatim from Context.
- Match the language requested by the user message (English/Sinhala/Tamil).
- Follow-up consistency: Turn 2 should not contradict Turn 1.

Return ONLY a JSON object:
{
  "replace_turn_1": true/false,
  "turn_1": "<string or null>",
  "replace_turn_2": true/false,
  "turn_2": "<string or null>"
}
"""

def is_idk(text: str) -> bool:
    if not text:
        return False
    t = text.lower().strip()
    return (
        "i don't know" in t
        or "i do not know" in t
        or "don't know based on the provided text" in t
        or "unknown based on the provided text" in t
        or t == "i don't know."
        or t == "i don't know"
    )

with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        line = line.strip()
        if not line:
            continue

        rec = json.loads(line)
        msgs = rec.get("messages", [])

        # Expect: system, user1, asst1, user2, asst2
        if len(msgs) < 5:
            continue

        task = rec.get("task", "")
        a1 = msgs[2].get("content", "")
        a2 = msgs[4].get("content", "")

        # Treat label 'unknown' as valid for classify_doc_type (do not regen just because it's 'unknown')
        a1_is_idk = is_idk(a1) and not (task == "classify_doc_type" and a1.strip() == "unknown")
        a2_is_idk = is_idk(a2)

        if not (a1_is_idk or a2_is_idk):
            continue

        payload = {
            "id": rec.get("id"),
            "task": task,
            "lang": rec.get("lang"),
            "style": rec.get("style"),
            "messages": msgs,
            "a1_is_idk": a1_is_idk,
            "a2_is_idk": a2_is_idk,
        }

        req = {
            "custom_id": f"idkregen-{rec.get('id')}-{i}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": MODEL,
                "reasoning": {"effort": "minimal"},
                "max_output_tokens": 400,
                "text": {"format": {"type": "json_object"}},  # JSON mode :contentReference[oaicite:1]{index=1}
                "input": [
                    {"role": "system", "content": REGEN_SYSTEM},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            },
        }

        fout.write(json.dumps(req, ensure_ascii=False) + "\n")

print("Wrote:", OUT_PATH)