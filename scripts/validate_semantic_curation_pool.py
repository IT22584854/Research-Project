#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

IN_PATH = Path("data/4_instruction/eval_multiturn.jsonl")
OUT_PATH = Path("data/6_curation/validation_gpt5/eval_batch_input.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-5"  # will resolve to something like gpt-5-2025-08-07

VALIDATOR_SYSTEM = """
You are a strict document-grounded QA validator.

The conversation contains a Context section inside user messages.
All assistant answers must be evaluated ONLY against that context.

Rules:
1. Mark INVALID if any assistant claim is not supported by the context.
2. Mark INVALID if any hallucination or invented detail appears.
3. Mark INVALID if quoted text does not exactly appear in context.
4. Check follow-up consistency.
5. If assistant says "I don't know", verify that the context truly lacks the answer.

Paraphrasing is allowed if meaning matches context.

Return ONLY a single JSON object with exactly these keys:
verdict, hallucinations, unsupported_claims, quote_errors, followup_consistency, confidence

Schema:
{
  "verdict": "VALID" | "INVALID",
  "hallucinations": ["..."],
  "unsupported_claims": ["..."],
  "quote_errors": ["..."],
  "followup_consistency": "consistent" | "inconsistent",
  "confidence": 0.0
}
"""

with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        line = line.strip()
        if not line:
            continue

        record = json.loads(line)
        rid = record["id"]

        req = {
            "custom_id": f"val-{rid}-{i}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": MODEL,
                "reasoning": {"effort": "minimal"},        # ✅ critical
                "max_output_tokens": 800,               # optional but helpful
                "text": {"format": {"type": "json_object"}},
                "input": [
                    {"role": "system", "content": VALIDATOR_SYSTEM},
                    {"role": "user", "content": json.dumps(record, ensure_ascii=False)},
                ],
            },
        }

        fout.write(json.dumps(req, ensure_ascii=False) + "\n")

print("Batch input written to:", OUT_PATH)