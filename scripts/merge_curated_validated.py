#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Optional

IN_DATA = Path("data/6_curation/curation_pool_1000.jsonl")
BATCH_OUT = Path("data/6_curation/validation_gpt5/batch_output.jsonl")

OUT_ALL = Path("data/6_curation/validation_gpt5/curation_pool_1000_validated.jsonl")
OUT_BAD = Path("data/6_curation/validation_gpt5/curation_pool_1000_invalid.jsonl")
OUT_SUM = Path("data/6_curation/validation_gpt5/validation_summary.json")

# -------- helpers --------

def parse_custom_id(custom_id: str) -> str:
    """
    Supports:
      val-<id>
      val-<id>-<index>
    Returns record id.
    """
    if not custom_id:
        return ""
    s = custom_id.strip()
    if s.startswith("val-"):
        s = s[4:]
    # If it ends with -digits, strip that suffix
    s = re.sub(r"-\d+$", "", s)
    return s

def extract_output_text(batch_line: Dict[str, Any]) -> str:
    """
    Batch output line shape:
      {custom_id, response: {status_code, body: <Responses API response json>}, error: null}
    We try to extract the model's text output.
    """
    resp = (batch_line.get("response") or {})
    body = (resp.get("body") or {})

    # Some SDKs include output_text directly:
    if isinstance(body.get("output_text"), str):
        return body["output_text"].strip()

    # Typical Responses API shape: body["output"] is list of items
    out = body.get("output")
    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict):
                continue
            # "message" items contain content parts
            if item.get("type") == "message":
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                            txt = part.get("text")
                            if isinstance(txt, str) and txt.strip():
                                return txt.strip()

    # Fallback: stringify body (rare)
    return ""

def safe_json_loads(txt: str) -> Optional[dict]:
    txt = (txt or "").strip()
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        # Try to extract a {...} blob if model added extra tokens
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

# -------- main --------

def main():
    if not IN_DATA.exists():
        raise SystemExit(f"Missing input dataset: {IN_DATA}")
    if not BATCH_OUT.exists():
        raise SystemExit(f"Missing batch output: {BATCH_OUT}")

    # 1) Load validations keyed by record_id
    validations: Dict[str, dict] = {}
    malformed = 0

    with BATCH_OUT.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            custom_id = obj.get("custom_id", "")
            record_id = parse_custom_id(custom_id)
            if not record_id:
                continue

            # Only successful lines are in batch_output.jsonl
            txt = extract_output_text(obj)
            verdict_obj = safe_json_loads(txt)

            if not isinstance(verdict_obj, dict):
                malformed += 1
                verdict_obj = {
                    "verdict": "INVALID",
                    "hallucinations": [],
                    "unsupported_claims": ["Validator returned non-JSON output"],
                    "quote_errors": [],
                    "followup_consistency": "inconsistent",
                    "confidence": 0.0,
                    "_raw_validator_text": (txt or "")[:1500],
                }

            validations[record_id] = verdict_obj

    # 2) Merge into dataset
    total = 0
    merged = 0
    missing = 0
    verdict_counts = Counter()
    follow_counts = Counter()

    invalid_reasons = Counter()

    with IN_DATA.open("r", encoding="utf-8") as fin, \
         OUT_ALL.open("w", encoding="utf-8") as fall, \
         OUT_BAD.open("w", encoding="utf-8") as fbad:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            rid = rec.get("id", "")

            v = validations.get(rid)
            if v is None:
                missing += 1
                rec["_validation_gpt5"] = {
                    "verdict": "INVALID",
                    "hallucinations": [],
                    "unsupported_claims": ["Missing validator output for this record_id"],
                    "quote_errors": [],
                    "followup_consistency": "inconsistent",
                    "confidence": 0.0,
                }
            else:
                merged += 1
                rec["_validation_gpt5"] = v

            verdict = (rec["_validation_gpt5"].get("verdict") or "INVALID").upper()
            verdict_counts[verdict] += 1
            follow_counts[str(rec["_validation_gpt5"].get("followup_consistency", "unknown"))] += 1

            # Collect some “reason stats” if present
            for k in ("hallucinations", "unsupported_claims", "quote_errors", "contradictions", "incomplete_answers"):
                arr = rec["_validation_gpt5"].get(k)
                if isinstance(arr, list) and arr:
                    invalid_reasons[k] += 1

            fall.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if verdict != "VALID":
                fbad.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "input_records": total,
        "validator_records_loaded": len(validations),
        "merged_records": merged,
        "missing_records": missing,
        "malformed_validator_outputs": malformed,
        "verdict_counts": dict(verdict_counts),
        "followup_consistency_counts": dict(follow_counts),
        "error_type_presence_counts": dict(invalid_reasons),
        "paths": {
            "validated_all": str(OUT_ALL),
            "invalid_only": str(OUT_BAD),
            "summary": str(OUT_SUM),
        },
    }

    OUT_SUM.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nWrote:")
    print(" -", OUT_ALL)
    print(" -", OUT_BAD)
    print(" -", OUT_SUM)

if __name__ == "__main__":
    main()