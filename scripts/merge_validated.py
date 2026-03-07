#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, re
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Optional

IN_DATA = Path("data/6_curation/validation_gpt5/curation_pool_1000_repaired.jsonl")
BATCH_OUT = Path("data/6_curation/validation_gpt5/repaired_batch_output.jsonl")

OUT_ALL = Path("data/6_curation/validation_gpt5/curation_pool_1000_repaired_validated.jsonl")
OUT_BAD = Path("data/6_curation/validation_gpt5/curation_pool_1000_repaired_invalid.jsonl")
OUT_SUM = Path("data/6_curation/validation_gpt5/repaired_validation_summary.json")

def parse_custom_id(custom_id: str) -> str:
    s = (custom_id or "").strip()
    # supports: val-cur_xxx-0, reval-cur_xxx-0, etc.
    s = re.sub(r"^(reval-|val-)", "", s)   # <-- key fix
    s = re.sub(r"-\d+$", "", s)            # strip trailing index
    return s

def extract_output_text(batch_line: Dict[str, Any]) -> str:
    body = ((batch_line.get("response") or {}).get("body") or {})
    if isinstance(body.get("output_text"), str) and body["output_text"].strip():
        return body["output_text"].strip()

    out = body.get("output")
    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                            t = part.get("text")
                            if isinstance(t, str) and t.strip():
                                return t.strip()
    return ""

def safe_json_loads(txt: str) -> Optional[dict]:
    txt = (txt or "").strip()
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

def main():
    validations: Dict[str, dict] = {}
    malformed = 0

    with BATCH_OUT.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rid = parse_custom_id(obj.get("custom_id"))
            txt = extract_output_text(obj)
            jo = safe_json_loads(txt)
            if not isinstance(jo, dict):
                malformed += 1
                continue
            validations[rid] = jo

    total = 0
    merged = 0
    missing = 0
    verdict_counts = Counter()

    with IN_DATA.open("r", encoding="utf-8") as fin, \
         OUT_ALL.open("w", encoding="utf-8") as fall, \
         OUT_BAD.open("w", encoding="utf-8") as fbad:

        for line in fin:
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)
            rid = rec.get("id", "")

            v = validations.get(rid)
            if v is None:
                missing += 1
                rec["_reval_gpt5"] = {
                    "verdict": "INVALID",
                    "unsupported_claims": ["Missing re-validation output for this record_id"],
                    "hallucinations": [],
                    "quote_errors": [],
                    "followup_consistency": "inconsistent",
                    "confidence": 0.0,
                }
            else:
                merged += 1
                rec["_reval_gpt5"] = v

            verdict = (rec["_reval_gpt5"].get("verdict") or "INVALID").upper()
            verdict_counts[verdict] += 1

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