import json
from pathlib import Path
from collections import Counter

EXTRACTED = Path("data/6_curation/validation_gpt5/eval_validator_extracted.jsonl")

cats = Counter()
total_invalid = 0

with EXTRACTED.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("verdict") != "INVALID":
            continue
        total_invalid += 1
        raw = r.get("raw") or {}

        has_h = len(raw.get("hallucinations") or []) > 0
        has_u = len(raw.get("unsupported_claims") or []) > 0
        has_q = len(raw.get("quote_errors") or []) > 0
        fol = raw.get("followup_consistency")

        key = []
        if has_h: key.append("H")
        if has_u: key.append("U")
        if has_q: key.append("Q")
        if fol == "inconsistent": key.append("F")

        cats["+".join(key) if key else "none"] += 1

print("Total INVALID:", total_invalid)
for k, v in cats.most_common():
    print(f"{k:8} {v:3}  ({(100*v/total_invalid):.1f}%)")