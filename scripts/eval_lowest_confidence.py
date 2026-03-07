import json
from pathlib import Path

EXTRACTED = Path("data/6_curation/validation_gpt5/eval_validator_extracted.jsonl")

rows = []
with EXTRACTED.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("verdict") == "INVALID":
            conf = r.get("confidence")
            rows.append((conf if conf is not None else -1, r))

rows.sort(key=lambda x: x[0])  # lowest confidence first

print("=== 20 lowest-confidence INVALID examples ===")
for conf, r in rows[:20]:
    raw = r.get("raw", {})
    print("\n---")
    print("custom_id:", r.get("custom_id"))
    print("confidence:", conf)
    print("followup_consistency:", r.get("followup_consistency"))
    print("hallucinations_n:", r.get("hallucinations_n"),
          "unsupported_n:", r.get("unsupported_claims_n"),
          "quote_errors_n:", r.get("quote_errors_n"))
    print("unsupported_claims:", (raw.get("unsupported_claims") or [])[:3])
    print("quote_errors:", (raw.get("quote_errors") or [])[:3])
    print("hallucinations:", (raw.get("hallucinations") or [])[:3])