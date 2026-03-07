import json
from pathlib import Path

EXTRACTED = Path("data/6_curation/validation_gpt5/eval_validator_extracted.jsonl")

needle = "I don't know based on the provided text."

hits = []
with EXTRACTED.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        raw = r.get("raw") or {}
        unsup = raw.get("unsupported_claims") or []
        if any(needle == x for x in unsup):
            hits.append(r)

print("Hits:", len(hits))
print("First 20 custom_ids:")
for r in hits[:20]:
    print(" ", r.get("custom_id"))