import json
from pathlib import Path

ORIG_VALIDATED = Path("data/6_curation/validation_gpt5/curation_pool_1000_validated.jsonl")
REPAIRED_VALIDATED = Path("data/6_curation/validation_gpt5/curation_pool_1000_repaired_validated.jsonl")

OUT = Path("data/6_curation/validation_gpt5/curation_pool_1000_FINAL_TRAIN.jsonl")

def iter_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def verdict(obj, key):
    return str(((obj.get(key) or {}).get("verdict") or "")).upper()

seen = set()
kept = 0

with OUT.open("w", encoding="utf-8") as fout:
    # 1) original valids
    for obj in iter_jsonl(ORIG_VALIDATED):
        if verdict(obj, "_validation_gpt5") != "VALID":
            continue
        rid = obj.get("id")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += 1

    # 2) repaired valids (use _reval_gpt5)
    for obj in iter_jsonl(REPAIRED_VALIDATED):
        if verdict(obj, "_reval_gpt5") != "VALID":
            continue
        rid = obj.get("id")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += 1

print("Final training records:", kept)
print("Wrote:", OUT)