import json

def dedupe_file(path):
    seen = set()
    kept = []
    removed = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rid = obj.get("id")
            if rid in seen:
                removed += 1
                continue
            seen.add(rid)
            kept.append(obj)

    with open(path, "w", encoding="utf-8") as f:
        for obj in kept:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"{path}: removed {removed} duplicates")

dedupe_file("data/4_instruction/train_multiturn.jsonl")
dedupe_file("data/4_instruction/eval_multiturn.jsonl")
dedupe_file("data/6_curation/curation_pool_1000.jsonl")