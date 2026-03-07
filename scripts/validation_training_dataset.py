import json
from collections import Counter

ids = []

with open("data/4_instruction/eval_multiturn.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            ids.append(obj["id"])

counts = Counter(ids)
dupes = [k for k, v in counts.items() if v > 1]

print("Duplicate ID count:", len(dupes))
print("Total duplicates:", sum(counts[k] - 1 for k in dupes))
print("Example duplicate IDs:", dupes[:5])