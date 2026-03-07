import json
from collections import Counter

IDK = "I don't know based on the provided text."

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def audit(rows, name):
    print(f"\n=== {name} ===")
    print("Total rows:", len(rows))

    ids = [r.get("id") for r in rows]
    c_ids = Counter(ids)
    dupes = sum(1 for v in c_ids.values() if v > 1)
    print("Unique ids:", len(c_ids), "| Duplicate id keys:", dupes)

    tasks = Counter(r.get("task") for r in rows)
    langs = Counter(r.get("lang") for r in rows)
    styles = Counter(r.get("style") for r in rows)
    splits = Counter(r.get("split") for r in rows)
    msglens = Counter(len(r.get("messages", [])) for r in rows)

    print("Splits inside file:", dict(splits))
    print("Top tasks:", tasks.most_common(10))
    print("Langs:", langs.most_common())
    print("Styles:", styles.most_common())
    print("Message lengths:", dict(msglens))

    bad = 0
    idk = 0
    for r in rows:
        msgs = r.get("messages", [])
        if len(msgs) != 5:
            bad += 1
            continue
        roles = [m.get("role") for m in msgs]
        if roles != ["system","user","assistant","user","assistant"]:
            bad += 1
        if msgs[2].get("content","").strip() == IDK: idk += 1
        if msgs[4].get("content","").strip() == IDK: idk += 1

    print("Bad structure rows:", bad)
    print("IDK answers (turns counted):", idk)

train = load_jsonl("data/4_instruction/train_multiturn.jsonl")
eval_  = load_jsonl("data/4_instruction/eval_multiturn.jsonl")

audit(train, "TRAIN")
audit(eval_, "EVAL")

train_docs = set(r["doc_id"] for r in train)
eval_docs  = set(r["doc_id"] for r in eval_)
overlap = train_docs & eval_docs

print("\n=== Doc leakage check (doc_id overlap train vs eval) ===")
print("Train doc_ids:", len(train_docs))
print("Eval doc_ids:", len(eval_docs))
print("Overlap:", len(overlap))
if overlap:
    print("Example overlaps:", list(overlap)[:20])
