import json

path = r"D:\SL_Medical_Corpus\data\7_gold_standard\gold_eval_REPAIRED_99_FINAL.jsonl"

keywords = [
    "Crawl4AI Error",
    "not fully supported",
    "XML compatible",
    "All strings must be XML compatible"
]

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        text = " ".join(m["content"] for m in rec["messages"])

        for k in keywords:
            if k.lower() in text.lower():
                print(rec["id"], rec["doc_id"], k)