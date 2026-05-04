import json
import re

OLD_PATH = r"D:\SL_Medical_Corpus\data\10_instruction_eval\judged_rows.json"
NEW_PATH = r"D:\SL_Medical_Corpus\data\10_instruction_eval_retry\judged_rows.json"
OUT_PATH = r"D:\SL_Medical_Corpus\data\10_instruction_eval\judged_rows_merged.json"

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()

def row_key(item):
    parts = []
    for turn in item["conversations"]:
        parts.append(f"{turn['from']}::{normalize_text(turn['value'])}")
    conv_key = " || ".join(parts)
    return f"{item.get('source_record_id')}##{item.get('example_type')}##{conv_key}"

with open(OLD_PATH, "r", encoding="utf-8") as f:
    old_rows = json.load(f)

with open(NEW_PATH, "r", encoding="utf-8") as f:
    new_rows = json.load(f)

merged = []
seen = set()

for item in old_rows + new_rows:
    key = row_key(item)
    if key not in seen:
        seen.add(key)
        merged.append(item)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"Old judged rows: {len(old_rows)}")
print(f"Retry judged rows: {len(new_rows)}")
print(f"Merged judged rows: {len(merged)}")
print(f"Saved: {OUT_PATH}")