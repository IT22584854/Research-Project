import json
import re
from pathlib import Path

RAW_DATASET_PATH = r"D:\SL_Medical_Corpus\generated_dataset_full\sharegpt_dataset.json"
JUDGED_ROWS_PATH = r"D:\SL_Medical_Corpus\data\10_instruction_eval\judged_rows.json"
OUT_PATH = r"D:\SL_Medical_Corpus\data\10_instruction_eval\retry_unjudged_dataset.json"

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()

def is_valid_example(item):
    if "conversations" not in item or not isinstance(item["conversations"], list):
        return False
    conv = item["conversations"]
    if len(conv) < 2:
        return False
    for turn in conv:
        if turn.get("from") not in {"human", "gpt"}:
            return False
        if not str(turn.get("value", "")).strip():
            return False
    return True

def conv_key_from_training_item(item):
    parts = []
    for turn in item["conversations"]:
        parts.append(f"{turn['from']}::{normalize_text(turn['value'])}")
    return " || ".join(parts)

def conv_key_from_judged_row(item):
    parts = []
    for turn in item["conversations"]:
        parts.append(f"{turn['from']}::{normalize_text(turn['value'])}")
    return " || ".join(parts)

with open(RAW_DATASET_PATH, "r", encoding="utf-8") as f:
    raw_dataset = json.load(f)

with open(JUDGED_ROWS_PATH, "r", encoding="utf-8") as f:
    judged_rows = json.load(f)

cleaned_dataset = []
seen = set()

for item in raw_dataset:
    if not is_valid_example(item):
        continue
    key = conv_key_from_training_item(item)
    if key in seen:
        continue
    seen.add(key)
    cleaned_dataset.append(item)

judged_keys = set(conv_key_from_judged_row(x) for x in judged_rows)

retry_items = [x for x in cleaned_dataset if conv_key_from_training_item(x) not in judged_keys]

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(retry_items, f, indent=2, ensure_ascii=False)

print(f"Raw dataset: {len(raw_dataset)}")
print(f"Structurally cleaned dataset: {len(cleaned_dataset)}")
print(f"Already judged: {len(judged_rows)}")
print(f"Retry items: {len(retry_items)}")
print(f"Saved: {OUT_PATH}")