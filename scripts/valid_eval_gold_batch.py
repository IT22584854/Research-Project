#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

IN_PATH = Path(r"D:\SL_Medical_Corpus\data\7_gold_standard\gold_eval_REPAIRED_99_FINAL_CLEAN.jsonl")
OUT_VALID = Path(r"D:\SL_Medical_Corpus\data\7_gold_standard\validation_final\gold_eval_FINAL_VALID.jsonl")
OUT_INVALID = Path(r"D:\SL_Medical_Corpus\data\7_gold_standard\validation_final\gold_eval_FINAL_INVALID.jsonl")

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s+", " ", s)
    return s

def extract_context(first_user_text: str) -> str:
    m = re.search(r"(?is)\bcontext\s*:\s*(.*)", first_user_text or "")
    return m.group(1).strip() if m else ""

def validate_record(rec: dict):
    if not isinstance(rec, dict):
        return False, "not_dict"

    if rec.get("task") != "short_answer_qa":
        return False, "wrong_task"

    lang = rec.get("lang")
    if lang not in {"en", "si", "ta"}:
        return False, "wrong_lang"

    msgs = rec.get("messages", [])
    if len(msgs) != 5:
        return False, "wrong_message_count"

    roles = [m.get("role") for m in msgs]
    if roles != ["system", "user", "assistant", "user", "assistant"]:
        return False, "wrong_role_pattern"

    first_user = msgs[1].get("content", "")
    answer_1 = normalize(msgs[2].get("content", ""))
    answer_2 = normalize(msgs[4].get("content", ""))
    context = normalize(extract_context(first_user))

    if not context:
        return False, "missing_context_block"
    if not answer_1:
        return False, "empty_main_answer"
    if not answer_2:
        return False, "empty_evidence_answer"

    if len(answer_1.split()) > 40:
        return False, "main_answer_too_long"

    evidence = answer_2.strip()

    # allow surrounding quotes
    if (
        (evidence.startswith('"') and evidence.endswith('"')) or
        (evidence.startswith("'") and evidence.endswith("'"))
    ):
        evidence = evidence[1:-1].strip()

    if evidence not in context:
        return False, "evidence_not_in_context"

    return True, "ok"

def main():
    valid = []
    invalid = []

    for rec in read_jsonl(IN_PATH):
        ok, reason = validate_record(rec)
        rec["_validation"] = reason
        if ok:
            valid.append(rec)
        else:
            invalid.append(rec)

    write_jsonl(OUT_VALID, valid)
    write_jsonl(OUT_INVALID, invalid)

    print("VALID:", len(valid))
    print("INVALID:", len(invalid))
    print("Wrote valid ->", OUT_VALID)
    print("Wrote invalid ->", OUT_INVALID)

if __name__ == "__main__":
    main()