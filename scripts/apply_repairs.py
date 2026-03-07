#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, re
from pathlib import Path

INVALID_IN = Path("data/6_curation/validation_gpt5/curation_pool_1000_invalid.jsonl")
REPAIR_OUT = Path("data/6_curation/validation_gpt5/repair_batch_output.jsonl")

OUT_REPAIRED = Path("data/6_curation/validation_gpt5/curation_pool_1000_repaired.jsonl")

def parse_custom_id(custom_id: str) -> str:
    s = (custom_id or "").strip()
    if s.startswith("repair-"):
        s = s[7:]
    s = re.sub(r"-\d+$", "", s)
    return s

def extract_output_text(batch_line: dict) -> str:
    body = ((batch_line.get("response") or {}).get("body") or {})
    # Responses API may include output_text sometimes; otherwise output -> message parts.
    if isinstance(body.get("output_text"), str):
        return body["output_text"].strip()

    out = body.get("output")
    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                            t = part.get("text")
                            if isinstance(t, str) and t.strip():
                                return t.strip()
    return ""

def safe_json_obj(txt: str) -> dict | None:
    txt = (txt or "").strip()
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

# Load repair results by record id
repairs = {}
with REPAIR_OUT.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        rid = parse_custom_id(obj.get("custom_id"))
        txt = extract_output_text(obj)
        jo = safe_json_obj(txt)
        if isinstance(jo, dict) and "a1" in jo and "a2" in jo:
            repairs[rid] = jo

n_in = 0
n_applied = 0

with INVALID_IN.open("r", encoding="utf-8") as fin, OUT_REPAIRED.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        n_in += 1
        rec = json.loads(line)
        rid = rec.get("id")
        r = repairs.get(rid)

        if r:
            # apply fixes
            msgs = rec.get("messages", [])
            if len(msgs) >= 5:
                msgs[2]["content"] = str(r.get("a1", "")).strip()
                msgs[4]["content"] = str(r.get("a2", "")).strip()
                rec["messages"] = msgs
            rec["_repair_gpt5"] = {
                "applied": True,
                "repair_notes": r.get("repair_notes", ""),
            }
            n_applied += 1
        else:
            rec["_repair_gpt5"] = {"applied": False}

        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("Invalid records:", n_in)
print("Repairs applied:", n_applied)
print("Wrote:", OUT_REPAIRED)