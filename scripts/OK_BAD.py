import json
from pathlib import Path

IN_PATH = Path(r"data\6_curation\idk_regen_eval\eval_idk_regen_batch_output.jsonl")
OUT_OK  = IN_PATH.with_name("eval_idk_regen_extracted.jsonl")
OUT_BAD = IN_PATH.with_name("eval_idk_regen_failed.jsonl")

def extract_text(body: dict) -> str | None:
    # Newer Responses API usually returns output: [{content:[{type:'output_text', text:'...'}]}]
    out = body.get("output")
    if isinstance(out, list):
        chunks = []
        for item in out:
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                        t = c.get("text")
                        if t:
                            chunks.append(t)
        if chunks:
            return "\n".join(chunks).strip()

    # Fallbacks
    t = body.get("output_text")
    if isinstance(t, str) and t.strip():
        return t.strip()

    return None

ok = bad = 0

with IN_PATH.open("r", encoding="utf-8") as fin, \
     OUT_OK.open("w", encoding="utf-8") as fok, \
     OUT_BAD.open("w", encoding="utf-8") as fbad:

    for line in fin:
        row = json.loads(line)

        custom_id = row.get("custom_id")
        resp = row.get("response") or {}
        status_code = resp.get("status_code")
        body = resp.get("body") or {}

        text = extract_text(body)

        record = {
            "custom_id": custom_id,
            "status_code": status_code,
            "request_id": resp.get("request_id"),
            "response_id": body.get("id"),
            "created_at": body.get("created_at"),
            "text": text,
        }

        if status_code == 200 and text:
            fok.write(json.dumps(record, ensure_ascii=False) + "\n")
            ok += 1
        else:
            fbad.write(json.dumps(record, ensure_ascii=False) + "\n")
            bad += 1

print("OK:", ok)
print("BAD:", bad)
print("Wrote:", OUT_OK)
print("Wrote:", OUT_BAD)