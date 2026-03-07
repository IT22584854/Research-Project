import json
from pathlib import Path

EVAL_IN = Path(r"data\4_instruction\eval_multiturn_VALID.jsonl")
CANDS  = Path(r"data\6_curation\idk_regen_eval\eval_idk_candidates.jsonl")

OUT_DIR = Path(r"data\6_curation\idk_regen_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_OUT = OUT_DIR / "eval_idk_regen_batch_input.jsonl"

# Load candidates: (id, assistant_turn_idx)
targets = set()
with CANDS.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        targets.add((str(r["id"]), int(r["assistant_turn_idx"])))

print("Targets:", len(targets))

wrote = 0

with EVAL_IN.open("r", encoding="utf-8") as fin, BATCH_OUT.open("w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        rid = str(obj.get("id"))
        msgs = obj.get("messages", [])

        # enumerate assistant turns
        a_idx = -1
        for mi, m in enumerate(msgs):
            if m.get("role") == "assistant":
                a_idx += 1
                if (rid, a_idx) not in targets:
                    continue

                # We regenerate by sending the conversation up to THIS assistant turn as PENDING
                # Keep system + all messages up to just before this assistant message
                prompt_msgs = msgs[:mi]  # exclude the IDK assistant message itself

                # custom_id ties output back to exact turn
                custom_id = f"idkregenE-{rid}-{a_idx}"

                req = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        # Choose the same model you used in training regen
                        "model": "gpt-5",
                        "input": prompt_msgs,
                        # Optional: keep it grounded and concise
                        "text": {
                            "format": {"type": "text"}
                        },
                    }
                }

                fout.write(json.dumps(req, ensure_ascii=False) + "\n")
                wrote += 1

print("Wrote batch requests:", wrote)
print("Batch file:", BATCH_OUT)