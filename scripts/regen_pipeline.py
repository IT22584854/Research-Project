#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
End-to-end dataset cleaning + selective regeneration pipeline.

1) Remove Crawl4AI error records
2) Keep TRUE refusals
3) Flag likely FALSE refusals for regen for selected tasks
4) Create Batch input JSONL to regenerate only refused assistant turns
5) Merge regenerated turns back into final dataset

Usage (example):
python regen_pipeline.py ^
  --in_jsonl "D:\SL_Medical_Corpus\data\4_instruction\train_multiturn_FINAL_PATCHED_CLEAN_REFUSAL_LOCALIZED_FINAL.jsonl" ^
  --work_dir "D:\SL_Medical_Corpus\data\4_instruction\regen_work" ^
  --model "gpt-5" ^
  --min_context_words 60

Merge-only:
python regen_pipeline.py --merge_only --work_dir "D:\SL_Medical_Corpus\data\4_instruction\regen_work"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Settings you can tweak
# ----------------------------

CRAWL_ERROR_KEYWORDS = [
    "Crawl4AI Error",
    "not fully supported",
    "All strings must be XML compatible",
    "XML compatible",
]

REGEN_TASKS_DEFAULT = {"rewrite_patient_friendly", "extractive_qa", "bullet_points"}

REFUSAL_PATTERNS = [
    r"\bi\s*don'?t\s*know\b",
    r"\bcannot\s+answer\b",
    r"\bcan'?t\s+answer\b",
    r"\bunable\s+to\s+answer\b",
    r"\bbased\s+on\s+the\s+provided\s+text\b",
    # Sinhala-ish
    r"මට\s+ලබා\s+දී\s+ඇති",
    r"ලබා\s+දී\s+ඇති\s+පෙළ",
    # Tamil-ish
    r"நான்\s+வழங்கப்பட்ட",
    r"கொடுக்கப்பட்ட\s+உரை",
]

REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), flags=re.IGNORECASE)
MIN_CONTEXT_WORDS_DEFAULT = 60


# ----------------------------
# JSONL helpers
# ----------------------------

def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# Core detection logic
# ----------------------------

def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text or ""))


def word_count(text: str) -> int:
    return len((text or "").split())


def record_all_text(rec: dict) -> str:
    """Concatenate all message contents for robust keyword scanning."""
    return "\n".join((m.get("content") or "") for m in rec.get("messages", []))


def is_crawl_error_record(rec: dict) -> bool:
    #  case-insensitive keyword scan
    full = record_all_text(rec).lower()
    return any(k.lower() in full for k in CRAWL_ERROR_KEYWORDS)


def extract_context_block(user_text: str) -> str:
    """
    If the user message contains a label like 'Context:' (or Document/Text/Passage),
    return only what follows. Otherwise return the full user text.

    Helps avoid inflated word counts from boilerplate instructions.
    """
    m = re.search(r"(?is)\b(context|document|text|passage)\s*:\s*(.*)", user_text or "")
    return (m.group(2).strip() if m else (user_text or ""))


def get_primary_context(rec: dict) -> str:
    """
    Assumes messages[1] is the primary user message that contains context.
    If your format changes, adjust here.
    """
    msgs = rec.get("messages", [])
    if len(msgs) >= 2 and msgs[1].get("role") == "user":
        return extract_context_block(msgs[1].get("content", "") or "")
    # fallback: scan first user message
    for m in msgs:
        if m.get("role") == "user":
            return extract_context_block(m.get("content", "") or "")
    return ""


def get_assistant_indices(rec: dict) -> List[int]:
    return [i for i, m in enumerate(rec.get("messages", [])) if m.get("role") == "assistant"]


# ----------------------------
# Regen prompt shaping
# ----------------------------

def build_regen_instruction(task: str) -> str:
    if task == "rewrite_patient_friendly":
        return (
            "Rewrite the provided content in patient-friendly simple language. "
            "Do NOT add new facts. Use ONLY information present in the context. "
            "Do NOT refuse unless the context contains no meaningful informational content."
        )
    if task == "extractive_qa":
        return (
            "Answer using an EXACT QUOTE copied from the context (verbatim). "
            "If the exact answer text is not present in the context, reply with: "
            "\"I don't know based on the provided text.\""
        )
    if task == "bullet_points":
        return (
            "Extract 5–8 key points as bullet points using ONLY the context. "
            "Do NOT add facts. If the context is only navigation/links and lacks informational content, "
            "reply with: \"I don't know based on the provided text.\""
        )
    return (
        "Answer using only the provided context. Do not add new facts. "
        "If the answer is not in the context, reply: \"I don't know based on the provided text.\""
    )


def build_batch_request_item(rec: dict, assistant_msg_index: int, model: str) -> dict:
    """
    Build one batch line for POST /v1/responses.
    custom_id = <record_id>|<assistant_index>|<task>
    """
    msgs = rec.get("messages", [])
    task = str(rec.get("task", "unknown"))
    rid = str(rec.get("id", ""))

    if not msgs:
        raise ValueError(f"Record {rid} has no messages.")

    system_msg = (
        msgs[0].get("content")
        if msgs[0].get("role") == "system"
        else (
            "You are a helpful assistant. Use only the provided document content. Do not invent facts. "
            "If the answer is not in the text, say you don't know."
        )
    )

    # First user turn (usually contains Context)
    user1 = ""
    if len(msgs) > 1 and msgs[1].get("role") == "user":
        user1 = msgs[1].get("content", "") or ""
    else:
        for m in msgs:
            if m.get("role") == "user":
                user1 = m.get("content", "") or ""
                break

    instruction = build_regen_instruction(task)
    user1_aug = (
        f"{user1}\n\n"
        f"[Regeneration Instruction]\n{instruction}\n\n"
        f"Return only the assistant response for this turn."
    )

    asst_idxs = get_assistant_indices(rec)
    if not asst_idxs:
        raise ValueError(f"Record {rid} has no assistant messages.")

    # Determine if this is the 2nd assistant turn (multi-turn)
    is_second_asst = (len(asst_idxs) >= 2 and assistant_msg_index == asst_idxs[1])

    input_messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user1_aug},
    ]

    if is_second_asst:
        first_asst_text = msgs[asst_idxs[0]].get("content", "") or ""

        # follow-up user is usually immediately after first assistant
        follow_user = ""
        follow_user_idx = asst_idxs[0] + 1
        if follow_user_idx < len(msgs) and msgs[follow_user_idx].get("role") == "user":
            follow_user = msgs[follow_user_idx].get("content", "") or ""
        else:
            for m in msgs[2:]:
                if m.get("role") == "user":
                    follow_user = m.get("content", "") or ""
                    break

        input_messages.append({"role": "assistant", "content": first_asst_text})
        #  include regen instruction for turn-2 too
        input_messages.append(
            {
                "role": "user",
                "content": (
                    f"{follow_user}\n\n"
                    f"[Regeneration Instruction]\n{instruction}\n\n"
                    f"Return only the assistant response for this turn."
                ),
            }
        )

    custom_id = f"{rid}|{assistant_msg_index}|{task}"

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": input_messages,
        },
    }


# ----------------------------
# Pipeline steps
# ----------------------------

def split_clean_and_regen(
    in_path: Path,
    out_clean_path: Path,
    out_regen_path: Path,
    regen_tasks: set,
    min_context_words: int,
) -> Tuple[int, int, int]:
    """
    Removes crawl errors, then splits into:
    - CLEAN: keep as-is
    - NEEDS_REGEN: has refusal AND task in regen_tasks AND context seems answerable
    """
    removed_crawl = 0
    clean: List[dict] = []
    regen: List[dict] = []

    for rec in read_jsonl(in_path):
        if is_crawl_error_record(rec):
            removed_crawl += 1
            continue

        task = str(rec.get("task", "unknown"))
        context = get_primary_context(rec)

        asst_idxs = get_assistant_indices(rec)
        answers = [(rec["messages"][i].get("content", "") or "") for i in asst_idxs]

        has_refusal = any(is_refusal(a) for a in answers)
        answerable = word_count(context) >= min_context_words

        if task in regen_tasks and has_refusal and answerable:
            regen.append(rec)
        else:
            clean.append(rec)

    write_jsonl(out_clean_path, clean)
    write_jsonl(out_regen_path, regen)
    return len(clean), len(regen), removed_crawl


def build_batch_input(regen_records_path: Path, batch_input_path: Path, model: str) -> int:
    """
    Build batch_input.jsonl containing one request per refused assistant turn.
    """
    requests: List[dict] = []
    n_turns = 0

    for rec in read_jsonl(regen_records_path):
        msgs = rec.get("messages", [])
        for idx in get_assistant_indices(rec):
            text = (msgs[idx].get("content", "") or "")
            if is_refusal(text):
                requests.append(build_batch_request_item(rec, idx, model))
                n_turns += 1

    write_jsonl(batch_input_path, requests)
    return n_turns


# ----------------------------
# Responses API output extraction (robust)
# ----------------------------

def _extract_text_from_responses_api(resp: dict) -> Optional[str]:
    """
    Try a few known output shapes from the Responses API.
    Return a single string or None.
    """
    if not isinstance(resp, dict):
        return None

    t = resp.get("output_text")
    if isinstance(t, str) and t.strip():
        return t.strip()

    out = resp.get("output")
    if isinstance(out, list):
        chunks: List[str] = []
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                # common cases
                if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                    chunks.append(c["text"])
                    continue
                # occasional alt key
                if isinstance(c.get("output_text"), str):
                    chunks.append(c["output_text"])
        if chunks:
            return "\n".join(chunks).strip()

    # Sometimes it's nested in choices-like structures (rare, but safe)
    if isinstance(resp.get("text"), str) and resp["text"].strip():
        return resp["text"].strip()

    return None


def _unwrap_batch_line(line: dict) -> dict:
    """
    Batch output JSONL lines typically wrap the real response object like:
      line["response"]["body"] = <Responses API object>
    This function returns the most likely actual Responses API object dict.
    """
    if not isinstance(line, dict):
        return {}

    # Most common: {"response": {"status_code": 200, "body": {...}}}
    r = line.get("response")
    if isinstance(r, dict):
        b = r.get("body")
        if isinstance(b, dict):
            return b
        # sometimes already the body
        if any(k in r for k in ("output", "output_text", "id")):
            return r

    # Sometimes: {"body": {...}}
    b2 = line.get("body")
    if isinstance(b2, dict):
        # could be actual response or wrapper
        if "body" in b2 and isinstance(b2["body"], dict):
            return b2["body"]
        return b2

    # Fallback: maybe the line itself is the response object
    return line


def parse_batch_outputs(batch_output_path: Path) -> Dict[Tuple[str, int], str]:
    """
    Parse batch output JSONL.
    Returns mapping: (record_id, assistant_msg_index) -> regenerated_text
    """
    updates: Dict[Tuple[str, int], str] = {}

    for line in read_jsonl(batch_output_path):
        custom_id = line.get("custom_id", "")
        if not custom_id:
            continue

        parts = custom_id.split("|")
        if len(parts) < 2:
            continue

        rec_id = parts[0]
        try:
            asst_index = int(parts[1])
        except ValueError:
            continue

        resp_obj = _unwrap_batch_line(line)
        text = _extract_text_from_responses_api(resp_obj)

        # extra fallback
        if not text and isinstance(line.get("text"), str):
            text = line["text"].strip()

        if not text:
            continue

        updates[(rec_id, asst_index)] = text

    return updates


def apply_updates_to_record(rec: dict, updates: Dict[Tuple[str, int], str]) -> dict:
    rid = str(rec.get("id", ""))
    msgs = rec.get("messages", [])

    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        key = (rid, i)
        if key in updates:
            m["content"] = updates[key]

    rec["messages"] = msgs
    return rec


def merge_clean_and_regen(
    clean_path: Path,
    regen_records_path: Path,
    batch_output_path: Path,
    merged_out_path: Path,
) -> Tuple[int, int, int]:
    """
    Merge all clean records + regenerated records with applied updates.
    Returns: (clean_count, regen_count, updated_turns)
    """
    updates = parse_batch_outputs(batch_output_path)

    merged: List[dict] = []
    clean_count = 0
    regen_count = 0

    for rec in read_jsonl(clean_path):
        merged.append(rec)
        clean_count += 1

    updated_turns = 0
    for rec in read_jsonl(regen_records_path):
        rid = str(rec.get("id", ""))
        for i in get_assistant_indices(rec):
            if (rid, i) in updates:
                updated_turns += 1
        merged.append(apply_updates_to_record(rec, updates))
        regen_count += 1

    write_jsonl(merged_out_path, merged)
    return clean_count, regen_count, updated_turns


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, default="", help="Input JSONL dataset path")
    ap.add_argument("--work_dir", type=str, required=True, help="Working directory for intermediate files")
    ap.add_argument("--model", type=str, default="gpt-5", help="Model name for regeneration (Responses API)")
    ap.add_argument("--min_context_words", type=int, default=MIN_CONTEXT_WORDS_DEFAULT)
    ap.add_argument(
        "--regen_tasks",
        type=str,
        default="rewrite_patient_friendly,extractive_qa,bullet_points",
        help="Comma-separated list of tasks eligible for regeneration",
    )
    ap.add_argument("--merge_only", action="store_true", help="Only run merge step (after batch output exists)")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    clean_path = work_dir / "train_CLEAN_no_crawl_errors.jsonl"
    regen_path = work_dir / "train_NEEDS_REGEN.jsonl"
    batch_input_path = work_dir / "batch_input.jsonl"
    batch_output_path = work_dir / "batch_output.jsonl"
    merged_out_path = work_dir / "train_FINAL_no_crawl_regenfixed.jsonl"

    regen_tasks = {t.strip() for t in args.regen_tasks.split(",") if t.strip()}

    if args.merge_only:
        if not clean_path.exists() or not regen_path.exists():
            raise SystemExit("Missing clean/regen files. Run the full pipeline first.")
        if not batch_output_path.exists():
            raise SystemExit(f"Missing batch output file: {batch_output_path}")

        c, r, u = merge_clean_and_regen(clean_path, regen_path, batch_output_path, merged_out_path)
        print(f"Merged {c} clean + {r} regen records.")
        print(f"Applied updates to ~{u} assistant turns.")
        print(f"Wrote final -> {merged_out_path}")
        return

    if not args.in_jsonl:
        raise SystemExit("--in_jsonl is required unless --merge_only is used.")

    in_path = Path(args.in_jsonl)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    clean_n, regen_n, removed_crawl = split_clean_and_regen(
        in_path=in_path,
        out_clean_path=clean_path,
        out_regen_path=regen_path,
        regen_tasks=regen_tasks,
        min_context_words=args.min_context_words,
    )

    print(f"Removed crawl-error records: {removed_crawl}")
    print(f"Clean records: {clean_n}")
    print(f"Needs regeneration: {regen_n}")
    print(f"Wrote clean -> {clean_path}")
    print(f"Wrote regen -> {regen_path}")

    regen_turns = build_batch_input(regen_path, batch_input_path, model=args.model)
    print(f"Batch requests (refused turns to regenerate): {regen_turns}")
    print(f"Wrote batch input -> {batch_input_path}")

    print("\nNEXT STEP:")
    print("1) Submit batch_input.jsonl to OpenAI Batch API for /v1/responses.")
    print(f"2) Save the batch output JSONL as: {batch_output_path}")
    print("3) Run merge step:")
    print(f'   python regen_pipeline.py --merge_only --work_dir "{work_dir}"')


if __name__ == "__main__":
    main()