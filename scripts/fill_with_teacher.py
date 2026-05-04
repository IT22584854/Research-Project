#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage B (Multi-turn): pending_multiturn.jsonl -> train_multiturn.jsonl + eval_multiturn.jsonl

Key improvements:
- EVAL_FIRST option: process eval split first so eval grows early
- Split filters: SPLIT_MODE=train|eval|all (no need to create separate pending files)
- Separate flush thresholds: FLUSH_EVERY_EVAL smaller so eval isn't lost on interruptions
- Low-signal context gating: prevents hallucinations on References/contact-only chunks
- Robust JSON parsing + retries + JSON mode (response_format=json_object when supported)
- Optional hybrid models by task (USE_HYBRID_MODELS=1)
- Graceful stop on insufficient_quota/429: flush buffers and exit cleanly
- Task-aware max_tokens: smaller for safe tasks to speed up + reduce cost
- Progress checkpoint file: data/4_instruction/fill_progress.json (optional)

Expected message format per example:
system
user (contains Context)
assistant "[PENDING]" or filled
user (follow-up)
assistant "[PENDING]" or filled
"""

import os
import re
import json
import time
import random
from typing import List, Dict, Set, Optional, Iterator, Tuple

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# -----------------------------
# Paths
# -----------------------------
PENDING_PATH = os.getenv("PENDING_PATH", "data/4_instruction/pending_multiturn.jsonl")
OUT_TRAIN    = os.getenv("OUT_TRAIN", "data/4_instruction/train_multiturn.jsonl")
OUT_EVAL     = os.getenv("OUT_EVAL",  "data/4_instruction/eval_multiturn.jsonl")
PROGRESS_PATH = os.getenv("PROGRESS_PATH", "data/4_instruction/fill_progress.json")

# -----------------------------
# Teacher settings
# -----------------------------
TEACHER_MODEL        = os.getenv("TEACHER_MODEL", "gpt-4o")
TEACHER_MODEL_STRONG = os.getenv("TEACHER_MODEL_STRONG", "gpt-4o")       # risky tasks
TEACHER_MODEL_FAST   = os.getenv("TEACHER_MODEL_FAST", "gpt-4o-mini")    # safe tasks
USE_HYBRID_MODELS    = os.getenv("USE_HYBRID_MODELS", "0").strip() == "1"

MAX_ITEMS      = int(os.getenv("MAX_ITEMS", "0"))  # 0 = all
MAX_TOKENS     = int(os.getenv("TEACHER_MAX_TOKENS", "512"))
TEMPERATURE    = float(os.getenv("TEACHER_TEMPERATURE", "0.1"))
MAX_RETRIES    = int(os.getenv("MAX_RETRIES", "5"))

# Flush behavior (separate train vs eval)
FLUSH_EVERY_TRAIN = int(os.getenv("FLUSH_EVERY_TRAIN", "25"))
FLUSH_EVERY_EVAL  = int(os.getenv("FLUSH_EVERY_EVAL", "5"))

# Process eval first (two-pass streaming)
EVAL_FIRST = os.getenv("EVAL_FIRST", "0").strip() == "1"

# Split filter mode: all|train|eval
SPLIT_MODE = os.getenv("SPLIT_MODE", "all").strip().lower()
if SPLIT_MODE not in ("all", "train", "eval"):
    SPLIT_MODE = "all"

# Use JSON mode if supported by model/sdk
USE_JSON_MODE = os.getenv("USE_JSON_MODE", "1").strip() == "1"

client = OpenAI()
IDK_TEXT = "I don't know based on the provided text."

# -----------------------------
# IO helpers
# -----------------------------
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def append_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_existing_ids(paths: List[str]) -> Set[str]:
    ids: Set[str] = set()
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            for row in read_jsonl(p):
                rid = row.get("id")
                if rid:
                    ids.add(rid)
        except Exception:
            # tolerate partial/corrupt tail
            pass
    return ids

def save_progress(meta: dict):
    try:
        os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
        with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# -----------------------------
# Validation / filters
# -----------------------------
def forbid_patterns(text: str) -> bool:
    bad = [
        "as an ai", "i can't access", "i cannot access",
        "tool", "browser", "database", "supabase", "hospital_lookup"
    ]
    t = (text or "").lower()
    return any(b in t for b in bad)

def validate(task: str, answer: str) -> Optional[str]:
    if not answer or len(answer.strip()) < 2:
        return "empty"
    if forbid_patterns(answer):
        return "forbidden"
    if task == "classify_doc_type" and len(answer.strip().split()) > 3:
        return "classify_too_long"
    return None

def normalize_answer(ans: str) -> str:
    ans = (ans or "").strip()
    return ans if ans else IDK_TEXT

# -----------------------------
# Robust JSON parsing helper
# -----------------------------
def parse_json_strict(txt: str) -> dict:
    txt = (txt or "").strip()
    # direct parse
    try:
        return json.loads(txt)
    except Exception:
        pass

    # extract {...}
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if m:
        blob = m.group(0)
        blob = blob.replace("“", "\"").replace("”", "\"").replace("’", "'")
        return json.loads(blob)

    raise ValueError("No JSON object found in model output")

# -----------------------------
# Context quality checks (prevents hallucinations)
# -----------------------------
REF_LIKE = re.compile(r"(##\s*references\b|^\s*references\b)", re.IGNORECASE | re.MULTILINE)
CITATION_HEAVY_LINE = re.compile(r"(doi:|pmid|https?://|Vol\.|pp\.|\(\d{4}\)|et al\.)", re.IGNORECASE)
PHONE_EMAIL = re.compile(r"(\+?\d[\d\s\-]{6,}|\bemail\b|\be-?mail\b|\bfax\b|\btele\b)", re.IGNORECASE)

def extract_context(user1: str) -> str:
    m = re.search(r"\nContext:\n(.*)$", user1, flags=re.DOTALL)
    return (m.group(1) if m else "").strip()

def is_low_signal_context(ctx: str) -> bool:
    if not ctx:
        return True
    if REF_LIKE.search(ctx):
        return True

    lines = [l.strip() for l in ctx.splitlines() if l.strip()]
    if not lines:
        return True

    citation_lines = sum(1 for l in lines if CITATION_HEAVY_LINE.search(l))
    if citation_lines / max(1, len(lines)) > 0.45:
        return True

    contact_lines = sum(1 for l in lines if PHONE_EMAIL.search(l))
    longish = sum(1 for l in lines if len(l.split()) >= 10 and not CITATION_HEAVY_LINE.search(l))
    if contact_lines >= 2 and longish == 0:
        return True

    if len(ctx) < 300:
        return True

    return False

# -----------------------------
# Model selection (optional hybrid)
# -----------------------------
SAFE_TASKS = {"extractive_qa", "short_answer_qa", "bullet_points", "classify_doc_type", "mixed_language_qa"}
RISKY_TASKS = {"summarize", "rewrite_patient_friendly"}

def pick_model(task: str) -> str:
    if not USE_HYBRID_MODELS:
        return TEACHER_MODEL
    if task in RISKY_TASKS:
        return TEACHER_MODEL_STRONG
    return TEACHER_MODEL_FAST

def pick_max_tokens(task: str) -> int:
    # task-aware token cap to reduce spend and speed up
    if task in SAFE_TASKS:
        return min(MAX_TOKENS, 256)
    if task in RISKY_TASKS:
        return min(MAX_TOKENS, 512)
    return MAX_TOKENS

# -----------------------------
# Teacher call (single-call for both turns)
# -----------------------------
def teacher_both_turns(task: str, system_text: str, user1: str, user2: str) -> Dict[str, str]:
    # soften summarize follow-up: "up to 3" avoids unnecessary IDK
    user2_fixed = user2
    if task == "summarize":
        user2_fixed = re.sub(
            r"List\s+3\s+key\s+takeaways",
            "List up to 3 key takeaways",
            user2_fixed,
            flags=re.IGNORECASE
        )

    grounding = (
        "STRICT RULES (must follow):\n"
        "1) Use ONLY the provided Context.\n"
        "2) Do NOT add medical advice or facts not present in the Context.\n"
        f"3) If the Context does not contain the answer, reply exactly: \"{IDK_TEXT}\"\n"
        "4) Do NOT mention tools, browsing, databases, or external sources.\n"
        "5) Follow the language requested.\n"
        "6) If the Context is only references/citations/contact info with no usable content for the task, reply exactly with the IDK sentence.\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY valid JSON (double quotes) with keys \"a1\" and \"a2\".\n"
        "\"a1\" = assistant reply to USER_TURN_1.\n"
        "\"a2\" = assistant reply to USER_TURN_2.\n"
        "No extra text. No markdown fences."
    )

    combined_user = (
        "You will answer a 2-turn conversation.\n\n"
        "USER_TURN_1:\n"
        f"{user1}\n\n"
        "USER_TURN_2:\n"
        f"{user2_fixed}\n"
    )

    model = pick_model(task)
    max_tokens = pick_max_tokens(task)

    kwargs = {}
    if USE_JSON_MODE:
        # If supported, this almost eliminates malformed JSON outputs.
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (system_text + "\n\n" + grounding).strip()},
            {"role": "user", "content": combined_user},
        ],
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
        **kwargs,
    )

    txt = (resp.choices[0].message.content or "").strip()
    obj = parse_json_strict(txt)

    return {
        "a1": str(obj.get("a1", "")).strip(),
        "a2": str(obj.get("a2", "")).strip(),
    }

# -----------------------------
# Conversation filling
# -----------------------------
def fill_conversation(ex: dict) -> Tuple[dict, bool]:
    """
    Returns: (example, filled_anything)
    """
    task = ex.get("task", "")
    msgs = ex.get("messages", [])

    if len(msgs) < 5:
        return ex, False
    if msgs[0].get("role") != "system": return ex, False
    if msgs[1].get("role") != "user": return ex, False
    if msgs[2].get("role") != "assistant": return ex, False
    if msgs[3].get("role") != "user": return ex, False
    if msgs[4].get("role") != "assistant": return ex, False

    a1_pending = (msgs[2].get("content", "").strip() == "[PENDING]")
    a2_pending = (msgs[4].get("content", "").strip() == "[PENDING]")
    if not a1_pending and not a2_pending:
        return ex, False

    system_text = msgs[0].get("content", "")
    user1 = msgs[1].get("content", "")
    user2 = msgs[3].get("content", "")

    # Hard guard: if context is low-signal and task is risky OR bullet_points, force IDK
    ctx = extract_context(user1)
    if (task in RISKY_TASKS or task == "bullet_points") and is_low_signal_context(ctx):
        if a1_pending: msgs[2]["content"] = IDK_TEXT
        if a2_pending: msgs[4]["content"] = IDK_TEXT
        ex["messages"] = msgs
        return ex, True

    for attempt in range(MAX_RETRIES):
        try:
            both = teacher_both_turns(task, system_text, user1, user2)

            a1 = normalize_answer(both.get("a1"))
            a2 = normalize_answer(both.get("a2"))

            err1 = validate(task, a1)
            err2 = validate(task, a2)

            if (err1 or err2) and attempt < MAX_RETRIES - 1:
                time.sleep((2 ** attempt) + random.random())
                continue

            if err1: a1 = IDK_TEXT
            if err2: a2 = IDK_TEXT

            if a1_pending: msgs[2]["content"] = a1
            if a2_pending: msgs[4]["content"] = a2

            ex["messages"] = msgs
            return ex, True

        except (json.JSONDecodeError, ValueError):
            if attempt == MAX_RETRIES - 1:
                if a1_pending: msgs[2]["content"] = IDK_TEXT
                if a2_pending: msgs[4]["content"] = IDK_TEXT
                ex["messages"] = msgs
                return ex, True
            time.sleep((2 ** attempt) + random.random())
            continue

        except Exception as e:
            # Graceful stop for quota exhaustion (prevents losing buffered items)
            msg = str(e).lower()
            if "insufficient_quota" in msg:
                raise RuntimeError("INSUFFICIENT_QUOTA")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep((2 ** attempt) + random.random())

    return ex, False

# -----------------------------
# Iteration order
# -----------------------------
def iter_pending_eval_then_train() -> Iterator[dict]:
    # Pass 1: eval items
    for ex in read_jsonl(PENDING_PATH):
        if ex.get("split") == "eval":
            yield ex
    # Pass 2: train items
    for ex in read_jsonl(PENDING_PATH):
        if ex.get("split") != "eval":
            yield ex

def split_allowed(ex: dict) -> bool:
    s = ex.get("split")
    if SPLIT_MODE == "all":
        return True
    if SPLIT_MODE == "eval":
        return s == "eval"
    if SPLIT_MODE == "train":
        return s != "eval"
    return True

# -----------------------------
# Main
# -----------------------------
def main():
    done_ids = load_existing_ids([OUT_TRAIN, OUT_EVAL])
    print(f"Already filled conversations: {len(done_ids)}")
    print(f"SPLIT_MODE={SPLIT_MODE} | EVAL_FIRST={EVAL_FIRST} | USE_JSON_MODE={USE_JSON_MODE} | USE_HYBRID_MODELS={USE_HYBRID_MODELS}")

    buf_train: List[dict] = []
    buf_eval: List[dict] = []
    processed = 0
    written = 0

    stream = iter_pending_eval_then_train() if EVAL_FIRST else read_jsonl(PENDING_PATH)

    t0 = time.time()

    try:
        for ex in stream:
            if not split_allowed(ex):
                continue

            ex_id = ex.get("id")
            if ex_id and ex_id in done_ids:
                continue

            ex, filled = fill_conversation(ex)
            if not filled:
                continue

            if ex.get("split") == "eval":
                buf_eval.append(ex)
            else:
                buf_train.append(ex)

            processed += 1

            # Flush eval frequently to avoid losing it on interruptions
            if len(buf_eval) >= FLUSH_EVERY_EVAL:
                append_jsonl(OUT_EVAL, buf_eval)
                written += len(buf_eval)
                buf_eval = []

            if len(buf_train) >= FLUSH_EVERY_TRAIN:
                append_jsonl(OUT_TRAIN, buf_train)
                written += len(buf_train)
                buf_train = []

            if processed % 25 == 0:
                elapsed = time.time() - t0
                save_progress({
                    "processed_this_run": processed,
                    "written_this_run": written,
                    "elapsed_sec": round(elapsed, 1),
                    "pending_path": PENDING_PATH,
                    "split_mode": SPLIT_MODE,
                    "eval_first": EVAL_FIRST,
                    "use_hybrid_models": USE_HYBRID_MODELS,
                    "teacher_model": TEACHER_MODEL,
                    "teacher_fast": TEACHER_MODEL_FAST,
                    "teacher_strong": TEACHER_MODEL_STRONG,
                })
                print(f"Processed {processed} conversations | Written so far: {written} | Elapsed: {elapsed/60:.1f} min")

            if MAX_ITEMS and processed >= MAX_ITEMS:
                break

    except RuntimeError as e:
        # Clean stop on quota issues
        if str(e) == "INSUFFICIENT_QUOTA":
            print("️ Stopping: insufficient_quota. Flushing buffers and exiting cleanly...")

    finally:
        # Final flush ALWAYS
        if buf_eval:
            append_jsonl(OUT_EVAL, buf_eval)
            written += len(buf_eval)
        if buf_train:
            append_jsonl(OUT_TRAIN, buf_train)
            written += len(buf_train)

        elapsed = time.time() - t0
        save_progress({
            "processed_this_run": processed,
            "written_this_run": written,
            "elapsed_sec": round(elapsed, 1),
            "completed": True,
        })

        print(f"Done. Newly processed: {processed} | Newly written: {written}")
        print(f"Elapsed: {elapsed/60:.1f} min")
        print(f"Outputs:\n  {OUT_TRAIN}\n  {OUT_EVAL}")

if __name__ == "__main__":
    main()