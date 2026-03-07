#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

# ----------------------------
# FILE PATHS
# ----------------------------
FILES = {
    "train_original_final": Path(r"D:\SL_Medical_Corpus\data\4_instruction\regen_work\train_FINAL_no_crawl_regenfixed.jsonl"),
    "train_clean": Path(r"D:\SL_Medical_Corpus\data\4_instruction\regen_work\train_FINAL_CLEAN_KEEP_TRUE_REFUSALS.jsonl"),
    "eval_clean": Path(r"D:\SL_Medical_Corpus\data\4_instruction\eval_CLEAN_KEEP_TRUE_REFUSALS.jsonl"),
    "gold_train": Path(r"D:\SL_Medical_Corpus\data\7_gold_standard\gold_train_DOC_DISJOINT.jsonl"),
    "gold_eval": Path(r"D:\SL_Medical_Corpus\data\7_gold_standard\gold_eval_DOC_DISJOINT.jsonl"),
    "train_plus_gold": Path(r"D:\SL_Medical_Corpus\data\4_instruction\regen_work\TRAIN_CLEAN_PLUS_GOLD.jsonl"),
}

MIN_CONTEXT_WORDS = 60

REFUSAL_PATTERNS = [
    r"\bi\s*don'?t\s*know\b",
    r"\bbased\s+on\s+the\s+provided\s+text\b",
    r"\bcannot\s+answer\b",
    r"\bcan'?t\s+answer\b",
    r"\bunable\s+to\s+answer\b",
    r"මට\s+ලබා\s+දී\s+ඇති",
    r"ලබා\s+දී\s+ඇති\s+පෙළ",
    r"நான்\s+வழங்கப்பட்ட",
    r"கொடுக்கப்பட்ட\s+உரை",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), flags=re.IGNORECASE)


# ----------------------------
# HELPERS
# ----------------------------
def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_context_block(user_text: str) -> str:
    m = re.search(r"(?is)\b(context|document|text|passage)\s*:\s*(.*)", user_text or "")
    return m.group(2).strip() if m else (user_text or "")


def get_primary_context(rec: dict) -> str:
    msgs = rec.get("messages", [])
    if len(msgs) >= 2 and msgs[1].get("role") == "user":
        return extract_context_block(msgs[1].get("content", "") or "")
    for m in msgs:
        if m.get("role") == "user":
            return extract_context_block(m.get("content", "") or "")
    return ""


def word_count(text: str) -> int:
    return len((text or "").split())


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text or ""))


def is_garbage_assistant(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if re.fullmatch(r"[\W_]+", t):
        return True
    return False


def pct(num: int, den: int) -> float:
    return round((num / den) * 100, 2) if den else 0.0


# ----------------------------
# CORE METRICS
# ----------------------------
def analyze_dataset(path: Path) -> dict:
    total_records = 0
    total_assistant_turns = 0
    refusal_turns = 0
    true_refusal_turns = 0
    suspicious_refusal_turns = 0
    records_with_any_refusal = 0
    records_with_suspicious_refusal = 0
    garbage_assistant_turns = 0

    lang_counter = Counter()
    style_counter = Counter()
    task_counter = Counter()
    lang_style_counter = Counter()
    doc_ids = set()

    context_word_counts = []

    for rec in read_jsonl(path):
        total_records += 1

        doc_id = rec.get("doc_id")
        if doc_id:
            doc_ids.add(str(doc_id))

        lang_counter[str(rec.get("lang", "unknown"))] += 1
        style_counter[str(rec.get("style", "unknown"))] += 1
        task_counter[str(rec.get("task", "unknown"))] += 1
        lang_style_counter[(str(rec.get("lang", "unknown")), str(rec.get("style", "unknown")))] += 1

        context = get_primary_context(rec)
        wc = word_count(context)
        context_word_counts.append(wc)

        rec_has_refusal = False
        rec_has_suspicious = False

        for m in rec.get("messages", []):
            if m.get("role") != "assistant":
                continue

            total_assistant_turns += 1
            txt = m.get("content", "") or ""

            if is_garbage_assistant(txt):
                garbage_assistant_turns += 1

            if is_refusal(txt):
                refusal_turns += 1
                rec_has_refusal = True

                if wc >= MIN_CONTEXT_WORDS:
                    suspicious_refusal_turns += 1
                    rec_has_suspicious = True
                else:
                    true_refusal_turns += 1

        if rec_has_refusal:
            records_with_any_refusal += 1
        if rec_has_suspicious:
            records_with_suspicious_refusal += 1

    avg_context_words = round(sum(context_word_counts) / len(context_word_counts), 2) if context_word_counts else 0.0

    return {
        "file": str(path),
        "records": total_records,
        "assistant_turns": total_assistant_turns,
        "doc_ids": len(doc_ids),
        "refusal_turns": refusal_turns,
        "true_refusal_turns": true_refusal_turns,
        "suspicious_refusal_turns": suspicious_refusal_turns,
        "records_with_any_refusal": records_with_any_refusal,
        "records_with_suspicious_refusal": records_with_suspicious_refusal,
        "garbage_assistant_turns": garbage_assistant_turns,
        "refusal_rate_turns_pct": pct(refusal_turns, total_assistant_turns),
        "true_refusal_rate_of_refusals_pct": pct(true_refusal_turns, refusal_turns),
        "suspicious_refusal_rate_of_refusals_pct": pct(suspicious_refusal_turns, refusal_turns),
        "affected_records_rate_pct": pct(records_with_any_refusal, total_records),
        "avg_context_words": avg_context_words,
        "lang_counter": dict(lang_counter),
        "style_counter": dict(style_counter),
        "task_counter": dict(task_counter),
        "lang_style_counter": {f"{k[0]}|{k[1]}": v for k, v in lang_style_counter.items()},
    }


# ----------------------------
# REPORT
# ----------------------------
def print_summary(name: str, stats: dict):
    print("=" * 80)
    print(name)
    print("=" * 80)
    print("file:", stats["file"])
    print("records:", stats["records"])
    print("assistant_turns:", stats["assistant_turns"])
    print("unique_doc_ids:", stats["doc_ids"])
    print("refusal_turns:", stats["refusal_turns"])
    print("true_refusal_turns:", stats["true_refusal_turns"])
    print("suspicious_refusal_turns:", stats["suspicious_refusal_turns"])
    print("records_with_any_refusal:", stats["records_with_any_refusal"])
    print("records_with_suspicious_refusal:", stats["records_with_suspicious_refusal"])
    print("garbage_assistant_turns:", stats["garbage_assistant_turns"])
    print("refusal_rate_turns_pct:", stats["refusal_rate_turns_pct"])
    print("true_refusal_rate_of_refusals_pct:", stats["true_refusal_rate_of_refusals_pct"])
    print("suspicious_refusal_rate_of_refusals_pct:", stats["suspicious_refusal_rate_of_refusals_pct"])
    print("affected_records_rate_pct:", stats["affected_records_rate_pct"])
    print("avg_context_words:", stats["avg_context_words"])

    print("\nlangs:")
    for k, v in sorted(stats["lang_counter"].items()):
        print(f"  {k}: {v}")

    print("\nstyles:")
    for k, v in sorted(stats["style_counter"].items()):
        print(f"  {k}: {v}")

    print("\ntasks:")
    for k, v in sorted(stats["task_counter"].items()):
        print(f"  {k}: {v}")

    print("\nlang|style:")
    for k, v in sorted(stats["lang_style_counter"].items()):
        print(f"  {k}: {v}")

    print()


def main():
    results = {}

    for name, path in FILES.items():
        if not path.exists():
            print(f"[WARN] Missing file for {name}: {path}")
            continue
        results[name] = analyze_dataset(path)

    for name, stats in results.items():
        print_summary(name, stats)

    # useful before/after comparisons
    if "train_original_final" in results and "train_clean" in results:
        orig = results["train_original_final"]
        clean = results["train_clean"]

        print("=" * 80)
        print("TRAIN BEFORE/AFTER COMPARISON")
        print("=" * 80)
        print("original_records:", orig["records"])
        print("clean_records:", clean["records"])
        print("retention_rate_pct:", pct(clean["records"], orig["records"]))
        print("original_refusal_turns:", orig["refusal_turns"])
        print("clean_refusal_turns:", clean["refusal_turns"])
        print("original_suspicious_refusal_turns:", orig["suspicious_refusal_turns"])
        print("clean_suspicious_refusal_turns:", clean["suspicious_refusal_turns"])
        print()

    if "eval_clean" in results:
        ev = results["eval_clean"]
        print("=" * 80)
        print("CLEAN EVAL SUMMARY")
        print("=" * 80)
        print("eval_records:", ev["records"])
        print("eval_refusal_turns:", ev["refusal_turns"])
        print("eval_true_refusal_turns:", ev["true_refusal_turns"])
        print("eval_suspicious_refusal_turns:", ev["suspicious_refusal_turns"])
        print()

    if "gold_train" in results and "gold_eval" in results:
        gt = results["gold_train"]
        ge = results["gold_eval"]

        print("=" * 80)
        print("GOLD SET SUMMARY")
        print("=" * 80)
        print("gold_train_records:", gt["records"])
        print("gold_eval_records:", ge["records"])
        print("gold_total_records:", gt["records"] + ge["records"])
        print("gold_train_doc_ids:", gt["doc_ids"])
        print("gold_eval_doc_ids:", ge["doc_ids"])
        print("gold_doc_overlap_expected:", 0)
        print()

    # save JSON report too
    out_json = Path(r"D:\SL_Medical_Corpus\data\7_gold_standard\dataset_metrics_report.json")
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved JSON report -> {out_json}")


if __name__ == "__main__":
    main()