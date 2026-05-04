import os
import math
import json
from collections import Counter
from statistics import mean
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
TABLE_NAME = "sl_med_translations"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_all_rows(table_name, page_size=1000):
    all_rows = []
    start = 0

    while True:
        end = start + page_size - 1
        resp = (
            supabase.table(table_name)
            .select(
                "id,doc_id,chunk_index,source_lang,source_text,"
                "translation_si,translation_ta,translation_en,"
                "status,error_message,translated_at,created_at"
            )
            .range(start, end)
            .execute()
        )

        rows = resp.data or []
        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < page_size:
            break

        start += page_size

    return all_rows


def simple_tokenize(text):
    if not text:
        return []
    return text.strip().split()


def normalize_text(text):
    if not text:
        return ""
    return " ".join(text.strip().split()).lower()


def shannon_entropy(tokens):
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def mattr(tokens, window_size=500):
    if not tokens:
        return 0.0
    if len(tokens) <= window_size:
        return len(set(tokens)) / len(tokens) if tokens else 0.0

    scores = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        scores.append(len(set(window)) / window_size)

    return sum(scores) / len(scores) if scores else 0.0


def safe_mean(values):
    return mean(values) if values else 0.0


def compute_text_metrics(texts):
    token_lists = [simple_tokenize(t) for t in texts]
    flat_tokens = [tok for toks in token_lists for tok in toks]
    lengths = [len(toks) for toks in token_lists]

    total_tokens = len(flat_tokens)
    vocab_size = len(set(flat_tokens))
    ttr = vocab_size / total_tokens if total_tokens else 0.0

    normalized = [normalize_text(t) for t in texts]
    duplicate_count = len(normalized) - len(set(normalized))
    duplicate_rate = duplicate_count / len(texts) if texts else 0.0

    return {
        "num_samples": len(texts),
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "avg_tokens_per_sample": total_tokens / len(texts) if texts else 0.0,
        "ttr": ttr,
        "mattr_500": mattr(flat_tokens, 500),
        "entropy": shannon_entropy(flat_tokens),
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "avg_len": safe_mean(lengths),
        "duplicate_count": duplicate_count,
        "duplicate_rate": duplicate_rate,
    }


def compute_length_ratios(source_texts, target_texts):
    ratios = []

    for src, tgt in zip(source_texts, target_texts):
        src_len = len(simple_tokenize(src))
        tgt_len = len(simple_tokenize(tgt))
        if src_len > 0:
            ratios.append(tgt_len / src_len)

    return {
        "avg_length_ratio": safe_mean(ratios),
        "min_length_ratio": min(ratios) if ratios else 0.0,
        "max_length_ratio": max(ratios) if ratios else 0.0,
    }


def compute_dataset_metrics(rows):
    total_rows = len(rows)

    source_lang_counts = Counter(r.get("source_lang") or "unknown" for r in rows)
    status_counts = Counter(r.get("status") or "unknown" for r in rows)

    source_texts = [r.get("source_text") or "" for r in rows if (r.get("source_text") or "").strip()]
    si_texts = [r.get("translation_si") or "" for r in rows if (r.get("translation_si") or "").strip()]
    ta_texts = [r.get("translation_ta") or "" for r in rows if (r.get("translation_ta") or "").strip()]
    en_texts = [r.get("translation_en") or "" for r in rows if (r.get("translation_en") or "").strip()]

    source_metrics = compute_text_metrics(source_texts)
    si_metrics = compute_text_metrics(si_texts)
    ta_metrics = compute_text_metrics(ta_texts)
    en_metrics = compute_text_metrics(en_texts)

    source_for_si = []
    target_si = []
    source_for_ta = []
    target_ta = []
    source_for_en = []
    target_en = []

    for r in rows:
        src = r.get("source_text") or ""

        si = r.get("translation_si") or ""
        ta = r.get("translation_ta") or ""
        en = r.get("translation_en") or ""

        if src.strip() and si.strip():
            source_for_si.append(src)
            target_si.append(si)

        if src.strip() and ta.strip():
            source_for_ta.append(src)
            target_ta.append(ta)

        if src.strip() and en.strip():
            source_for_en.append(src)
            target_en.append(en)

    results = {
        "total_rows": total_rows,
        "source_lang_counts": dict(source_lang_counts),
        "status_counts": dict(status_counts),

        "rows_with_translation_si": len(si_texts),
        "rows_with_translation_ta": len(ta_texts),
        "rows_with_translation_en": len(en_texts),

        "source_text_metrics": source_metrics,
        "translation_si_metrics": si_metrics,
        "translation_ta_metrics": ta_metrics,
        "translation_en_metrics": en_metrics,

        "translation_si_length_ratio_vs_source": compute_length_ratios(source_for_si, target_si),
        "translation_ta_length_ratio_vs_source": compute_length_ratios(source_for_ta, target_ta),
        "translation_en_length_ratio_vs_source": compute_length_ratios(source_for_en, target_en),
    }

    return results


if __name__ == "__main__":
    rows = fetch_all_rows(TABLE_NAME)
    metrics = compute_dataset_metrics(rows)

    print("=== TRANSLATION DATASET METRICS ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    with open("translation_dataset_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nSaved metrics to translation_dataset_metrics.json")