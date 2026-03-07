#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Research-grade dataset metrics for synthetic instruction datasets.

Computes:
- Dataset retention rate
- Noise reduction rate
- Over-Refusal Rate (ORR)
- Task entropy
- Linguistic Diversity Index (LDI)
- Type-Token Ratio (TTR)
- Duplicate response rate
- Self-BLEU (sample-based)
- Vendi score (embedding-based if sentence-transformers is available, else TF-IDF fallback)
- Mutual information between selected variables
- Context-answer similarity
- Extractive QA exact-match grounding
- Document disjointness / leakage between splits

Expected JSONL record format:
{
  "id": "...",
  "doc_id": "...",
  "task": "...",
  "language_primary": "...",   # optional
  "site": "...",               # optional
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "...Context: ..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
}

Usage:
python research_metrics.py ^
  --in_jsonl "D:\SL_Medical_Corpus\data\4_instruction\regen_work\TRAIN_CLEAN_PLUS_GOLD.jsonl" ^
  --out_dir "D:\SL_Medical_Corpus\eval_reports\research_metrics" ^
  --original_total_records 4887 ^
  --removed_records 6

Optional leakage check:
python research_metrics.py ^
  --in_jsonl "D:\...\TRAIN_CLEAN_PLUS_GOLD.jsonl" ^
  --out_dir "D:\...\eval_reports\research_metrics" ^
  --split_a "D:\...\train.jsonl" ^
  --split_b "D:\...\eval.jsonl"

Optional dependencies:
- pip install numpy pandas scikit-learn nltk
- pip install sentence-transformers   (optional, for stronger embeddings)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Optional: stronger semantic embeddings for Vendi/context-answer similarity
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


CRAWL_ERROR_KEYWORDS = [
    "Crawl4AI Error",
    "not fully supported",
    "All strings must be XML compatible",
    "XML compatible",
]

REFUSAL_PATTERNS = [
    r"\bi\s*don'?t\s*know\b",
    r"\bcannot\s+answer\b",
    r"\bcan'?t\s+answer\b",
    r"\bunable\s+to\s+answer\b",
    r"\bbased\s+on\s+the\s+provided\s+text\b",
    r"මට\s+ලබා\s+දී\s+ඇති",
    r"ලබා\s+දී\s+ඇති\s+පෙළ",
    r"நான்\s+வழங்கப்பட்ட",
    r"கொடுக்கப்பட்ட\s+உரை",
]

REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), flags=re.IGNORECASE)
WORD_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]


def word_count(text: str) -> int:
    return len((text or "").split())


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text or ""))


def record_all_text(rec: dict) -> str:
    return "\n".join((m.get("content") or "") for m in rec.get("messages", []))


def is_crawl_error_record(rec: dict) -> bool:
    full = record_all_text(rec)
    return any(k in full for k in CRAWL_ERROR_KEYWORDS)


def extract_context_block(user_text: str) -> str:
    m = re.search(r"(?is)\bcontext\s*:\s*(.*)", user_text or "")
    return (m.group(1).strip() if m else (user_text or ""))


def get_primary_context(rec: dict) -> str:
    msgs = rec.get("messages", [])
    if len(msgs) >= 2 and msgs[1].get("role") == "user":
        return extract_context_block(msgs[1].get("content", "") or "")
    for m in msgs:
        if m.get("role") == "user":
            return extract_context_block(m.get("content", "") or "")
    return ""


def get_assistant_messages(rec: dict) -> List[str]:
    return [m.get("content", "") or "" for m in rec.get("messages", []) if m.get("role") == "assistant"]


def get_last_assistant_message(rec: dict) -> str:
    msgs = get_assistant_messages(rec)
    return msgs[-1] if msgs else ""


def shannon_entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p, 2)
    return ent


def normalized_entropy_from_counts(counts: Counter) -> float:
    nonzero = [c for c in counts.values() if c > 0]
    k = len(nonzero)
    if k <= 1:
        return 0.0
    return shannon_entropy_from_counts(counts) / math.log(k, 2)


def mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std())


def bin_lengths(lengths: List[int], n_bins: int = 5) -> List[str]:
    if not lengths:
        return []
    arr = np.array(lengths)
    if len(set(arr.tolist())) == 1:
        return [f"bin_{0}"] * len(lengths)
    quantiles = np.quantile(arr, np.linspace(0, 1, n_bins + 1))
    labels = []
    for x in arr:
        idx = np.searchsorted(quantiles, x, side="right") - 1
        idx = min(max(idx, 0), n_bins - 1)
        labels.append(f"bin_{idx}")
    return labels


def exact_doc_leakage(split_a: Path, split_b: Path) -> dict:
    a = set()
    b = set()
    for rec in read_jsonl(split_a):
        if "doc_id" in rec:
            a.add(str(rec["doc_id"]))
    for rec in read_jsonl(split_b):
        if "doc_id" in rec:
            b.add(str(rec["doc_id"]))
    overlap = a & b
    leakage_rate = (len(overlap) / len(a)) if len(a) > 0 else None
    return {
        "split_a_doc_ids": len(a),
        "split_b_doc_ids": len(b),
        "shared_doc_ids": len(overlap),
        "leakage_rate_wrt_split_a": leakage_rate,
        "document_disjointness": float(len(overlap) == 0) if (len(a) + len(b) > 0) else None,
        "shared_examples": sorted(list(overlap))[:20],
    }


def compute_ttr(texts: List[str]) -> dict:
    toks = []
    for t in texts:
        toks.extend(tokenize(t))
    total_tokens = len(toks)
    vocab = set(toks)
    ttr = (len(vocab) / total_tokens) if total_tokens else 0.0
    return {
        "total_tokens": total_tokens,
        "vocab_size": len(vocab),
        "type_token_ratio": ttr,
    }


def compute_duplicate_rate(texts: List[str]) -> dict:
    normed = [" ".join((t or "").split()) for t in texts if (t or "").strip()]
    counts = Counter(normed)
    duplicate_groups = sum(1 for c in counts.values() if c > 1)
    duplicated_items = sum(c for c in counts.values() if c > 1)
    duplicate_rate = (duplicated_items / len(normed)) if normed else 0.0
    return {
        "n_texts": len(normed),
        "duplicate_groups": duplicate_groups,
        "duplicated_items": duplicated_items,
        "duplicate_rate": duplicate_rate,
    }


def compute_self_bleu(texts: List[str], sample_size: int = 300, ref_pool: int = 30, seed: int = 42) -> dict:
    if not NLTK_AVAILABLE:
        return {
            "self_bleu_mean": None,
            "note": "nltk not installed; run `pip install nltk`",
        }

    normed = [" ".join((t or "").split()) for t in texts if (t or "").strip()]
    if len(normed) < 2:
        return {"self_bleu_mean": 0.0, "n_samples_used": len(normed)}

    random.seed(seed)
    sample = normed if len(normed) <= sample_size else random.sample(normed, sample_size)
    smoothie = SmoothingFunction().method1

    scores = []
    for hyp in sample:
        refs = [r for r in normed if r != hyp]
        if not refs:
            continue
        if len(refs) > ref_pool:
            refs = random.sample(refs, ref_pool)

        hyp_tok = tokenize(hyp)
        ref_tok = [tokenize(r) for r in refs]
        if not hyp_tok or not ref_tok:
            continue

        score = sentence_bleu(
            ref_tok,
            hyp_tok,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie,
        )
        scores.append(score)

    mean_bleu, std_bleu = mean_std(scores)
    return {
        "self_bleu_mean": mean_bleu,
        "self_bleu_std": std_bleu,
        "n_samples_used": len(scores),
        "reference_pool_per_sample": ref_pool,
    }


def get_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> Tuple[np.ndarray, str]:
    if ST_AVAILABLE:
        model = SentenceTransformer(model_name)
        emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        return np.asarray(emb), f"sentence-transformers:{model_name}"

    # Fallback: TF-IDF vectors
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    mat = vec.fit_transform(texts)
    dense = mat.toarray()
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dense = dense / norms
    return dense, "tfidf-fallback"


def compute_vendi_score(texts: List[str], max_samples: int = 500, seed: int = 42) -> dict:
    """
    Embedding-based Vendi-style score from a similarity matrix.
    Returns:
    - vendi_score_effective_rank
    - normalized_vendi_score in [0,1]
    """
    normed = [" ".join((t or "").split()) for t in texts if (t or "").strip()]
    if not normed:
        return {"vendi_score_effective_rank": 0.0, "normalized_vendi_score": 0.0, "embedding_backend": None}

    random.seed(seed)
    if len(normed) > max_samples:
        normed = random.sample(normed, max_samples)

    emb, backend = get_embeddings(normed)
    sim = emb @ emb.T

    # Symmetrize and stabilize
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 1.0)

    # Eigenvalues of PSD-like similarity matrix
    vals = np.linalg.eigvalsh(sim)
    vals = np.clip(vals, 1e-12, None)
    probs = vals / vals.sum()

    entropy = -np.sum(probs * np.log(probs))
    effective_rank = float(np.exp(entropy))
    normalized = float(effective_rank / len(normed))

    return {
        "vendi_score_effective_rank": effective_rank,
        "normalized_vendi_score": normalized,
        "n_samples_used": len(normed),
        "embedding_backend": backend,
    }


def compute_context_answer_similarity(contexts: List[str], answers: List[str], max_samples: int = 1000, seed: int = 42) -> dict:
    pairs = [(c, a) for c, a in zip(contexts, answers) if (c or "").strip() and (a or "").strip()]
    if not pairs:
        return {"mean_similarity": 0.0, "std_similarity": 0.0, "backend": None}

    random.seed(seed)
    if len(pairs) > max_samples:
        pairs = random.sample(pairs, max_samples)

    ctxs = [p[0] for p in pairs]
    ans = [p[1] for p in pairs]

    if ST_AVAILABLE:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        ctx_emb = model.encode(ctxs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        ans_emb = model.encode(ans, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        sims = np.sum(np.asarray(ctx_emb) * np.asarray(ans_emb), axis=1)
        backend = "sentence-transformers:all-MiniLM-L6-v2"
    else:
        vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
        mat = vec.fit_transform(ctxs + ans)
        ctx_mat = mat[: len(ctxs)]
        ans_mat = mat[len(ctxs) :]
        sims = np.array([
            cosine_similarity(ctx_mat[i], ans_mat[i])[0, 0]
            for i in range(len(ctxs))
        ])
        backend = "tfidf-fallback"

    mean_sim, std_sim = mean_std(sims.tolist())
    return {
        "mean_similarity": mean_sim,
        "std_similarity": std_sim,
        "n_pairs_used": len(pairs),
        "backend": backend,
    }


def compute_extractive_grounding(records: List[dict]) -> dict:
    total = 0
    exact_match = 0
    for rec in records:
        task = str(rec.get("task", ""))
        if task != "extractive_qa":
            continue
        context = get_primary_context(rec)
        answer = get_last_assistant_message(rec)
        if not context.strip() or not answer.strip():
            continue
        total += 1
        if answer.strip() in context:
            exact_match += 1
    accuracy = (exact_match / total) if total else None
    return {
        "n_extractive_records_evaluated": total,
        "exact_substring_match_count": exact_match,
        "extractive_grounding_accuracy": accuracy,
    }


def compute_mutual_information(records: List[dict], language_field: str = "language_primary") -> dict:
    tasks = []
    langs = []
    resp_lens = []

    for rec in records:
        task = str(rec.get("task", "unknown"))
        lang = str(rec.get(language_field, "unknown"))
        ans = get_last_assistant_message(rec)
        rlen = word_count(ans)

        tasks.append(task)
        langs.append(lang)
        resp_lens.append(rlen)

    if not tasks:
        return {}

    len_bins = bin_lengths(resp_lens, n_bins=5)

    task_lang_mi = mutual_info_score(tasks, langs) if tasks and langs else None
    task_resp_len_mi = mutual_info_score(tasks, len_bins) if tasks and len_bins else None
    lang_resp_len_mi = mutual_info_score(langs, len_bins) if langs and len_bins else None

    return {
        "mutual_information_task_language": float(task_lang_mi) if task_lang_mi is not None else None,
        "mutual_information_task_response_length_bin": float(task_resp_len_mi) if task_resp_len_mi is not None else None,
        "mutual_information_language_response_length_bin": float(lang_resp_len_mi) if lang_resp_len_mi is not None else None,
        "response_length_bins": 5,
    }


def summarize_records(
    records: List[dict],
    min_context_words: int,
    original_total_records: Optional[int],
    removed_records: Optional[int],
    language_field: str,
) -> dict:
    total_records = len(records)

    crawl_error_records = 0
    task_counts = Counter()
    lang_counts = Counter()
    source_counts = Counter()

    total_refusal_records = 0
    answerable_records = 0
    over_refusal_records = 0

    assistant_texts = []
    assistant_lengths = []
    contexts = []
    answers = []

    for rec in records:
        if is_crawl_error_record(rec):
            crawl_error_records += 1

        task = str(rec.get("task", "unknown"))
        task_counts[task] += 1

        lang = rec.get(language_field)
        if lang:
            lang_counts[str(lang)] += 1

        source = rec.get("site") or rec.get("source") or rec.get("domain")
        if source:
            source_counts[str(source)] += 1

        context = get_primary_context(rec)
        asst_msgs = get_assistant_messages(rec)
        has_refusal = any(is_refusal(a) for a in asst_msgs)

        if has_refusal:
            total_refusal_records += 1

        answerable = (word_count(context) >= min_context_words)
        if answerable:
            answerable_records += 1
            if has_refusal:
                over_refusal_records += 1

        for a in asst_msgs:
            if a.strip():
                assistant_texts.append(a)
                assistant_lengths.append(word_count(a))

        last_ans = get_last_assistant_message(rec)
        if context.strip() and last_ans.strip():
            contexts.append(context)
            answers.append(last_ans)

    retention_rate = None
    noise_reduction_rate = None
    compression_ratio = None
    if original_total_records:
        retention_rate = total_records / original_total_records
        compression_ratio = total_records / original_total_records
    if original_total_records and removed_records is not None:
        noise_reduction_rate = removed_records / original_total_records

    orr = (over_refusal_records / answerable_records) if answerable_records else None
    avg_resp_len, std_resp_len = mean_std(assistant_lengths)

    metrics = {
        "dataset_cleaning": {
            "original_total_records": original_total_records,
            "final_total_records": total_records,
            "removed_records": removed_records,
            "retention_rate": retention_rate,
            "compression_ratio": compression_ratio,
            "noise_reduction_rate": noise_reduction_rate,
            "crawl_error_records_still_present": crawl_error_records,
        },
        "refusal_metrics": {
            "total_refusal_containing_records": total_refusal_records,
            "answerable_records_heuristic": answerable_records,
            "over_refusal_records": over_refusal_records,
            "over_refusal_rate": orr,
        },
        "distribution_metrics": {
            "task_entropy_normalized": normalized_entropy_from_counts(task_counts),
            "linguistic_diversity_index": normalized_entropy_from_counts(lang_counts) if lang_counts else None,
            "source_entropy_bits": shannon_entropy_from_counts(source_counts) if source_counts else None,
            "avg_assistant_response_length_words": avg_resp_len,
            "std_assistant_response_length_words": std_resp_len,
        },
        "task_distribution": dict(task_counts),
        "language_distribution": dict(lang_counts),
    }

    metrics["ttr"] = compute_ttr(assistant_texts)
    metrics["duplicates"] = compute_duplicate_rate(assistant_texts)
    metrics["self_bleu"] = compute_self_bleu(assistant_texts)
    metrics["vendi"] = compute_vendi_score(assistant_texts)
    metrics["context_answer_similarity"] = compute_context_answer_similarity(contexts, answers)
    metrics["extractive_grounding"] = compute_extractive_grounding(records)
    metrics["mutual_information"] = compute_mutual_information(records, language_field=language_field)

    return metrics


def build_markdown_report(metrics: dict, leakage: Optional[dict]) -> str:
    dc = metrics["dataset_cleaning"]
    rf = metrics["refusal_metrics"]
    dm = metrics["distribution_metrics"]
    ttr = metrics["ttr"]
    dup = metrics["duplicates"]
    sb = metrics["self_bleu"]
    vendi = metrics["vendi"]
    cas = metrics["context_answer_similarity"]
    exg = metrics["extractive_grounding"]
    mi = metrics["mutual_information"]

    lines = []
    lines.append("# Research Dataset Metrics\n")

    lines.append("## 1. Dataset Cleaning Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k, v in [
        ("Original total records", dc.get("original_total_records")),
        ("Final total records", dc.get("final_total_records")),
        ("Removed records", dc.get("removed_records")),
        ("Retention rate", dc.get("retention_rate")),
        ("Compression ratio", dc.get("compression_ratio")),
        ("Noise reduction rate", dc.get("noise_reduction_rate")),
        ("Crawl-error records still present", dc.get("crawl_error_records_still_present")),
    ]:
        lines.append(f"| {k} | {format_value(v)} |")
    lines.append("")

    lines.append("## 2. Refusal Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k, v in [
        ("Total refusal-containing records", rf.get("total_refusal_containing_records")),
        ("Answerable records (heuristic)", rf.get("answerable_records_heuristic")),
        ("Over-refusal records", rf.get("over_refusal_records")),
        ("Over-Refusal Rate (ORR)", rf.get("over_refusal_rate")),
    ]:
        lines.append(f"| {k} | {format_value(v)} |")
    lines.append("")

    lines.append("## 3. Diversity Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k, v in [
        ("Normalized task entropy", dm.get("task_entropy_normalized")),
        ("Linguistic Diversity Index (LDI)", dm.get("linguistic_diversity_index")),
        ("Source entropy (bits)", dm.get("source_entropy_bits")),
        ("Average assistant response length (words)", dm.get("avg_assistant_response_length_words")),
        ("Std assistant response length (words)", dm.get("std_assistant_response_length_words")),
        ("Vocabulary size", ttr.get("vocab_size")),
        ("Total tokens", ttr.get("total_tokens")),
        ("Type-Token Ratio (TTR)", ttr.get("type_token_ratio")),
        ("Vendi score (effective rank)", vendi.get("vendi_score_effective_rank")),
        ("Normalized Vendi score", vendi.get("normalized_vendi_score")),
        ("Vendi backend", vendi.get("embedding_backend")),
        ("Vendi sample size", vendi.get("n_samples_used")),
    ]:
        lines.append(f"| {k} | {format_value(v)} |")
    lines.append("")

    lines.append("## 4. Redundancy Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k, v in [
        ("Duplicate response groups", dup.get("duplicate_groups")),
        ("Duplicated response items", dup.get("duplicated_items")),
        ("Duplicate response rate", dup.get("duplicate_rate")),
        ("Self-BLEU mean", sb.get("self_bleu_mean")),
        ("Self-BLEU std", sb.get("self_bleu_std")),
        ("Self-BLEU samples used", sb.get("n_samples_used")),
    ]:
        lines.append(f"| {k} | {format_value(v)} |")
    lines.append("")

    lines.append("## 5. Grounding Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k, v in [
        ("Mean context-answer similarity", cas.get("mean_similarity")),
        ("Std context-answer similarity", cas.get("std_similarity")),
        ("Context-answer similarity backend", cas.get("backend")),
        ("Context-answer pairs used", cas.get("n_pairs_used")),
        ("Extractive QA records evaluated", exg.get("n_extractive_records_evaluated")),
        ("Extractive exact substring match count", exg.get("exact_substring_match_count")),
        ("Extractive grounding accuracy", exg.get("extractive_grounding_accuracy")),
    ]:
        lines.append(f"| {k} | {format_value(v)} |")
    lines.append("")

    lines.append("## 6. Dependency Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k, v in [
        ("Mutual information: task-language", mi.get("mutual_information_task_language")),
        ("Mutual information: task-response length bin", mi.get("mutual_information_task_response_length_bin")),
        ("Mutual information: language-response length bin", mi.get("mutual_information_language_response_length_bin")),
    ]:
        lines.append(f"| {k} | {format_value(v)} |")
    lines.append("")

    lines.append("## 7. Task Distribution\n")
    lines.append("| Task | Count |")
    lines.append("|---|---:|")
    for task, count in sorted(metrics["task_distribution"].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {task} | {count} |")
    lines.append("")

    if metrics["language_distribution"]:
        lines.append("## 8. Language Distribution\n")
        lines.append("| Language | Count |")
        lines.append("|---|---:|")
        for lang, count in sorted(metrics["language_distribution"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {lang} | {count} |")
        lines.append("")

    if leakage:
        lines.append("## 9. Benchmark Integrity / Leakage\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for k, v in [
            ("Split A doc_ids", leakage.get("split_a_doc_ids")),
            ("Split B doc_ids", leakage.get("split_b_doc_ids")),
            ("Shared doc_ids", leakage.get("shared_doc_ids")),
            ("Leakage rate wrt split A", leakage.get("leakage_rate_wrt_split_a")),
            ("Document disjointness", leakage.get("document_disjointness")),
        ]:
            lines.append(f"| {k} | {format_value(v)} |")
        if leakage.get("shared_examples"):
            lines.append("")
            lines.append("Shared doc_id examples:")
            for x in leakage["shared_examples"]:
                lines.append(f"- `{x}`")
        lines.append("")

    return "\n".join(lines)


def format_value(v) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="Main dataset JSONL")
    ap.add_argument("--out_dir", required=True, help="Directory to save outputs")
    ap.add_argument("--min_context_words", type=int, default=60)
    ap.add_argument("--original_total_records", type=int, default=None)
    ap.add_argument("--removed_records", type=int, default=None)
    ap.add_argument("--language_field", type=str, default="language_primary")
    ap.add_argument("--split_a", type=str, default=None, help="Optional split A JSONL for doc leakage check")
    ap.add_argument("--split_b", type=str, default=None, help="Optional split B JSONL for doc leakage check")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = list(read_jsonl(in_path))
    metrics = summarize_records(
        records=records,
        min_context_words=args.min_context_words,
        original_total_records=args.original_total_records,
        removed_records=args.removed_records,
        language_field=args.language_field,
    )

    leakage = None
    if args.split_a and args.split_b:
        leakage = exact_doc_leakage(Path(args.split_a), Path(args.split_b))

    write_json(out_dir / "metrics.json", metrics)
    write_markdown(out_dir / "metrics.md", build_markdown_report(metrics, leakage))

    print("Wrote:")
    print(" -", out_dir / "metrics.json")
    print(" -", out_dir / "metrics.md")
    if leakage:
        print(" - leakage included in report")


if __name__ == "__main__":
    main()