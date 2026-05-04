import os
import re
import json
import math
import random
import argparse
import statistics
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Optional imports
HAS_TEXTSTAT = False
HAS_SENTENCE_TRANSFORMERS = False
HAS_BERTSCORE = False
HAS_MAUVE = False
HAS_BLEURT = False
HAS_NLTK = False

try:
    import textstat
    HAS_TEXTSTAT = True
except Exception:
    pass

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    pass

try:
    from bert_score import score as bertscore_score
    HAS_BERTSCORE = True
except Exception:
    pass

try:
    import mauve
    HAS_MAUVE = True
except Exception:
    pass

try:
    from bleurt import score as bleurt_score
    HAS_BLEURT = True
except Exception:
    pass

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    HAS_NLTK = True
except Exception:
    pass


# ============================================================
# HELPERS
# ============================================================

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", str(text).lower())


def sentence_split(text: str) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def safe_mean(values: List[float]) -> Optional[float]:
    return round(float(sum(values) / len(values)), 6) if values else None


def safe_std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None
    return round(float(statistics.pstdev(values)), 6)


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    return round(float(np.percentile(values, p)), 6)


def entropy_from_counter(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = [v / total for v in counter.values()]
    return round(float(-sum(p * math.log(p, 2) for p in probs if p > 0)), 6)


def type_token_ratio(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 6)


def mattr(tokens: List[str], window: int = 500) -> float:
    if not tokens:
        return 0.0
    if len(tokens) < window:
        return round(len(set(tokens)) / max(len(tokens), 1), 6)

    scores = []
    for i in range(len(tokens) - window + 1):
        window_tokens = tokens[i:i + window]
        scores.append(len(set(window_tokens)) / window)

    return round(float(sum(scores) / len(scores)), 6)


def distinct_n(texts: List[str], n: int = 1) -> float:
    total = 0
    unique = set()

    for text in texts:
        toks = tokenize(text)
        if len(toks) < n:
            continue
        grams = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        unique.update(grams)

    if total == 0:
        return 0.0
    return round(len(unique) / total, 6)


def pairwise_cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    x_norm = x / norms
    sim = np.matmul(x_norm, x_norm.T)
    return sim


def vendi_score_from_similarity(sim: np.ndarray) -> Dict[str, float]:
    """
    Vendi score based on eigenvalue entropy.
    Returns raw effective rank and normalized score.
    """
    if sim.size == 0:
        return {"vendi_score": 0.0, "normalized_vendi_score": 0.0}

    n = sim.shape[0]
    sim = (sim + sim.T) / 2.0
    eigvals = np.linalg.eigvalsh(sim)

    eigvals = np.clip(eigvals, a_min=0, a_max=None)
    if eigvals.sum() == 0:
        return {"vendi_score": 0.0, "normalized_vendi_score": 0.0}

    p = eigvals / eigvals.sum()
    entropy = -np.sum([pi * np.log(pi) for pi in p if pi > 0])
    vendi = float(np.exp(entropy))
    normalized = float(vendi / n)

    return {
        "vendi_score": round(vendi, 6),
        "normalized_vendi_score": round(normalized, 6)
    }


def average_pairwise_cosine_distance(embeddings: np.ndarray) -> float:
    if len(embeddings) < 2:
        return 0.0
    sim = pairwise_cosine_similarity_matrix(embeddings)
    n = sim.shape[0]
    triu = sim[np.triu_indices(n, k=1)]
    distances = 1.0 - triu
    return round(float(np.mean(distances)), 6)


def centroid_dispersion(embeddings: np.ndarray) -> float:
    if len(embeddings) == 0:
        return 0.0
    centroid = embeddings.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(embeddings, axis=1)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        return 0.0
    sims = np.dot(embeddings, centroid.T).reshape(-1) / ((norms * centroid_norm) + 1e-12)
    distances = 1.0 - sims
    return round(float(np.mean(distances)), 6)


def estimate_reasoning_markers(text: str) -> Dict[str, int]:
    text_low = text.lower()
    markers = {
        "conditional_markers": len(re.findall(r"\b(if|when|unless|provided|in case|depending on)\b", text_low)),
        "enumeration_markers": len(re.findall(r"\b(first|second|third|next|finally|step)\b", text_low)),
        "contrast_markers": len(re.findall(r"\b(however|but|although|while|whereas)\b", text_low)),
        "causal_markers": len(re.findall(r"\b(because|therefore|so that|thus|hence)\b", text_low)),
    }
    return markers


def compute_readability(text: str) -> Dict[str, Optional[float]]:
    if not HAS_TEXTSTAT or not normalize_ws(text):
        return {
            "flesch_reading_ease": None,
            "flesch_kincaid_grade": None,
            "gunning_fog": None,
            "smog_index": None
        }

    try:
        return {
            "flesch_reading_ease": round(float(textstat.flesch_reading_ease(text)), 6),
            "flesch_kincaid_grade": round(float(textstat.flesch_kincaid_grade(text)), 6),
            "gunning_fog": round(float(textstat.gunning_fog(text)), 6),
            "smog_index": round(float(textstat.smog_index(text)), 6),
        }
    except Exception:
        return {
            "flesch_reading_ease": None,
            "flesch_kincaid_grade": None,
            "gunning_fog": None,
            "smog_index": None
        }


def avg_sentence_length(text: str) -> float:
    sents = sentence_split(text)
    if not sents:
        return 0.0
    lengths = [len(tokenize(s)) for s in sents]
    return round(float(sum(lengths) / len(lengths)), 6)


def build_doc_language_lookup(source_csv: Optional[str], id_column: str, language_column: str) -> Dict[str, str]:
    if not source_csv or not os.path.exists(source_csv):
        return {}
    df = pd.read_csv(source_csv)
    if id_column not in df.columns or language_column not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        out[str(row[id_column])] = str(row[language_column]) if pd.notna(row[language_column]) else "unknown"
    return out


def get_example_question_answer(item: Dict[str, Any]) -> Tuple[str, str]:
    conv = item.get("conversations", [])
    question = ""
    answer = ""
    if conv:
        question = conv[0].get("value", "")
        answer = conv[-1].get("value", "")
    return question, answer


def flatten_texts(dataset: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    all_turns = []
    questions = []
    answers = []

    for item in dataset:
        q, a = get_example_question_answer(item)
        if q:
            questions.append(q)
        if a:
            answers.append(a)

        for turn in item.get("conversations", []):
            val = turn.get("value", "")
            if val:
                all_turns.append(val)

    return {
        "all_turns": all_turns,
        "questions": questions,
        "answers": answers
    }


def summarize_lengths(lengths: List[int]) -> Dict[str, Optional[float]]:
    return {
        "mean": safe_mean(lengths),
        "std": safe_std(lengths),
        "min": min(lengths) if lengths else None,
        "p25": percentile(lengths, 25),
        "median": percentile(lengths, 50),
        "p75": percentile(lengths, 75),
        "max": max(lengths) if lengths else None,
    }


# ============================================================
# CORE METRICS
# ============================================================

def compute_structure_metrics(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    example_type_counts = Counter()
    turn_counts = []
    question_token_lengths = []
    answer_token_lengths = []
    all_example_token_lengths = []
    answer_sentence_counts = []
    answer_avg_sentence_lengths = []
    reasoning_marker_totals = Counter()

    for item in dataset:
        example_type_counts[item.get("example_type", "unknown")] += 1

        conv = item.get("conversations", [])
        turn_counts.append(len(conv))

        full_example_text = " ".join(turn.get("value", "") for turn in conv)
        all_example_token_lengths.append(len(tokenize(full_example_text)))

        q, a = get_example_question_answer(item)
        question_token_lengths.append(len(tokenize(q)))
        answer_token_lengths.append(len(tokenize(a)))

        a_sents = sentence_split(a)
        answer_sentence_counts.append(len(a_sents))
        answer_avg_sentence_lengths.append(avg_sentence_length(a))

        rm = estimate_reasoning_markers(a)
        reasoning_marker_totals.update(rm)

    total_examples = len(dataset)
    refusal_count = example_type_counts.get("single_qa_refusal", 0)

    return {
        "num_examples": total_examples,
        "example_type_distribution": dict(example_type_counts),
        "refusal_ratio": round(refusal_count / total_examples, 6) if total_examples else 0.0,
        "turns_per_example": summarize_lengths(turn_counts),
        "question_token_lengths": summarize_lengths(question_token_lengths),
        "answer_token_lengths": summarize_lengths(answer_token_lengths),
        "example_token_lengths": summarize_lengths(all_example_token_lengths),
        "answer_sentence_counts": summarize_lengths(answer_sentence_counts),
        "answer_avg_sentence_length": summarize_lengths(answer_avg_sentence_lengths),
        "reasoning_marker_averages_per_example": {
            k: round(v / total_examples, 6) if total_examples else 0.0
            for k, v in reasoning_marker_totals.items()
        }
    }


def compute_lexical_metrics(dataset: List[Dict[str, Any]], mattr_window: int = 500) -> Dict[str, Any]:
    texts = flatten_texts(dataset)

    all_tokens = []
    question_tokens = []
    answer_tokens = []

    for t in texts["all_turns"]:
        all_tokens.extend(tokenize(t))
    for t in texts["questions"]:
        question_tokens.extend(tokenize(t))
    for t in texts["answers"]:
        answer_tokens.extend(tokenize(t))

    question_counter = Counter(question_tokens)
    answer_counter = Counter(answer_tokens)
    all_counter = Counter(all_tokens)

    return {
        "all_text": {
            "total_tokens": len(all_tokens),
            "vocab_size": len(set(all_tokens)),
            "ttr": type_token_ratio(all_tokens),
            "mattr": mattr(all_tokens, window=mattr_window),
            "entropy": entropy_from_counter(all_counter),
            "distinct_1": distinct_n(texts["all_turns"], 1),
            "distinct_2": distinct_n(texts["all_turns"], 2),
        },
        "questions": {
            "total_tokens": len(question_tokens),
            "vocab_size": len(set(question_tokens)),
            "ttr": type_token_ratio(question_tokens),
            "mattr": mattr(question_tokens, window=mattr_window),
            "entropy": entropy_from_counter(question_counter),
            "distinct_1": distinct_n(texts["questions"], 1),
            "distinct_2": distinct_n(texts["questions"], 2),
        },
        "answers": {
            "total_tokens": len(answer_tokens),
            "vocab_size": len(set(answer_tokens)),
            "ttr": type_token_ratio(answer_tokens),
            "mattr": mattr(answer_tokens, window=mattr_window),
            "entropy": entropy_from_counter(answer_counter),
            "distinct_1": distinct_n(texts["answers"], 1),
            "distinct_2": distinct_n(texts["answers"], 2),
        }
    }


def compute_readability_metrics(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    answer_texts = [get_example_question_answer(item)[1] for item in dataset if get_example_question_answer(item)[1]]

    readability_rows = [compute_readability(a) for a in answer_texts]
    metrics = defaultdict(list)

    for row in readability_rows:
        for k, v in row.items():
            if v is not None:
                metrics[k].append(v)

    return {
        k: {
            "mean": safe_mean(v),
            "std": safe_std(v),
            "min": min(v) if v else None,
            "max": max(v) if v else None,
        }
        for k, v in metrics.items()
    }


def compute_self_bleu(texts: List[str], sample_size: int = 300, seed: int = 42) -> Dict[str, Any]:
    if not HAS_NLTK:
        return {"available": False, "reason": "nltk not installed"}

    texts = [normalize_ws(t) for t in texts if normalize_ws(t)]
    if len(texts) < 2:
        return {"available": True, "self_bleu": None, "sample_size_used": len(texts)}

    rng = random.Random(seed)
    if len(texts) > sample_size:
        texts = rng.sample(texts, sample_size)

    smoothie = SmoothingFunction().method1
    bleu_scores = []

    tokenized = [tokenize(t) for t in texts]
    for i, hyp in enumerate(tokenized):
        refs = tokenized[:i] + tokenized[i+1:]
        if not refs or not hyp:
            continue
        try:
            score = sentence_bleu(refs, hyp, smoothing_function=smoothie)
            bleu_scores.append(score)
        except Exception:
            continue

    return {
        "available": True,
        "self_bleu": round(float(np.mean(bleu_scores)), 6) if bleu_scores else None,
        "sample_size_used": len(texts)
    }


def compute_embedding_metrics(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    sample_size: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    if not HAS_SENTENCE_TRANSFORMERS:
        return {"available": False, "reason": "sentence-transformers not installed"}

    cleaned = [normalize_ws(t) for t in texts if normalize_ws(t)]
    if not cleaned:
        return {"available": True, "num_texts_used": 0}

    rng = random.Random(seed)
    if len(cleaned) > sample_size:
        cleaned = rng.sample(cleaned, sample_size)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(cleaned, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)

    sim = pairwise_cosine_similarity_matrix(embeddings)
    vendi = vendi_score_from_similarity(sim)

    return {
        "available": True,
        "embedding_model": model_name,
        "num_texts_used": len(cleaned),
        "average_pairwise_cosine_distance": average_pairwise_cosine_distance(embeddings),
        "centroid_dispersion": centroid_dispersion(embeddings),
        **vendi
    }


def compute_judge_metrics(judged_rows_path: Optional[str]) -> Dict[str, Any]:
    if not judged_rows_path or not os.path.exists(judged_rows_path):
        return {"available": False, "reason": "judged_rows_merged.json not provided"}

    with open(judged_rows_path, "r", encoding="utf-8") as f:
        judged_rows = json.load(f)

    groundedness_counts = Counter()
    question_quality_counts = Counter()
    answer_quality_counts = Counter()
    language_handling_counts = Counter()
    hallucination_counts = Counter()
    nonsense_counts = Counter()
    should_refuse_counts = Counter()

    grounded_map = {"fully_grounded": 2, "partially_grounded": 1, "not_grounded": 0}
    quality_map = {"good": 2, "borderline": 1, "poor": 0}
    lang_map = {"good": 2, "uncertain": 1, "poor": 0}

    def avg_score(field: str, mapping: Dict[str, int]) -> Optional[float]:
        vals = []
        for row in judged_rows:
            val = row.get("judgment", {}).get(field)
            if val in mapping:
                vals.append(mapping[val])
        return round(float(np.mean(vals)), 6) if vals else None

    per_language = defaultdict(list)
    per_type = defaultdict(list)

    for row in judged_rows:
        j = row.get("judgment", {})
        lang = row.get("doc_language", "unknown")
        ex_type = row.get("example_type", "unknown")

        groundedness_counts[j.get("groundedness", "unknown")] += 1
        question_quality_counts[j.get("question_quality", "unknown")] += 1
        answer_quality_counts[j.get("answer_quality", "unknown")] += 1
        language_handling_counts[j.get("language_handling", "unknown")] += 1
        hallucination_counts[str(j.get("hallucination", None))] += 1
        nonsense_counts[str(j.get("nonsense_or_unrealistic", None))] += 1
        should_refuse_counts[str(j.get("should_refuse", None))] += 1

        per_language[lang].append(row)
        per_type[ex_type].append(row)

    total = len(judged_rows)
    out = {
        "available": True,
        "num_judged_examples": total,
        "groundedness_counts": dict(groundedness_counts),
        "question_quality_counts": dict(question_quality_counts),
        "answer_quality_counts": dict(answer_quality_counts),
        "language_handling_counts": dict(language_handling_counts),
        "hallucination_counts": dict(hallucination_counts),
        "nonsense_counts": dict(nonsense_counts),
        "should_refuse_counts": dict(should_refuse_counts),
        "rates": {
            "fully_grounded_rate": round(groundedness_counts["fully_grounded"] / total, 6) if total else None,
            "partially_grounded_rate": round(groundedness_counts["partially_grounded"] / total, 6) if total else None,
            "not_grounded_rate": round(groundedness_counts["not_grounded"] / total, 6) if total else None,
            "hallucination_rate": round(hallucination_counts["True"] / total, 6) if total else None,
            "nonsense_rate": round(nonsense_counts["True"] / total, 6) if total else None,
            "should_refuse_rate": round(should_refuse_counts["True"] / total, 6) if total else None,
        },
        "scores": {
            "groundedness_score": avg_score("groundedness", grounded_map),
            "question_quality_score": avg_score("question_quality", quality_map),
            "answer_quality_score": avg_score("answer_quality", quality_map),
            "language_handling_score": avg_score("language_handling", lang_map),
        },
        "per_language": {},
        "per_example_type": {}
    }

    for lang, rows in per_language.items():
        n = len(rows)
        out["per_language"][lang] = {
            "count": n,
            "fully_grounded_rate": round(sum(r["judgment"]["groundedness"] == "fully_grounded" for r in rows) / n, 6) if n else None,
            "hallucination_rate": round(sum(bool(r["judgment"]["hallucination"]) for r in rows) / n, 6) if n else None,
            "should_refuse_rate": round(sum(bool(r["judgment"]["should_refuse"]) for r in rows) / n, 6) if n else None,
            "nonsense_rate": round(sum(bool(r["judgment"]["nonsense_or_unrealistic"]) for r in rows) / n, 6) if n else None,
        }

    for ex_type, rows in per_type.items():
        n = len(rows)
        out["per_example_type"][ex_type] = {
            "count": n,
            "fully_grounded_rate": round(sum(r["judgment"]["groundedness"] == "fully_grounded" for r in rows) / n, 6) if n else None,
            "hallucination_rate": round(sum(bool(r["judgment"]["hallucination"]) for r in rows) / n, 6) if n else None,
            "should_refuse_rate": round(sum(bool(r["judgment"]["should_refuse"]) for r in rows) / n, 6) if n else None,
            "nonsense_rate": round(sum(bool(r["judgment"]["nonsense_or_unrealistic"]) for r in rows) / n, 6) if n else None,
        }

    return out


# ============================================================
# OPTIONAL REFERENCE-BASED METRICS
# ============================================================

def load_reference_file(reference_path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not reference_path or not os.path.exists(reference_path):
        return None

    if reference_path.lower().endswith(".json"):
        with open(reference_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    if reference_path.lower().endswith(".csv"):
        df = pd.read_csv(reference_path)
        return df.to_dict(orient="records")

    return None


def build_prediction_reference_pairs(
    dataset: List[Dict[str, Any]],
    reference_rows: Optional[List[Dict[str, Any]]],
    reference_answer_column: str = "reference_answer"
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Expected reference file to contain either:
    - aligned rows in the same order, OR
    - source_record_id + reference_answer
    """
    if not reference_rows:
        return [], [], {"available": False, "reason": "No reference file supplied"}

    preds = [get_example_question_answer(item)[1] for item in dataset]

    refs = []
    if "source_record_id" in reference_rows[0]:
        ref_map = {str(r["source_record_id"]): str(r.get(reference_answer_column, "")) for r in reference_rows}
        for item in dataset:
            refs.append(ref_map.get(str(item.get("source_record_id")), ""))
    else:
        refs = [str(r.get(reference_answer_column, "")) for r in reference_rows[:len(dataset)]]

    paired_preds = []
    paired_refs = []
    for p, r in zip(preds, refs):
        p = normalize_ws(p)
        r = normalize_ws(r)
        if p and r:
            paired_preds.append(p)
            paired_refs.append(r)

    return paired_preds, paired_refs, {
        "available": True,
        "num_pairs": len(paired_preds)
    }


def compute_bertscore(preds: List[str], refs: List[str], lang: str = "en") -> Dict[str, Any]:
    if not HAS_BERTSCORE:
        return {"available": False, "reason": "bert-score not installed"}
    if not preds or not refs:
        return {"available": False, "reason": "No prediction-reference pairs"}

    P, R, F1 = bertscore_score(preds, refs, lang=lang, verbose=True)
    return {
        "available": True,
        "num_pairs": len(preds),
        "precision_mean": round(float(P.mean().item()), 6),
        "recall_mean": round(float(R.mean().item()), 6),
        "f1_mean": round(float(F1.mean().item()), 6),
    }


def compute_bleurt(preds: List[str], refs: List[str], checkpoint: Optional[str]) -> Dict[str, Any]:
    if not HAS_BLEURT:
        return {"available": False, "reason": "bleurt not installed"}
    if not checkpoint:
        return {"available": False, "reason": "BLEURT checkpoint path not provided"}
    if not preds or not refs:
        return {"available": False, "reason": "No prediction-reference pairs"}

    scorer = bleurt_score.BleurtScorer(checkpoint)
    scores = scorer.score(references=refs, candidates=preds)

    return {
        "available": True,
        "num_pairs": len(scores),
        "mean": round(float(np.mean(scores)), 6),
        "std": round(float(np.std(scores)), 6),
        "min": round(float(np.min(scores)), 6),
        "max": round(float(np.max(scores)), 6),
    }


def compute_mauve_metric(preds: List[str], refs: List[str], max_text_length: int = 512) -> Dict[str, Any]:
    if not HAS_MAUVE:
        return {"available": False, "reason": "mauve-text not installed"}
    if not preds or not refs:
        return {"available": False, "reason": "No prediction-reference pairs"}

    out = mauve.compute_mauve(
        p_text=preds,
        q_text=refs,
        max_text_length=max_text_length,
        verbose=True
    )

    return {
        "available": True,
        "num_pairs": len(preds),
        "mauve": round(float(out.mauve), 6)
    }


# ============================================================
# PER-LANGUAGE ANALYTICS
# ============================================================

def add_doc_languages(dataset: List[Dict[str, Any]], doc_lang_lookup: Dict[str, str]) -> List[Dict[str, Any]]:
    out = []
    for item in dataset:
        item2 = dict(item)
        record_id = str(item.get("source_record_id", ""))
        item2["doc_language"] = doc_lang_lookup.get(record_id, "unknown")
        out.append(item2)
    return out


def compute_per_language_metrics(dataset: List[Dict[str, Any]], mattr_window: int = 500) -> Dict[str, Any]:
    buckets = defaultdict(list)
    for item in dataset:
        buckets[item.get("doc_language", "unknown")].append(item)

    out = {}
    for lang, rows in buckets.items():
        texts = flatten_texts(rows)
        answer_texts = texts["answers"]

        out[lang] = {
            "count": len(rows),
            "structure": compute_structure_metrics(rows),
            "lexical": compute_lexical_metrics(rows, mattr_window=mattr_window),
            "self_bleu_answers": compute_self_bleu(answer_texts, sample_size=min(200, len(answer_texts)))
        }

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Full research-grade analytics for cleaned instruction dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to instruction_dataset_evaluated_clean.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--source_csv", type=str, default=None, help="Optional source CSV with doc_id/language")
    parser.add_argument("--id_column", type=str, default="doc_id")
    parser.add_argument("--language_column", type=str, default="language")
    parser.add_argument("--judged_rows_path", type=str, default=None, help="Optional path to judged_rows_merged.json")
    parser.add_argument("--reference_path", type=str, default=None, help="Optional aligned reference file for BERTScore/BLEURT/MAUVE")
    parser.add_argument("--reference_answer_column", type=str, default="reference_answer")
    parser.add_argument("--bleurt_checkpoint", type=str, default=None)
    parser.add_argument("--mattr_window", type=int, default=500)
    parser.add_argument("--self_bleu_sample_size", type=int, default=300)
    parser.add_argument("--embedding_sample_size", type=int, default=1000)
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    doc_lang_lookup = build_doc_language_lookup(
        source_csv=args.source_csv,
        id_column=args.id_column,
        language_column=args.language_column
    )
    dataset = add_doc_languages(dataset, doc_lang_lookup)

    texts = flatten_texts(dataset)

    print("Computing structure metrics...")
    structure_metrics = compute_structure_metrics(dataset)

    print("Computing lexical diversity metrics...")
    lexical_metrics = compute_lexical_metrics(dataset, mattr_window=args.mattr_window)

    print("Computing readability/complexity metrics...")
    readability_metrics = compute_readability_metrics(dataset)

    print("Computing Self-BLEU...")
    self_bleu_answers = compute_self_bleu(
        texts["answers"],
        sample_size=args.self_bleu_sample_size,
        seed=args.seed
    )

    print("Computing embedding-based diversity metrics...")
    embedding_metrics_answers = compute_embedding_metrics(
        texts["answers"],
        model_name=args.embedding_model,
        sample_size=args.embedding_sample_size,
        seed=args.seed
    )

    print("Computing per-language metrics...")
    per_language_metrics = compute_per_language_metrics(dataset, mattr_window=args.mattr_window)

    print("Loading judge metrics...")
    judge_metrics = compute_judge_metrics(args.judged_rows_path)

    print("Loading optional reference-based metrics...")
    reference_rows = load_reference_file(args.reference_path)
    preds, refs, ref_info = build_prediction_reference_pairs(
        dataset,
        reference_rows,
        reference_answer_column=args.reference_answer_column
    )

    bertscore_metrics = compute_bertscore(preds, refs, lang="en")
    bleurt_metrics = compute_bleurt(preds, refs, checkpoint=args.bleurt_checkpoint)
    mauve_metrics = compute_mauve_metric(preds, refs)

    final_metrics = {
        "dataset_path": args.dataset_path,
        "num_examples": len(dataset),
        "metrics": {
            "structure": structure_metrics,
            "lexical_diversity": lexical_metrics,
            "readability_and_complexity": readability_metrics,
            "self_bleu_answers": self_bleu_answers,
            "semantic_diversity_answers": embedding_metrics_answers,
            "judge_quality_metrics": judge_metrics,
            "per_language_metrics": per_language_metrics,
            "reference_pairing_info": ref_info,
            "bertscore": bertscore_metrics,
            "bleurt": bleurt_metrics,
            "mauve": mauve_metrics,
        },
        "environment": {
            "textstat_available": HAS_TEXTSTAT,
            "sentence_transformers_available": HAS_SENTENCE_TRANSFORMERS,
            "bertscore_available": HAS_BERTSCORE,
            "bleurt_available": HAS_BLEURT,
            "mauve_available": HAS_MAUVE,
            "nltk_available": HAS_NLTK,
        }
    }

    metrics_path = output_dir / "full_research_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    # Flat summary for quick reading
    quick_summary = {
        "num_examples": len(dataset),
        "refusal_ratio": structure_metrics["refusal_ratio"],
        "avg_turns_per_example": structure_metrics["turns_per_example"]["mean"],
        "avg_question_tokens": structure_metrics["question_token_lengths"]["mean"],
        "avg_answer_tokens": structure_metrics["answer_token_lengths"]["mean"],
        "all_text_ttr": lexical_metrics["all_text"]["ttr"],
        "all_text_mattr": lexical_metrics["all_text"]["mattr"],
        "all_text_entropy": lexical_metrics["all_text"]["entropy"],
        "answers_distinct_1": lexical_metrics["answers"]["distinct_1"],
        "answers_distinct_2": lexical_metrics["answers"]["distinct_2"],
        "self_bleu_answers": self_bleu_answers.get("self_bleu"),
        "semantic_avg_pairwise_cosine_distance": embedding_metrics_answers.get("average_pairwise_cosine_distance"),
        "semantic_vendi_score": embedding_metrics_answers.get("vendi_score"),
        "semantic_normalized_vendi_score": embedding_metrics_answers.get("normalized_vendi_score"),
        "groundedness_score": judge_metrics.get("scores", {}).get("groundedness_score"),
        "hallucination_rate": judge_metrics.get("rates", {}).get("hallucination_rate"),
        "fully_grounded_rate": judge_metrics.get("rates", {}).get("fully_grounded_rate"),
        "question_quality_score": judge_metrics.get("scores", {}).get("question_quality_score"),
        "answer_quality_score": judge_metrics.get("scores", {}).get("answer_quality_score"),
        "bertscore_f1": bertscore_metrics.get("f1_mean"),
        "bleurt_mean": bleurt_metrics.get("mean"),
        "mauve": mauve_metrics.get("mauve"),
    }

    quick_path = output_dir / "quick_summary.json"
    with open(quick_path, "w", encoding="utf-8") as f:
        json.dump(quick_summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {quick_path}")
    print("\nQuick summary:")
    print(json.dumps(quick_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()