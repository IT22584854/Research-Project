from __future__ import annotations
 
import logging
import re
 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
 
from src.judge import call_claude, MetricType
from utils.eval_embeddings import embed
 
logger = logging.getLogger(__name__)
 
_INTERNAL_PENALTY       = 0.40
_CONTEXT_PENALTY_EACH   = 0.15
_MAX_CONTEXT_PENALTY    = 0.45
_COSINE_PREFILTER       = 0.50   # only NLI-judge chunk pairs with some similarity
 
# Sentence splitter
_SENTENCE_END_RE = re.compile(
    r"(?<=[^\d])[.!?।](?=\s+[A-Z\u0D80-\u0DFF\u0B80-\u0BFF]|\s*$)"
)
 
 
def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END_RE.split(text)
    return [p.strip() for p in parts if p and p.strip() and len(p.split()) >= 4]
 
 
_INTERNAL_CONTRADICTION_PROMPT = """\
You are a strict medical logic checker.
 
Read the ANSWER below and identify any pairs of sentences that make
logically opposite or contradictory claims about the same topic.
 
Rules:
- The answer may be in Sinhala, Tamil, or English.
- Only flag genuine logical contradictions, not nuanced differences.
- Consider medical cases where contradictions are serious (e.g., conflicting treatment advice).
- Think step-by-step about the logical relationships between claims.
- Return ONLY a JSON object.
 
ANSWER:
{answer}
 
JSON format:
{{
  "reasoning": "<step-by-step analysis of claim relationships>",
  "contradictions_found": true or false,
  "confidence": <0.0 to 1.0>,
  "pairs": [
    {{"sentence_a": "...", "sentence_b": "...", "reason": "...", "severity": "critical|high|medium"}}
  ]
}}
"""
 
_CONTEXT_NLI_PROMPT = """\
You are a strict medical NLI (Natural Language Inference) judge.
 
Does the CONTEXT directly CONTRADICT the CLAIM?
(A contradiction means the context explicitly asserts something opposite to the claim.)
 
Step 1: Identify the main assertion in the CLAIM
Step 2: Check if CONTEXT says the opposite
Step 3: Be strict - if CONTEXT is ambiguous, it's NOT_CONTRADICTED
 
CONTEXT:
{context}
 
CLAIM:
{claim}
 
Reply with EXACTLY one word: CONTRADICTED or NOT_CONTRADICTED
 
Then provide your confidence (0.0-1.0) on the next line."""
 
 
def _check_internal_contradictions(answer: str) -> tuple[bool, list[dict], dict]:
    """
    Ask Claude if the answer contradicts itself.
    
    Returns
    -------
    tuple
        (found: bool, pairs: list, raw_response: dict)
    """
    prompt = _INTERNAL_CONTRADICTION_PROMPT.format(answer=answer[:3000])
    result = call_claude(
        prompt,
        max_tokens=600,
        metric_type=MetricType.CONTRADICTION,
        chain_of_thought=True,
        require_json=True
    )

    if "error" in result:
        logger.warning(f"Internal contradiction check error: {result.get('error')}")
        return False, [], result

    parsed = result.get("result", {})
    
    if isinstance(parsed, dict):
        found = bool(parsed.get("contradictions_found", False))
        pairs = parsed.get("pairs", [])
        return found, pairs, {
            "confidence": result.get("confidence"),
            "raw_response": result.get("raw_output"),
            "reasoning": parsed.get("reasoning", ""),
            "cached": result.get("cached", False)
        }
    else:
        # Fallback: no contradiction detected
        logger.warning(f"Unexpected result format in internal contradiction check")
        return False, [], result
 
 
def _context_contradictions(
    answer: str,
    retrieved_chunks: list[str],
) -> tuple[list[dict], dict]:
    """
    For each answer sentence with high cosine similarity to any chunk,
    ask Claude if the chunk CONTRADICTS it.
    
    Returns
    -------
    tuple
        (contradictions: list, metadata: dict)
    """
    sentences  = _split_sentences(answer)
    if not sentences or not retrieved_chunks:
        return [], {"skipped": True, "reason": "no_sentences_or_chunks"}
 
    sent_vecs  = embed(sentences)
    chunk_vecs = embed(retrieved_chunks)
    sim_matrix = cosine_similarity(sent_vecs, chunk_vecs)
 
    contradictions: list[dict] = []
    metadata = {
        "sentences_checked": len(sentences),
        "total_api_calls": 0,
        "contradictions_found": 0
    }
 
    for i, sentence in enumerate(sentences):
        sims      = sim_matrix[i]
        best_idx  = int(np.argmax(sims))
        best_sim  = float(sims[best_idx])
 
        if best_sim < _COSINE_PREFILTER:
            continue
 
        best_chunk = retrieved_chunks[best_idx]
        prompt = _CONTEXT_NLI_PROMPT.format(
            context=best_chunk[:2000],
            claim=sentence,
        )
        metadata["total_api_calls"] += 1
        
        result = call_claude(
            prompt,
            max_tokens=50,
            metric_type=MetricType.CONTRADICTION,
            require_json=False  # Response is just one word
        )

        raw_output = result.get("raw_output", "") or str(result.get("result", ""))
        confidence = result.get("confidence", 0.5)
        
        if "CONTRADICTED" in raw_output.strip().upper():
            contradictions.append({
                "sentence":   sentence,
                "chunk":      best_chunk[:300],
                "similarity": round(best_sim, 4),
                "confidence": confidence,
                "raw_response": raw_output[:100]
            })
            metadata["contradictions_found"] += 1
 
    return contradictions, metadata
 
 
def contradiction_score(
    answer: str,
    retrieved_chunks: list[str],
) -> tuple[float, dict]:
    """
    Detect internal and context-level contradictions in the answer.
 
    Returns
    -------
    score : float
        1.0 = no contradictions; lower = contradictions detected.
    details : dict
        Breakdown of internal and context contradictions found.
    """
    if not answer:
        return 1.0, {"skipped": True, "reason": "empty_answer"}
 
    # 1. Internal contradictions
    internal_found, internal_pairs, internal_meta = _check_internal_contradictions(answer)
    internal_penalty = _INTERNAL_PENALTY if internal_found else 0.0
 
    # 2. Context contradictions
    ctx_contradictions, ctx_meta = _context_contradictions(answer, retrieved_chunks)
    context_penalty    = min(
        len(ctx_contradictions) * _CONTEXT_PENALTY_EACH,
        _MAX_CONTEXT_PENALTY,
    )
 
    total_penalty = min(internal_penalty + context_penalty, 1.0)
    score         = round(1.0 - total_penalty, 4)
 
    details = {
        "score": score,
        "internal_contradiction": internal_found,
        "internal_pairs": internal_pairs,
        "internal_metadata": internal_meta,
        "context_contradictions": ctx_contradictions,
        "context_metadata": ctx_meta,
        "internal_penalty": round(internal_penalty, 4),
        "context_penalty": round(context_penalty, 4),
        "confidence": internal_meta.get("confidence", 0.5) if internal_meta else 0.5
    }
 
    return score, details