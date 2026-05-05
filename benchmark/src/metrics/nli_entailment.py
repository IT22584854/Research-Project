from __future__ import annotations
 
import logging
import re
 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
 
from src.judge import call_claude, MetricType
from utils.eval_embeddings import embed
 
logger = logging.getLogger(__name__)
 
# Cosine threshold to skip the LLM call for clearly unrelated chunks
_COSINE_PREFILTER = 0.40
 
# Each contradicted claim subtracts this much from the entailed count
_CONTRADICTION_PENALTY = 2.0
 
# Sentence splitter — reused from groundedness / atomic_faithfulness
_SENTENCE_END_RE = re.compile(
    r"(?<=[^\d])[.!?।](?=\s+[A-Z\u0D80-\u0DFF\u0B80-\u0BFF]|\s*$)"
)
_MIN_CLAIM_WORDS = 5
 
 
def _split_into_claims(answer: str) -> list[str]:
    parts = _SENTENCE_END_RE.split(answer)
    return [
        p.strip() for p in parts
        if p and p.strip() and len(p.split()) >= _MIN_CLAIM_WORDS
    ]
 
 
_NLI_PROMPT = """\
You are a strict medical NLI (Natural Language Inference) judge.
 
Given a CONTEXT passage and a CLAIM, classify their relationship.
 
Rules:
- ENTAILED   : the context directly supports or implies the claim is true
- NEUTRAL    : the context does not address the claim; claim may be true or false
- CONTRADICTED: the context directly states something that conflicts with the claim
 
The claim may be in Sinhala, Tamil, or English.  Judge meaning, not language.
Be strict: if the context only partially supports the claim, classify NEUTRAL.
 
Think through the logical relationships step-by-step:
1. What is the CLAIM asserting?
2. What does the CONTEXT say about this topic?
3. Is there direct support, contradiction, or neither?

CONTEXT:
{context}
 
CLAIM:
{claim}
 
Reply with EXACTLY one word: ENTAILED, NEUTRAL, or CONTRADICTED
Then on the next line provide your confidence (0.0-1.0)."""
 
 
def _nli_verdict(claim: str, context_chunk: str) -> tuple[str, float]:
    """
    Return 'ENTAILED', 'NEUTRAL', or 'CONTRADICTED' with confidence.
    
    Returns
    -------
    tuple
        (verdict: str, confidence: float)
    """
    prompt = _NLI_PROMPT.format(
        context=context_chunk[:2000],
        claim=claim,
    )
    result = call_claude(
        prompt,
        max_tokens=50,
        metric_type=MetricType.NLI_ENTAILMENT,
        require_json=False  # Response is free-form
    )

    raw = result.get("raw_output", "") or str(result.get("result", ""))
    confidence = result.get("confidence", 0.5)
 
    verdict = raw.strip().upper()
 
    if "ENTAILED" in verdict:
        return "ENTAILED", confidence
    if "CONTRADICT" in verdict:
        return "CONTRADICTED", confidence
    return "NEUTRAL", confidence
 
 
def nli_entailment_score(
    answer: str,
    retrieved_chunks: list[str],
) -> tuple[float, list[dict]]:
    """
    Score the answer by checking how many of its claims are genuinely
    ENTAILED (not merely semantically similar) by the retrieved context.
 
    Parameters
    ----------
    answer : str
        Agent response, any language.
    retrieved_chunks : list[str]
        Retrieved passages from Pinecone.
 
    Returns
    -------
    score : float
        NLI entailment score in [0.0, 1.0].
    details : list[dict]
        Per-claim breakdown with verdict and evidence.
    """
    claims = _split_into_claims(answer)
    if not claims:
        return 1.0, [{"skipped": True, "reason": "no_claims_found"}]
 
    if not retrieved_chunks:
        return 0.0, [{"skipped": True, "reason": "no_context"}]
 
    # Embed claims and chunks for similarity matching
    claim_vecs  = embed(claims)
    chunk_vecs  = embed(retrieved_chunks)
    sim_matrix  = cosine_similarity(claim_vecs, chunk_vecs)
 
    # For each claim, find best matching chunk and evaluate entailment
    details: list[dict] = []
    entailed_count = 0
    contradicted_count = 0
    neutral_count = 0
    
    for i, claim in enumerate(claims):
        sims = sim_matrix[i]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
 
        # Skip if similarity too low (context doesn't address this claim)
        if best_sim < _COSINE_PREFILTER:
            verdict, confidence = "NEUTRAL", 0.1
            best_chunk = None
        else:
            best_chunk = retrieved_chunks[best_idx]
            verdict, confidence = _nli_verdict(claim, best_chunk)
 
        # Track verdicts
        if verdict == "ENTAILED":
            entailed_count += 1
        elif verdict == "NEUTRAL":
            neutral_count += 1
        elif verdict == "CONTRADICTED":
            contradicted_count += 1
 
        # Record details
        details.append({
            "claim": claim,
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "best_sim": round(best_sim, 3),
            "evidence_chunk": best_chunk[:300] if best_chunk else None,
        })
 
    # Compute score
    # Score = (entailed - contradicted * penalty) / total
    weighted_entailed = entailed_count - (contradicted_count * _CONTRADICTION_PENALTY)
    score = max(0.0, weighted_entailed / len(claims)) if claims else 0.0
    score = min(1.0, score)
 
    # Add summary
    details.append({
        "summary": {
            "total_claims": len(claims),
            "entailed": entailed_count,
            "neutral": neutral_count,
            "contradicted": contradicted_count,
            "score": round(score, 3)
        }
    })
 
    return round(score, 4), details
