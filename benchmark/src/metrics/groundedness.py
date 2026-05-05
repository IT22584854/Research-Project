import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.eval_embeddings import embed


# ---------------------------------------------------------------------------
# Sentence splitter (multilingual-safe)
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(
    r"""
    (?<=[^\d])
    [.!?।]
    (?=\s+[A-Z\u0D80-\u0DFF\u0B80-\u0BFF]|\s*$)
    """,
    re.VERBOSE,
)

_MIN_CLAIM_WORDS = 5
# Lowered for semantic matching (ideas over exact wording)
SIMILARITY_THRESHOLD = 0.60  # Was 0.75 - now allows more semantic matches
TOKEN_OVERLAP_THRESHOLD = 0.30  # Was 0.50 - now allows paraphrased claims


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END_RE.split(text)
    sentences = [p.strip() for p in parts if p and p.strip()]
    return sentences if sentences else [text.strip()]


# ---------------------------------------------------------------------------
# Is a complete sentence?? (sentence-level)
# ---------------------------------------------------------------------------

def _is_complete_sentence(sentence: str) -> bool:
    """
    Filters out incomplete claims like:
    - 'The Director is Dr'
    - 'Symptoms include'
    - trailing unfinished phrases
    """
    return not re.search(
        r"\b(is|are|was|were|Dr|Mr|Mrs|include|includes|such as)\s*$",
        sentence.strip(),
        re.IGNORECASE
    )

# ---------------------------------------------------------------------------
# Claim extraction (sentence-level)
# ---------------------------------------------------------------------------

def _extract_claims(answer: str) -> list[str]:

    sentences = _split_sentences(answer)

    claims = []

    for sentence in sentences:
        sentence = sentence.strip()

        # --- BASIC CLEAN ---
        if len(sentence.split()) < 6:
            continue

        # --- REMOVE DISCLAIMER / GENERIC ---
        if any(phrase in sentence.lower() for phrase in [
            "this information",
            "for general",
            "consult a",
            "may be subject",
            "not a substitute"
        ]):
            continue

        # --- REMOVE FRAGMENTS (IMPORTANT FIX) ---
        # Starts with lowercase → likely broken clause
        if sentence[0].islower():
            continue

        # Starts with "who", "which", etc → incomplete
        if sentence.lower().startswith(("who", "which", "that")):
            continue

        # --- REMOVE NAME-ONLY CLAUSES ---
        # e.g. "Wijesuriya, who holds..."
        if "," in sentence:
            first_part = sentence.split(",")[0]
            if len(first_part.split()) <= 3:
                continue

        # --- OPTIONAL: SPLIT ONLY STRONG COMPOUND SENTENCES ---
        # (but keep meaningful parts)
        parts = re.split(r"\s+(and|but)\s+", sentence)

        for part in parts:
            part = part.strip()

            if len(part.split()) < 6:
                continue

            claims.append(part)

    return claims


# ---------------------------------------------------------------------------
# Token overlap check (lightweight entity validation)
# ---------------------------------------------------------------------------

def _token_overlap(claim: str, evidence: str) -> float:
    claim_tokens = set(re.findall(r"\w+", claim.lower()))
    evidence_tokens = set(re.findall(r"\w+", evidence.lower()))

    if not claim_tokens:
        return 0.0

    overlap = claim_tokens.intersection(evidence_tokens)
    return len(overlap) / len(claim_tokens)


# ---------------------------------------------------------------------------
# Groundedness scoring
# ---------------------------------------------------------------------------

def groundedness_score(
    answer: str,
    retrieved_chunks: list[str],
) -> tuple[float, list[dict]]:

    if not retrieved_chunks:
        return 0.0, []

    claims = _extract_claims(answer)

    if not claims:
        return 0.0, []

    claim_vectors = embed(claims)
    chunk_vectors = embed(retrieved_chunks)

    similarity_matrix = cosine_similarity(claim_vectors, chunk_vectors)

    supported = 0
    claim_results = []

    for i, sims in enumerate(similarity_matrix):

        best_index = int(np.argmax(sims))
        max_similarity = float(sims[best_index])
        best_chunk = retrieved_chunks[best_index]

        overlap_score = _token_overlap(claims[i], best_chunk)

        # ✅ FINAL SUPPORT DECISION (hybrid with semantic fallback)
        # Primary: both similarity and overlap meet thresholds
        # Fallback: high similarity (>0.70) but low overlap = likely paraphrased (semantic match)
        is_supported = (
            (max_similarity >= SIMILARITY_THRESHOLD and overlap_score >= TOKEN_OVERLAP_THRESHOLD)
            or (max_similarity >= 0.70 and overlap_score >= 0.20)  # Semantic fallback
        )

        if is_supported:
            supported += 1

        claim_results.append({
            "claim": claims[i],
            "supported": is_supported,
            "similarity": round(max_similarity, 4),
            "overlap": round(overlap_score, 4),
            # Optional (for debugging only — REMOVE for DB storage)
            "evidence": best_chunk
        })

    score = supported / len(claims)

    return round(score, 4), claim_results