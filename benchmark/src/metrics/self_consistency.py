from __future__ import annotations

import logging
import os
import re
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from anthropic import Anthropic, RateLimitError, APIStatusError
from dotenv import load_dotenv

from utils.eval_embeddings import embed

load_dotenv()
logger = logging.getLogger(__name__)

_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

_N_SAMPLES: int             = 3
_CONSISTENCY_THRESHOLD: float = 0.70

_SENTENCE_END_RE = re.compile(
    r"(?<=[^\d])[.!?।](?=\s+[A-Z\u0D80-\u0DFF\u0B80-\u0BFF]|\s*$)"
)
_MIN_SENTENCE_WORDS = 4

# Pause before starting sampling to let the token-per-minute window recover
# after SAFE's web-search calls.  SAFE uses ~10-15 calls × ~3k tokens each.
# At 30k TPM limit a 30s pause is sufficient to replenish the bucket.
_PRE_SAMPLE_COOLDOWN = 30
_INTER_SAMPLE_DELAY  = 8    # between individual sample calls


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END_RE.split(text)
    return [
        p.strip() for p in parts
        if p and p.strip() and len(p.split()) >= _MIN_SENTENCE_WORDS
    ]


def _call_with_backoff(fn, *args, max_retries: int = 4, **kwargs):
    """Exponential backoff: 15s → 30s → 60s → 120s on 429."""
    delays = [15, 30, 60, 120]
    for attempt, delay in enumerate(delays[:max_retries], start=1):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as exc:
            if attempt == max_retries:
                raise
            logger.warning(
                "Rate limit (attempt %d/%d). Waiting %ds. %s",
                attempt, max_retries, delay, exc,
            )
            time.sleep(delay)
        except APIStatusError as exc:
            if exc.status_code == 429:
                if attempt == max_retries:
                    raise
                logger.warning(
                    "429 status (attempt %d/%d). Waiting %ds. %s",
                    attempt, max_retries, delay, exc,
                )
                time.sleep(delay)
            else:
                raise


def _sample_response(
    question: str,
    retrieved_context: str,
    temperature: float = 0.7,
) -> str:
    try:
        response = _call_with_backoff(
            _client.messages.create,
            model="claude-sonnet-4-6",
            max_tokens=500,
            temperature=temperature,
            messages=[{
                "role": "user",
                "content": (
                    f"You are a medical information assistant for Sri Lanka.\n"
                    f"Answer the following question using the provided context.\n\n"
                    f"Context:\n{retrieved_context[:2000]}\n\n"
                    f"Question: {question}"
                ),
            }],
        )
        return response.content[0].text.strip()
    except (RateLimitError, APIStatusError) as exc:
        logger.error("Sample generation exhausted retries: %s", exc)
        return ""
    except Exception as exc:
        logger.warning("Sample generation failed: %s", exc)
        return ""


def _sentence_consistency_score(
    sentence: str,
    sampled_passages: list[str],
) -> float:
    if not sampled_passages:
        return 0.0

    sent_vec = embed([sentence])
    similarities: list[float] = []

    for passage in sampled_passages:
        passage_sents = _split_sentences(passage)
        if not passage_sents:
            continue
        passage_vecs = embed(passage_sents)
        sims = cosine_similarity(sent_vec, passage_vecs)[0]
        similarities.append(float(np.max(sims)))

    return float(np.mean(similarities)) if similarities else 0.0


def self_consistency_score(
    question: str,
    original_answer: str,
    retrieved_chunks: list[str],
    n_samples: int = _N_SAMPLES,
    skip_cooldown: bool = False,
) -> tuple[float, list[dict]]:
    """
    SelfCheckGPT-style sampling consistency.

    Parameters
    ----------
    skip_cooldown : bool
        Set True in tests or when self_consistency is called in isolation
        (not after safe_web_verification).  Default False inserts a
        cooldown to let the token-per-minute window recover after SAFE.
    """
    sentences = _split_sentences(original_answer)
    if not sentences:
        return 0.0, [{"error": "no_sentences"}]

    retrieved_context = "\n\n".join(retrieved_chunks[:3])

    # Wait for token bucket to recover after safe_web_verification
    if not skip_cooldown:
        logger.info(
            "self_consistency: waiting %ds for rate-limit cooldown after SAFE.",
            _PRE_SAMPLE_COOLDOWN,
        )
        time.sleep(_PRE_SAMPLE_COOLDOWN)

    sampled_passages: list[str] = []
    for i in range(n_samples):
        if i > 0:
            time.sleep(_INTER_SAMPLE_DELAY)
        sample = _sample_response(question, retrieved_context)
        if sample:
            sampled_passages.append(sample)
            logger.debug("Sample %d/%d generated (%d chars)", i + 1, n_samples, len(sample))
        else:
            logger.warning("Sample %d/%d empty — skipped", i + 1, n_samples)

    if not sampled_passages:
        logger.warning("All sample generations failed — returning 0.0")
        return 0.0, [{"error": "all_samples_failed"}]

    consistent = 0
    details: list[dict] = []

    for sentence in sentences:
        sim = _sentence_consistency_score(sentence, sampled_passages)
        is_consistent = sim >= _CONSISTENCY_THRESHOLD
        if is_consistent:
            consistent += 1
        details.append({
            "sentence":   sentence,
            "mean_sim":   round(sim, 4),
            "consistent": is_consistent,
        })

    score = consistent / len(sentences)
    return round(score, 4), details