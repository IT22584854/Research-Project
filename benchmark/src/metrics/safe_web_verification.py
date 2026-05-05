from __future__ import annotations

import json
import logging
import os
import re
import time

from anthropic import Anthropic, RateLimitError, APIStatusError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

_CONTRADICTION_WEIGHT = 1.5

FACTS_PER_CALL   = 4   # facts verified per single API call
BATCH_DELAY_SECS = 8   # seconds between batch calls


# ─────────────────────────────────────────────────────────────────────────────
# Retry helper
# ─────────────────────────────────────────────────────────────────────────────

def _call_with_backoff(fn, *args, max_retries: int = 3, **kwargs):
    """
    Exponential backoff on 429.  Schedule: 20s → 45s → 90s.
    After max_retries the exception is re-raised.
    """
    delays = [20, 45, 90]
    for attempt, delay in enumerate(delays[:max_retries], start=1):
        try:
            return fn(*args, **kwargs)
        except (RateLimitError, APIStatusError) as exc:
            status = getattr(exc, "status_code", 429)
            if status != 429 or attempt == max_retries:
                raise
            logger.warning(
                "Rate limit (attempt %d/%d). Waiting %ds.",
                attempt, max_retries, delay,
            )
            time.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Decompose into atomic facts
# ─────────────────────────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """\
You are a medical fact extractor.

Decompose the ANSWER into atomic, self-contained facts for web verification.

Rules:
- Answer may be Sinhala, Tamil, or English.
- Each fact must be independently verifiable (resolve pronouns, include names).
- Omit opinions, disclaimers, and "consult a doctor" sentences.
- Return ONLY a JSON array of strings.

QUESTION: {question}
ANSWER: {answer}
"""


def _decompose(question: str, answer: str) -> list[str]:
    try:
        resp = _call_with_backoff(
            _client.messages.create,
            model="claude-sonnet-4-6",
            max_tokens=600,
            temperature=0,
            messages=[{
                "role": "user",
                "content": _DECOMPOSE_PROMPT.format(
                    question=question,
                    answer=answer[:2500],
                ),
            }],
        )
        raw = resp.content[0].text.strip()

        try:
            facts = json.loads(raw)
            if isinstance(facts, list):
                return [str(f) for f in facts]
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            return [str(f) for f in json.loads(match.group())]

        return [
            l.strip().lstrip("•-–*").strip()
            for l in raw.splitlines()
            if l.strip() and len(l.split()) >= 4
        ]

    except Exception as exc:
        logger.warning("Decomposition failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Batched verification (multiple facts per API call)
# ─────────────────────────────────────────────────────────────────────────────

_BATCH_VERIFY_SYSTEM = """\
You are a strict medical fact-checker for Sri Lanka.
You have access to a web search tool.

You will receive a numbered list of facts to verify.
For each fact:
1. Issue ONE concise search query covering as many facts as possible.
2. Read the results.
3. Return a JSON array, one object per fact, in the same order:
   [
     {"id": 1, "verdict": "SUPPORTED"|"NOT_SUPPORTED"|"CONTRADICTED",
      "reason": "one sentence", "sources": ["url1"]},
     ...
   ]

Verdict rules:
- SUPPORTED:     results clearly confirm the fact.
- NOT_SUPPORTED: results don't address it or are insufficient.
- CONTRADICTED:  results explicitly state the opposite.

Prefer .gov.lk, who.int, nih.gov, peer-reviewed sources.
Return ONLY the JSON array, no other text.
"""


def _format_batch(facts: list[str], start_id: int) -> str:
    lines = [f"{start_id + i}. {fact}" for i, fact in enumerate(facts)]
    return "\n".join(lines)


def _verify_batch(facts: list[str], start_id: int) -> list[dict]:
    """
    Verify a batch of facts in a single API call.
    Returns one result dict per fact: {id, verdict, reason, sources, errored}.
    """
    n = len(facts)
    default = [
        {"id": start_id + i, "verdict": "NOT_SUPPORTED",
         "reason": "batch_call_failed", "sources": [], "errored": True}
        for i in range(n)
    ]

    try:
        resp = _call_with_backoff(
            _client.messages.create,
            model="claude-sonnet-4-6",
            max_tokens=600,
            temperature=0,
            system=_BATCH_VERIFY_SYSTEM,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": (
                    f"Verify these facts:\n\n"
                    f"{_format_batch(facts, start_id)}"
                ),
            }],
        )

        text = " ".join(
            b.text for b in resp.content if hasattr(b, "text") and b.text
        ).strip()

        # Extract JSON array
        try:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                results = []
                for item in parsed:
                    results.append({
                        "id":      item.get("id", start_id),
                        "verdict": item.get("verdict", "NOT_SUPPORTED"),
                        "reason":  item.get("reason", ""),
                        "sources": item.get("sources", []),
                        "errored": False,
                    })
                # Pad with errors if Claude returned fewer items than expected
                while len(results) < n:
                    results.append({
                        "id":      start_id + len(results),
                        "verdict": "NOT_SUPPORTED",
                        "reason":  "missing_from_batch_response",
                        "sources": [],
                        "errored": True,
                    })
                return results[:n]
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: couldn't parse JSON — mark all in batch as errored
        logger.warning("Could not parse batch response JSON for batch starting at %d", start_id)
        return default

    except (RateLimitError, APIStatusError) as exc:
        logger.error("Batch verification exhausted retries (start_id=%d): %s", start_id, exc)
        for d in default:
            d["reason"] = f"rate_limit_exhausted: {exc}"
        return default

    except Exception as exc:
        logger.warning("Batch verification failed (start_id=%d): %s", start_id, exc)
        for d in default:
            d["reason"] = f"verification_error: {exc}"
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────────────────────────────────────

def safe_web_verification(
    question: str,
    answer: str,
) -> tuple[float, list[dict], list[str]]:
    """
    SAFE-style atomic fact verification using batched web-search calls.

    Token consumption
    -----------------
    Old: 1 call/fact × ~4k tokens = 44k tokens for 11 facts  →  429
    New: ceil(11/4) = 3 calls × ~2k tokens = ~6k tokens total →  safe

    Scoring
    -------
    Only successfully verified facts (errored=False) count in the denominator.

    score = (n_supported - CONTRADICTION_WEIGHT × n_contradicted) / n_verified
    clamped to [0.0, 1.0].
    """
    facts = _decompose(question, answer)

    if not facts:
        logger.warning("SAFE: no facts extracted — returning 0.0")
        return 0.0, [{"error": "decomposition_failed"}], []

    all_results: list[dict] = []
    all_urls:    list[str]  = []

    # Split facts into batches and verify one batch per API call
    batches = [
        facts[i: i + FACTS_PER_CALL]
        for i in range(0, len(facts), FACTS_PER_CALL)
    ]

    for batch_num, batch in enumerate(batches):
        if batch_num > 0:
            time.sleep(BATCH_DELAY_SECS)

        start_id = batch_num * FACTS_PER_CALL + 1
        results  = _verify_batch(batch, start_id)
        all_results.extend(results)

        for r in results:
            all_urls.extend(r.get("sources", []))

    # Build per-fact details aligned with the original facts list
    supported    = 0
    contradicted = 0
    verified     = 0
    details: list[dict] = []

    for i, (fact, result) in enumerate(zip(facts, all_results)):
        errored = result.get("errored", False)
        verdict = result.get("verdict", "NOT_SUPPORTED")

        if not errored:
            verified += 1
            if verdict == "SUPPORTED":
                supported += 1
            elif verdict == "CONTRADICTED":
                contradicted += 1

        details.append({
            "fact":    fact,
            "verdict": verdict,
            "reason":  result.get("reason", ""),
            "sources": result.get("sources", []),
            "errored": errored,
        })

    if verified == 0:
        logger.warning("SAFE: no facts verified — returning 0.0")
        score = 0.0
    else:
        raw   = (supported - _CONTRADICTION_WEIGHT * contradicted) / verified
        score = max(0.0, min(1.0, raw))

    found_urls = list(dict.fromkeys(all_urls))

    logger.info(
        "SAFE: %d facts | %d batches | %d verified | %d supported "
        "| %d contradicted | %d errored | score=%.3f",
        len(facts), len(batches), verified, supported,
        contradicted, len(facts) - verified, score,
    )

    return round(score, 4), details, found_urls