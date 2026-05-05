import json
import re
import logging
from src.judge import call_claude, MetricType

logger = logging.getLogger(__name__)


def clean_json_response(text: str) -> str:
    """
    Removes markdown fences and extra formatting from LLM JSON output.
    """
    if not text:
        return ""

    text = text.strip()

    # Remove ```json and ``` wrappers
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```$", "", text)

    return text.strip()


def safe_json_parse(text: str):
    """
    Safely parse JSON with fallback handling.
    """
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)


def factual_accuracy(question, answer, retrieved_chunks):
    """
    Research-grade factual accuracy evaluation.

    Returns:
        (score: float, details: dict)
    """

    if not retrieved_chunks:
        return 0.0, {
            "error": "no_context",
            "confidence": 0.0,
            "verdict": "ERROR",
            "reason": "No retrieved context available"
        }

    # Build context window
    context = "\n\n".join(retrieved_chunks[:5])
    context = context[:4000]

    prompt = f"""You are a SEMANTIC medical fact-checking AI specialized in health domain evaluation.

Your task: Evaluate whether the agent's answer conveys the SAME MEANING as the context, even if worded differently.

EVALUATION RULES:
- Judge semantic meaning, NOT exact wording
- Paraphrased answers that convey the same information = SUPPORTED
- Equivalent terms = SUPPORTED
- If idea exists in context = SUPPORTED
- Only mark NOT_SUPPORTED if missing or contradicting
- Partial correctness allowed

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Return ONLY valid JSON (no markdown, no backticks):

{{
  "verdict": "SUPPORTED | PARTIALLY_SUPPORTED | NOT_SUPPORTED",
  "score": 0.0,
  "confidence": 0.0,
  "key_claims": [],
  "verified_claims": [],
  "unsupported_claims": [],
  "reasoning": "",
  "critical_issues": []
}}
"""

    response = call_claude(
        prompt=prompt,
        max_tokens=500,
        metric_type=MetricType.FACTUAL_ACCURACY,
        return_confidence=True,
        chain_of_thought=False,
        require_json=False  # IMPORTANT: we handle parsing ourselves now
    )

    # -----------------------------
    # HANDLE API ERROR
    # -----------------------------
    if "error" in response:
        logger.warning(f"Factual accuracy evaluation error: {response.get('error')}")
        return 0.0, {
            "error": response.get("error"),
            "confidence": response.get("confidence", 0.0),
            "raw_output": response.get("raw_output", ""),
            "metadata": response.get("metadata", {})
        }

    raw_output = response.get("raw_output", "")
    cleaned = clean_json_response(raw_output)

    parsed, parse_error = safe_json_parse(cleaned)

    # -----------------------------
    # HANDLE PARSE FAILURE (IMPORTANT FIX)
    # -----------------------------
    if parse_error:
        logger.error(f"JSON parse failed: {parse_error}")
        logger.debug(f"Raw output: {raw_output}")

        return 0.0, {
            "error": "json_parse_failed",
            "confidence": response.get("confidence", 0.0),
            "raw_output": raw_output,
            "cleaned_output": cleaned,
            "parse_error": parse_error,
            "metadata": response.get("metadata", {})
        }

    # -----------------------------
    # VALID RESPONSE
    # -----------------------------
    if isinstance(parsed, dict):
        score = float(parsed.get("score", 0.0))

        details = {
            **parsed,
            "confidence": response.get("confidence", 0.5),
            "cached": response.get("cached", False),
            "metadata": response.get("metadata", {})
        }

        return score, details

    # -----------------------------
    # FALLBACK (NEVER LOSE SIGNAL)
    # -----------------------------
    logger.warning(f"Unexpected parsed format: {type(parsed)}")

    return 0.0, {
        "error": "malformed_response",
        "confidence": 0.1,
        "raw_output": raw_output,
        "metadata": response.get("metadata", {})
    }