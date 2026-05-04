import json
import re
import logging
from src.judge import call_claude, MetricType

logger = logging.getLogger(__name__)

def factual_accuracy(question, answer, retrieved_chunks):
    """
    Research-grade factual accuracy evaluation.
    
    Parameters
    ----------
    question : str
        The original question
    answer : str
        Agent's answer to evaluate
    retrieved_chunks : list[str]
        Context chunks retrieved from vector DB

    Returns
    -------
    tuple
        (score: float 0-1, details: dict with confidence, reasoning, etc.)
    """
    if not retrieved_chunks:
        return 0.0, {
            "error": "no_context",
            "confidence": 0.0,
            "verdict": "ERROR",
            "reason": "No retrieved context available"
        }

    # Build context window
    context = "\n\n".join(retrieved_chunks[:5])  # Use more context (was 3)
    context = context[:4000]  # Increased from 3000

    prompt = f"""You are a SEMANTIC medical fact-checking AI specialized in health domain evaluation.

Your task: Evaluate whether the agent's answer conveys the SAME MEANING as the context, even if worded differently.

EVALUATION RULES:
- Judge semantic meaning, NOT exact wording
- Paraphrased answers that convey the same information = SUPPORTED
- Equivalent terms (e.g., "hospital" vs "medical center", "Monday" vs "Mon") = SUPPORTED
- If the IDEA is present in context, even in different words = SUPPORTED
- Only mark NOT_SUPPORTED if the core claim is genuinely missing or contradicts
- Partial correctness → PARTIALLY_SUPPORTED
- Answer language may differ from context; judge semantic meaning

CONTEXT (authoritative source):
{context}

QUESTION:
{question}

AGENT'S ANSWER:
{answer}

---

Provide your evaluation as JSON with the following structure:

{{
  "verdict": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "NOT_SUPPORTED",
  "score": <0.0 to 1.0>,
  "confidence": <0.0 to 1.0>,
  "key_claims": [<list of main claims in the answer>],
  "verified_claims": [<claims semantically supported - even if worded differently>],
  "unsupported_claims": [<claims that genuinely contradict or are missing from context>],
  "reasoning": "<brief explanation - focus on whether the IDEA is present, not exact words>",
  "critical_issues": [<any factual contradictions or serious problems>]
}}

Return ONLY the JSON object, no additional text."""

    # Call Claude with optimized settings for factual accuracy
    response = call_claude(
        prompt=prompt,
        max_tokens=500,
        metric_type=MetricType.FACTUAL_ACCURACY,
        return_confidence=True,
        chain_of_thought=False,  # Not needed for factual accuracy
        require_json=True
    )

    # Extract results
    if "error" in response:
        logger.warning(f"Factual accuracy evaluation error: {response.get('error')}")
        return 0.0, {
            "error": response.get("error"),
            "confidence": response.get("confidence", 0.0),
            "raw_output": response.get("raw_output", ""),
            "metadata": response.get("metadata", {})
        }

    parsed = response.get("result")
    confidence = response.get("confidence", 0.5)

    if isinstance(parsed, dict):
        score = float(parsed.get("score", 0.0))
        details = {
            **parsed,
            "confidence": confidence,
            "cached": response.get("cached", False),
            "metadata": response.get("metadata", {})
        }
        return score, details
    else:
        # Fallback for malformed response
        logger.warning(f"Unexpected result format: {type(parsed)}")
        return 0.0, {
            "error": "malformed_response",
            "confidence": 0.1,
            "raw_output": response.get("raw_output", ""),
            "metadata": response.get("metadata", {})
        }
