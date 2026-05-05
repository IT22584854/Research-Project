import os
import json
import time
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple
from anthropic import Anthropic, APIError, RateLimitError
from dotenv import load_dotenv
from enum import Enum

load_dotenv()
logger = logging.getLogger(__name__)

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Cache for API responses (avoid redundant calls)
_response_cache: Dict[str, Dict[str, Any]] = {}

class MetricType(Enum):
    """Metric types with optimized configurations"""
    FACTUAL_ACCURACY = {"temp": 0.0, "model": "claude-sonnet-4-6"}
    NLI_ENTAILMENT = {"temp": 0.0, "model": "claude-sonnet-4-6"}
    CONTRADICTION = {"temp": 0.1, "model": "claude-sonnet-4-6"}
    SEMANTIC_SIMILARITY = {"temp": 0.2, "model": "claude-sonnet-4-6"}
    COMPLETENESS = {"temp": 0.3, "model": "claude-sonnet-4-6"}
    DEFAULT = {"temp": 0.0, "model": "claude-sonnet-4-6"}


def _get_prompt_hash(prompt: str) -> str:
    """Generate hash of prompt for caching"""
    return hashlib.md5(prompt.encode()).hexdigest()


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract JSON from LLM output with multiple strategies.
    
    Tries:
    1. Parse entire response as JSON
    2. Extract {..} block from response
    3. Extract ```json``` block
    """
    text = text.strip()
    
    # Strategy 1: Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Find {..} block
    import re
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Find ```json``` block
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass
    
    return None


def call_claude(
    prompt: str,
    max_tokens: int = 300,
    metric_type: MetricType = MetricType.DEFAULT,
    return_confidence: bool = True,
    use_caching: bool = True,
    require_json: bool = True,
    chain_of_thought: bool = False
) -> Dict[str, Any]:
    """
    Enhanced Claude calling function for research-grade evaluation.
    
    Parameters
    ----------
    prompt : str
        The prompt to send to Claude
    max_tokens : int
        Maximum tokens in response
    metric_type : MetricType
        Type of metric (determines temperature, model)
    return_confidence : bool
        Whether to compute confidence score
    use_caching : bool
        Whether to use cached responses
    require_json : bool
        Whether output must be valid JSON
    chain_of_thought : bool
        Whether to request step-by-step reasoning
        
    Returns
    -------
    dict
        {
            "result": parsed JSON or raw text,
            "confidence": 0.0-1.0 confidence score,
            "raw_output": original response,
            "cached": whether result was cached,
            "metadata": {"model", "temp", "tokens_used", "latency_ms"}
        }
    """
    
    # Check cache
    prompt_hash = _get_prompt_hash(prompt)
    if use_caching and prompt_hash in _response_cache:
        return {
            **_response_cache[prompt_hash],
            "cached": True
        }
    
    config = metric_type.value if isinstance(metric_type, MetricType) else MetricType.DEFAULT.value
    model = config["model"]
    temperature = config["temp"]
    
    # Add chain-of-thought instruction if requested
    enhanced_prompt = prompt
    if chain_of_thought and require_json:
        enhanced_prompt = prompt + "\n\nThink step-by-step before providing the JSON result."
    
    max_retries = 5
    retry_delay = 0.5  # Initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": enhanced_prompt}
                ]
            )
            
            latency_ms = (time.time() - start_time) * 1000
            output_text = response.content[0].text.strip()
            
            # Try to parse JSON if required
            parsed_result = None
            confidence = 0.5  # Default confidence
            
            if require_json:
                parsed_result = _extract_json_from_text(output_text)
                
                if parsed_result is None:
                    if attempt < max_retries - 1:
                        logger.warning(f"Failed to parse JSON (attempt {attempt + 1}/{max_retries}): {output_text[:100]}")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        # Last attempt failed - return error with raw output
                        result = {
                            "error": "json_parse_failed",
                            "raw_output": output_text,
                            "result": None,
                            "confidence": 0.1,
                            "cached": False,
                            "metadata": {
                                "model": model,
                                "temperature": temperature,
                                "latency_ms": latency_ms,
                                "attempt": attempt + 1,
                                "max_retries": max_retries
                            }
                        }
                        _response_cache[prompt_hash] = result
                        return result
                
                # Calculate confidence based on parsed content
                if isinstance(parsed_result, dict):
                    # If result has explicit confidence field, use it
                    if "confidence" in parsed_result:
                        confidence = float(parsed_result.get("confidence", 0.5))
                    # Otherwise infer from verdict/score
                    elif "score" in parsed_result:
                        confidence = float(parsed_result.get("score", 0.5))
                    elif "verdict" in parsed_result:
                        verdict = str(parsed_result.get("verdict", "")).upper()
                        confidence = 0.9 if verdict in ["SUPPORTED", "ENTAILED"] else 0.5
                    else:
                        confidence = 0.7
            else:
                parsed_result = output_text
                confidence = 0.8
            
            # Success - cache and return
            result = {
                "result": parsed_result,
                "confidence": confidence,
                "raw_output": output_text,
                "cached": False,
                "metadata": {
                    "model": model,
                    "temperature": temperature,
                    "latency_ms": round(latency_ms, 2),
                    "attempt": attempt + 1,
                    "tokens_used": response.usage.output_tokens
                }
            }
            
            _response_cache[prompt_hash] = result
            return result
        
        except RateLimitError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}), waiting {retry_delay}s")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"Rate limit exceeded after {max_retries} attempts")
                return {
                    "error": "rate_limit",
                    "message": str(e),
                    "result": None,
                    "confidence": 0.0,
                    "cached": False,
                    "metadata": {
                        "model": model,
                        "attempt": attempt + 1,
                        "max_retries": max_retries
                    }
                }
        
        except APIError as e:
            if attempt < max_retries - 1:
                logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"API error after {max_retries} attempts: {str(e)}")
                return {
                    "error": "api_error",
                    "message": str(e),
                    "result": None,
                    "confidence": 0.0,
                    "cached": False,
                    "metadata": {
                        "model": model,
                        "attempt": attempt + 1,
                        "max_retries": max_retries
                    }
                }
        
        except Exception as e:
            logger.error(f"Unexpected error in call_claude: {str(e)}")
            return {
                "error": "unexpected_error",
                "message": str(e),
                "result": None,
                "confidence": 0.0,
                "cached": False,
                "metadata": {
                    "model": model,
                    "attempt": attempt + 1
                }
            }
    
    # Should not reach here
    return {
        "error": "max_retries_exceeded",
        "result": None,
        "confidence": 0.0,
        "cached": False,
        "metadata": {
            "model": model,
            "max_retries": max_retries
        }
    }


def clear_cache():
    """Clear response cache (useful for testing)"""
    global _response_cache
    _response_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return {
        "cache_size": len(_response_cache),
        "cached_responses": list(_response_cache.keys())[:10]  # Show first 10
    }
