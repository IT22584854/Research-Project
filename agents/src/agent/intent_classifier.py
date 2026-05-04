"""Intent classifier router for healthcare request routing."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

# Path setup for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.src.config import (
    ROUTER_API_KEY,
    ROUTER_BASE_URL,
    ROUTER_MAX_TOKENS,
    ROUTER_MODEL,
    ROUTER_TEMPERATURE,
)
from agents.src.graph.state import AgentInputState, AgentState
from agents.src.utils import setup_logger

logger = setup_logger("intent_classifier_agent")

ALLOWED_INTENTS = {
    "Emergency_Triage",
    "Symptom_Information",
    "Disease_Information",
    "Facility_Locator",
    "Provider_Locator",
    "Appointment_Booking",
    "Medication_Information",
    "Vaccine_Information",
    "Test_Diagnostics",
    "Treatment_Procedure",
    "Cost_Insurance",
    "General_Health_Education",
    "Non_Medical",
    "Unclear",
}

MEDICAL_INFO_INTENTS = ALLOWED_INTENTS - {"Emergency_Triage", "Non_Medical", "Unclear"}

ROUTER_ALLOWED_KEYS = (
    "language",
    "intent",
    "keywords",
    "english_translation_or_summary",
)

INTENT_ALIASES = {
    "Emergency": "Emergency_Triage",
    "Emergency Triage": "Emergency_Triage",
    "Medical_Emergency": "Emergency_Triage",
    "Medical Emergency": "Emergency_Triage",
    "NonMedical": "Non_Medical",
    "Non-Medical": "Non_Medical",
    "Non Medical": "Non_Medical",
}

EMERGENCY_CONTACT_RESPONSE = """Emergency Contact

Ambulance Service: 1990
Children and Women Welfare Hotline:1929
General Government Helpline : 1919"""

NON_MEDICAL_RESPONSE = (
    "I can help with Sri Lankan medical and public-health questions. "
    "Please ask a health-related question so I can route it safely."
)

SOCIAL_ONLY_INDICATORS = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
}

EMERGENCY_INDICATORS = (
    "emergency",
    "cannot breathe",
    "can't breathe",
    "difficulty breathing",
    "shortness of breath",
    "severe chest pain",
    "chest pain",
    "unconscious",
    "stroke",
    "seizure",
    "severe bleeding",
    "poisoning",
    "overdose",
    "suicide",
    "suicidal",
    "anaphylaxis",
    "severe allergic",
)

DETERMINISTIC_INTENT_PATTERNS = (
    ("Facility_Locator", ("hospital", "hospitals", "clinic", "clinics", "medical center", "health centre", "health center")),
    ("Provider_Locator", ("doctor", "doctors", "specialist", "consultant", "physician", "surgeon")),
    ("Appointment_Booking", ("appointment", "booking", "book", "schedule")),
    ("Cost_Insurance", ("cost", "price", "fee", "insurance", "claim")),
    ("Medication_Information", ("medicine", "medication", "drug", "tablet", "panadol", "paracetamol", "dose", "dosage")),
    ("Vaccine_Information", ("vaccine", "vaccination", "immunization", "jab")),
    ("Test_Diagnostics", ("test", "diagnostic", "diagnosis", "scan", "x-ray", "blood test", "lab report")),
    ("Treatment_Procedure", ("treatment", "procedure", "surgery", "operation", "physiotherapy", "therapy")),
    ("Symptom_Information", ("symptom", "symptoms", "fever", "cough", "pain", "rash", "vomiting", "diarrhea")),
    ("Disease_Information", ("disease", "condition", "common cold", "dengue", "diabetes", "hypertension", "asthma", "covid")),
    ("General_Health_Education", ("health", "prevention", "nutrition", "exercise", "diet")),
)

ROUTER_SYSTEM_PROMPT = """You are a specialized healthcare routing assistant for Sri Lankan queries.
Your task is to classify and extract routing metadata from the user's query.

Strict Rules:
1. Never provide medical answers, diagnoses, or prescriptions.
2. Never provide dosage instructions or emergency action instructions.
3. You must output ONLY a valid JSON object. Do not include any conversational text outside the JSON.
4. The JSON object must contain exactly these keys: "language", "intent", "keywords", and "english_translation_or_summary".

Intent categories are:
"Emergency_Triage",
"Symptom_Information",
"Disease_Information",
"Facility_Locator",
"Provider_Locator",
"Appointment_Booking",
"Medication_Information",
"Vaccine_Information",
"Test_Diagnostics",
"Treatment_Procedure",
"Cost_Insurance",
"General_Health_Education",
"Non_Medical",
"Unclear"

Return JSON only."""

router_model = ChatOpenAI(
    api_key=ROUTER_API_KEY,
    base_url=ROUTER_BASE_URL,
    model=ROUTER_MODEL,
    temperature=ROUTER_TEMPERATURE,
    max_tokens=ROUTER_MAX_TOKENS,
)

logger.info(
    "Router model initialized | model=%s | base_url=%s | has_api_key=%s",
    ROUTER_MODEL,
    ROUTER_BASE_URL,
    bool(ROUTER_API_KEY),
)


def _latest_user_text(messages) -> str:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def _strip_json_fence(content: str) -> str:
    text = (content or "").strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def _loads_router_json(content: str) -> Dict[str, Any]:
    """Load the router JSON even when a local model wraps it in thinking text."""
    text = _strip_json_fence(content)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, json.JSONDecodeError):
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise json.JSONDecodeError("No JSON object found", text, 0)


def sanitize_router_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the allowed router fields and never include explanatory metadata."""
    keywords = data.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    if not isinstance(keywords, list):
        keywords = []

    intent = str(data.get("intent") or "Unclear").strip()
    intent = INTENT_ALIASES.get(intent, intent)

    return {
        "language": str(data.get("language") or "").strip(),
        "intent": intent,
        "keywords": [str(keyword).strip() for keyword in keywords if str(keyword).strip()],
        "english_translation_or_summary": str(data.get("english_translation_or_summary") or "").strip(),
    }


def parse_router_output(content: str) -> Tuple[Dict[str, Any], str | None]:
    """Parse and validate router JSON, ignoring all fields outside the router contract."""
    try:
        parsed = _loads_router_json(content)
    except (TypeError, json.JSONDecodeError) as exc:
        return {
            "language": "",
            "intent": "Unclear",
            "keywords": [],
            "english_translation_or_summary": "",
        }, f"invalid_json: {exc}"

    if not isinstance(parsed, dict):
        return {
            "language": "",
            "intent": "Unclear",
            "keywords": [],
            "english_translation_or_summary": "",
        }, "router_json_not_object"

    payload = sanitize_router_payload(parsed)
    if payload["intent"] not in ALLOWED_INTENTS:
        payload["intent"] = "Unclear"
        return payload, "unsupported_intent"

    return payload, None


def build_rag_query(user_message: str, router_payload: Dict[str, Any]) -> str:
    """Build a retrieval query from normalized router fields and the original user text."""
    keywords = router_payload.get("keywords") or []
    keyword_text = ", ".join(keywords)
    summary = router_payload.get("english_translation_or_summary") or ""
    intent = router_payload.get("intent") or "General_Health_Education"

    parts = [
        f"Intent: {intent}",
        f"Summary: {summary}" if summary else "",
        f"Keywords: {keyword_text}" if keyword_text else "",
        f"Original user query: {user_message}" if user_message else "",
    ]
    return "\n".join(part for part in parts if part).strip()


def _keyword_candidates(text: str, patterns: Tuple[str, ...]) -> List[str]:
    keywords = [pattern for pattern in patterns if pattern in text]
    words = re.findall(r"[A-Za-z][A-Za-z-]{3,}", text)
    for word in words:
        if word.lower() not in {"want", "details", "about", "know", "with", "from", "please"}:
            keywords.append(word)
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        normalized = keyword.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_keywords.append(keyword)
    return unique_keywords[:8]


def deterministic_router_fallback(user_message: str) -> Dict[str, Any]:
    """Classify obvious requests locally when the router endpoint is unavailable or malformed."""
    text = (user_message or "").strip()
    normalized = text.lower()

    if not normalized:
        return {
            "language": "",
            "intent": "Unclear",
            "keywords": [],
            "english_translation_or_summary": "",
        }

    if normalized in SOCIAL_ONLY_INDICATORS:
        return {
            "language": "",
            "intent": "Non_Medical",
            "keywords": [normalized],
            "english_translation_or_summary": text,
        }

    emergency_matches = tuple(indicator for indicator in EMERGENCY_INDICATORS if indicator in normalized)
    if emergency_matches:
        return {
            "language": "",
            "intent": "Emergency_Triage",
            "keywords": list(emergency_matches[:8]),
            "english_translation_or_summary": text,
        }

    for intent, patterns in DETERMINISTIC_INTENT_PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            return {
                "language": "",
                "intent": intent,
                "keywords": _keyword_candidates(normalized, patterns),
                "english_translation_or_summary": text,
            }

    return {
        "language": "",
        "intent": "Unclear",
        "keywords": [],
        "english_translation_or_summary": text,
    }


def _router_user_prompt(messages: List[Any]) -> str:
    latest_user_text = _latest_user_text(messages)
    conversation = get_buffer_string(messages=messages)
    return (
        "Classify the latest user query using the full conversation only as context.\n\n"
        f"<Conversation>\n{conversation}\n</Conversation>\n\n"
        f"Latest user query: {latest_user_text}"
    )


def _invoke_router(messages: List[Any]) -> Tuple[Dict[str, Any], int, str | None]:
    started = time.perf_counter()
    response = router_model.invoke([
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": _router_user_prompt(messages)},
    ])
    latency_ms = int((time.perf_counter() - started) * 1000)
    payload, parse_error = parse_router_output(response.content)
    return payload, latency_ms, parse_error


@traceable(name="intent_classifier_router")
def classify_intent(state: AgentState):
    """Classify the request and route to emergency, query classifier, or medical info."""
    messages = state.get("messages", [])
    latest_user_text = _latest_user_text(messages)
    logger.info("Intent classifier router processing request")

    try:
        payload, latency_ms, parse_error = _invoke_router(messages)
    except Exception as exc:
        logger.error("Router model invocation failed: %s", exc)
        payload = deterministic_router_fallback(latest_user_text)
        latency_ms = 0
        parse_error = None if payload.get("intent") != "Unclear" else "router_invocation_failed"

    if parse_error:
        fallback_payload = deterministic_router_fallback(latest_user_text)
        if fallback_payload.get("intent") != "Unclear":
            logger.warning("Router parse/validation issue '%s'; using deterministic fallback", parse_error)
            payload = fallback_payload
            parse_error = None

    intent = payload.get("intent") or "Unclear"
    base_update = {
        "active_agent": "intent_classifier",
        "route_language": payload.get("language"),
        "route_intent": intent,
        "route_keywords": payload.get("keywords") or [],
        "route_summary": payload.get("english_translation_or_summary"),
        "router_model": ROUTER_MODEL,
        "router_base_url": ROUTER_BASE_URL,
        "router_payload": payload,
        "router_latency_ms": latency_ms,
    }

    if parse_error:
        logger.warning("Router parse/validation issue: %s", parse_error)
        return {
            **base_update,
            "active_agent": "query_classifier",
            "rag_query": None,
        }

    if intent == "Emergency_Triage":
        return {
            **base_update,
            "messages": [AIMessage(content=EMERGENCY_CONTACT_RESPONSE)],
            "active_agent": "emergency_response",
            "rag_query": None,
        }

    if intent == "Non_Medical":
        return {
            **base_update,
            "messages": [AIMessage(content=NON_MEDICAL_RESPONSE)],
            "active_agent": "non_medical_response",
            "rag_query": None,
        }

    if intent == "Unclear":
        return {
            **base_update,
            "active_agent": "query_classifier",
            "rag_query": None,
        }

    return {
        **base_update,
        "active_agent": "medical_info",
        "rag_query": build_rag_query(latest_user_text, payload),
    }


intent_classifier_graph = StateGraph(AgentState, input_schema=AgentInputState)
intent_classifier_graph.add_node(classify_intent)
intent_classifier_graph.add_edge(START, "classify_intent")
intent_classifier_graph.add_edge("classify_intent", END)

graph = intent_classifier_graph.compile()


if __name__ == "__main__":
    output_path = Path("intent_classifier_graph.png")
    graph.get_graph().draw_mermaid_png(output_file_path=output_path)
    logger.info("Graph exported to %s", output_path.resolve())
