<<<<<<< HEAD
"""Intent classifier router for healthcare request routing."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
=======
"""Intent classifier agent for user clarification and intent extraction."""
from datetime import datetime
import os
from pathlib import Path
import sys
>>>>>>> 0f118772db9d54090dd8a2af9a3c92da29899951

from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langsmith import traceable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.src.graph.state import ClarifyWithUser, AgentState, AgentInputState
from agents.src.prompts.triage_prompt import intent_classifier_prompt
from langchain_openai import ChatOpenAI
from agents.src.config import (
    LLM_MODEL, LLM_TEMPERATURE,
    LLM_PROVIDER,
    OPENAI_BASE_URL,
    CUSTOM_LLM_API_KEY,
    CUSTOM_LLM_BASE_URL,
    CUSTOM_LLM_MODEL,
    CUSTOM_LLM_MAX_TOKENS,
    CUSTOM_LLM_TEMPERATURE,
)
from agents.src.utils import (
    setup_logger,
    sanitize_input,
    retry_on_error,
    create_error_response,
    detect_user_language,
    get_language_instruction,
)

# ===== LOGGING =====
logger = setup_logger("intent_classifier_agent")

# ===== UTILITY FUNCTIONS ===== 

<<<<<<< HEAD
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

EMERGENCY_CONTACT_RESPONSE = """Emergency Contact

Ambulance Service: 1990
Children and Women Welfare Hotline:1929
General Government Helpline : 1919"""

NON_MEDICAL_RESPONSE = (
    "I can help with Sri Lankan medical and public-health questions. "
    "Please ask a health-related question so I can route it safely."
)

ROUTER_SYSTEM_PROMPT = """You are a specialized healthcare routing assistant for Sri Lankan queries.
Your task is to classify and extract routing metadata from the user's query.

Strict Rules:
1. Never provide medical answers, diagnoses, or prescriptions.
2. Never provide dosage instructions or emergency action instructions.
3. You must output ONLY a valid JSON object. Do not include any conversational text outside the JSON.
4. The JSON object must contain exactly these keys: "language", "intent", "keywords", and "english_translation_or_summary".
5. Do not include rationale or any other key.

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
=======
def get_today_str() -> str:
    """Get current date in a human-readable format."""
    now = datetime.now()
    return f"{now.strftime('%a %b')} {now.day}, {now.year}"
>>>>>>> 0f118772db9d54090dd8a2af9a3c92da29899951


def get_latest_user_text(messages) -> str:
    """Return the most recent user-authored message for language detection."""
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            return message.content
    return ""

# ===== CONFIGURATION =====
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

<<<<<<< HEAD

def _loads_router_json(content: str) -> Dict[str, Any]:
    """Load router JSON even when a local model wraps it in thinking text."""
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
=======
if LLM_PROVIDER == "custom_openai_compatible":
    model = ChatOpenAI(
        api_key=CUSTOM_LLM_API_KEY,
        base_url=CUSTOM_LLM_BASE_URL,
        model=CUSTOM_LLM_MODEL,
        temperature=CUSTOM_LLM_TEMPERATURE,
        max_tokens=CUSTOM_LLM_MAX_TOKENS,
    )
else:
    openai_kwargs = {
        "api_key": openai_api_key,
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
>>>>>>> 0f118772db9d54090dd8a2af9a3c92da29899951
    }
    if OPENAI_BASE_URL:
        openai_kwargs["base_url"] = OPENAI_BASE_URL
    model = ChatOpenAI(**openai_kwargs)

logger.info(
    "Intent model initialized | provider=%s | model=%s | has_api_key=%s",
    LLM_PROVIDER,
    (CUSTOM_LLM_MODEL if LLM_PROVIDER == "custom_openai_compatible" else LLM_MODEL),
    bool(CUSTOM_LLM_API_KEY if LLM_PROVIDER == "custom_openai_compatible" else openai_api_key),
)


<<<<<<< HEAD
def parse_router_output(content: str) -> Tuple[Dict[str, Any], str | None]:
=======


# ===== WORKFLOW NODES =====
@traceable(name="clarify_with_user")
def clarify_with_user(state: AgentState):
    """Clarify user intent and emit a RAG query when ready."""
    
    logger.info("Intent classifier agent processing request")
    
>>>>>>> 0f118772db9d54090dd8a2af9a3c92da29899951
    try:
        # Sanitize the latest user message
        messages = state.get("messages", [])
        latest_user_text = get_latest_user_text(messages)
        response_language = detect_user_language(latest_user_text)
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, HumanMessage):
                sanitized_content = sanitize_input(last_msg.content)
                logger.debug(f"Sanitized input length: {len(sanitized_content)}")
        
        structured_output_model = model.with_structured_output(ClarifyWithUser)
        
        @retry_on_error(logger=logger)
        def invoke_model():
            return structured_output_model.invoke([
                HumanMessage(content=intent_classifier_prompt.format(
                    messages=get_buffer_string(messages=state["messages"]),
                    date=get_today_str(),
                    response_language_instruction=get_language_instruction(response_language),
                ))
            ])
         
        response = invoke_model()
        
        logger.info(f"Need clarification: {response.need_clarification}")
        logger.debug(f"Follow up: {response.follow_up_question}")
        logger.debug(f"RAG query: {response.intent_summary}")

<<<<<<< HEAD
    payload = sanitize_router_payload(parsed)
    if payload["intent"] not in ALLOWED_INTENTS:
        payload["intent"] = "Unclear"
        return payload, "unsupported_intent"

    return payload, None


def build_rag_query(user_message: str, router_payload: Dict[str, Any]) -> str:
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
    text = (user_message or "").strip()
    normalized = text.lower()

    if not normalized:
        return {"language": "", "intent": "Unclear", "keywords": [], "english_translation_or_summary": ""}

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
=======
        if response.need_clarification:
            fallbacks = {
                "si": "කරුණාකර ටිකක් වැඩි විස්තරයක් දිය හැකිද?",
                "ta": "தயவுசெய்து இன்னும் கொஞ்சம் விரிவாக சொல்ல முடியுமா?",
                "en": "Could you share a bit more detail?",
            }
            follow_up = response.follow_up_question or fallbacks.get(response_language, fallbacks["en"])
>>>>>>> 0f118772db9d54090dd8a2af9a3c92da29899951
            return {
                "messages": [AIMessage(content=follow_up)],
                "active_agent": "intent_classifier",
                "rag_query": None,
            }

        # Check if this is a conversational response (no medical query needed)
        intent_summary = response.intent_summary or ""
        if intent_summary.startswith("CONVERSATIONAL:"):
            conversational_response = intent_summary.replace("CONVERSATIONAL:", "").strip()
            logger.info("Conversational message detected, responding directly")
            return {
                "messages": [AIMessage(content=conversational_response)],
                "active_agent": "intent_classifier",
                "rag_query": None,
            }

<<<<<<< HEAD

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
        return {**base_update, "active_agent": "query_classifier", "rag_query": None}

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
        return {**base_update, "active_agent": "query_classifier", "rag_query": None}

    return {
        **base_update,
        "active_agent": "medical_info",
        "rag_query": build_rag_query(latest_user_text, payload),
    }

=======
        # Generate RAG query for medical information
        rag_query_fallbacks = {
            "si": "පරිශීලක අවශ්‍යතාව පැහැදිලි නැත; කරුණාකර ඔබගේ ගැටලුව නැවත පැහැදිලි කරන්න.",
            "ta": "பயனர் தேவையை தெளிவாகப் புரிந்துகொள்ள முடியவில்லை; தயவுசெய்து உங்கள் கவலையை மீண்டும் தெளிவாகச் சொல்லுங்கள்.",
            "en": "User intent unclear; please restate the concern.",
        }
        rag_query = intent_summary or rag_query_fallbacks.get(response_language, rag_query_fallbacks["en"])
        logger.info(f"RAG query generated: {rag_query}")
        return {
            "rag_query": rag_query,
            "active_agent": "medical_info"
        }
        
    except Exception as e:
        logger.error(f"Error in intent classifier agent: {e}")
        return {
            "messages": [AIMessage(content=create_error_response("llm", language=response_language if 'response_language' in locals() else "en"))],
            "active_agent": "intent_classifier",
            "rag_query": None,
        }

# ===== GRAPH CONSTRUCTION =====
>>>>>>> 0f118772db9d54090dd8a2af9a3c92da29899951

intent_classifier_graph = StateGraph(AgentState, input_schema=AgentInputState)

intent_classifier_graph.add_node(clarify_with_user)
intent_classifier_graph.add_edge(START, "clarify_with_user")
intent_classifier_graph.add_edge("clarify_with_user", END)

graph = intent_classifier_graph.compile()

if __name__ == "__main__":
    # Guard graph visualization
    output_path = Path("intent_classifier_graph.png")
    graph.get_graph().draw_mermaid_png(output_file_path=output_path)
    logger.info(f"Graph exported to {output_path.resolve()}")

    logger.info("Starting Intent Classifier Agent (type 'quit' to exit)...")
    messages = []
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        # Sanitize user input
        user_input = sanitize_input(user_input)
        
        messages.append(HumanMessage(content=user_input))
        state = {"messages": messages}
        
        # Run the graph
        result = graph.invoke(state)
        
        # Append graph outputs to the transcript
        new_messages = result.get("messages", [])
        if isinstance(new_messages, list):
            messages.extend(new_messages)

        # Print the last message from the agent
        last_message = messages[-1]
        if isinstance(last_message, AIMessage):
            logger.info(f"Agent: {last_message.content}")

        # Check if the RAG query is ready
        if result.get("rag_query"):
            logger.info("RAG Query Ready:")
            logger.info(result["rag_query"])
            break


