"""Intent classifier agent for user clarification and intent extraction."""
from datetime import datetime
import os
from pathlib import Path
import sys

from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langsmith import traceable

# Path setup for imports
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

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    now = datetime.now()
    return f"{now.strftime('%a %b')} {now.day}, {now.year}"


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




# ===== WORKFLOW NODES =====
@traceable(name="clarify_with_user")
def clarify_with_user(state: AgentState):
    """Clarify user intent and emit a RAG query when ready."""
    
    logger.info("Intent classifier agent processing request")
    
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

        if response.need_clarification:
            fallbacks = {
                "si": "කරුණාකර ටිකක් වැඩි විස්තරයක් දිය හැකිද?",
                "ta": "தயவுசெய்து இன்னும் கொஞ்சம் விரிவாக சொல்ல முடியுமா?",
                "en": "Could you share a bit more detail?",
            }
            follow_up = response.follow_up_question or fallbacks.get(response_language, fallbacks["en"])
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


