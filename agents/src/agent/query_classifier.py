"""Query classifier agent for one-turn follow-up questions."""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

# Path setup for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.src.config import (
    CUSTOM_LLM_API_KEY,
    CUSTOM_LLM_BASE_URL,
    CUSTOM_LLM_MAX_TOKENS,
    CUSTOM_LLM_MODEL,
    CUSTOM_LLM_TEMPERATURE,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OPENAI_BASE_URL,
)
from agents.src.graph.state import AgentInputState, AgentState, ClarifyWithUser
from agents.src.prompts.triage_prompt import query_classifier_prompt
from agents.src.utils import (
    create_error_response,
    detect_user_language,
    get_language_instruction,
    retry_on_error,
    sanitize_input,
    setup_logger,
)

logger = setup_logger("query_classifier_agent")


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
    "Query classifier model initialized | provider=%s | model=%s | has_api_key=%s",
    LLM_PROVIDER,
    (CUSTOM_LLM_MODEL if LLM_PROVIDER == "custom_openai_compatible" else LLM_MODEL),
    bool(CUSTOM_LLM_API_KEY if LLM_PROVIDER == "custom_openai_compatible" else openai_api_key),
)


@traceable(name="query_classifier_follow_up")
def query_classifier_follow_up(state: AgentState):
    """Ask one concise follow-up question when routing is unclear."""
    logger.info("Query classifier generating follow-up question")

    try:
        messages = state.get("messages", [])
        latest_user_text = get_latest_user_text(messages)
        response_language = detect_user_language(latest_user_text)
        if messages and isinstance(messages[-1], HumanMessage):
            sanitize_input(messages[-1].content)

        structured_output_model = model.with_structured_output(ClarifyWithUser)

        @retry_on_error(logger=logger)
        def invoke_model():
            return structured_output_model.invoke([
                HumanMessage(content=query_classifier_prompt.format(
                    messages=get_buffer_string(messages=state["messages"]),
                    date=get_today_str(),
                    response_language_instruction=get_language_instruction(response_language),
                ))
            ])

        response = invoke_model()
        fallbacks = {
            "si": "කරුණාකර ඔබගේ සෞඛ්‍ය ගැටලුව හෝ අවශ්‍ය තොරතුර තව ටිකක් පැහැදිලි කරන්න.",
            "ta": "தயவுசெய்து உங்கள் உடல்நலக் கேள்வி அல்லது தேவையை இன்னும் கொஞ்சம் தெளிவாகச் சொல்லுங்கள்.",
            "en": "Could you clarify what health information you need?",
        }
        follow_up = response.follow_up_question or fallbacks.get(response_language, fallbacks["en"])
        return {
            "messages": [AIMessage(content=follow_up)],
            "active_agent": "query_classifier",
            "rag_query": None,
        }
    except Exception as e:
        logger.error("Error in query classifier agent: %s", e)
        response_language = detect_user_language(get_latest_user_text(state.get("messages", [])))
        return {
            "messages": [AIMessage(content=create_error_response("llm", language=response_language))],
            "active_agent": "query_classifier",
            "rag_query": None,
        }


query_classifier_graph = StateGraph(AgentState, input_schema=AgentInputState)
query_classifier_graph.add_node(query_classifier_follow_up)
query_classifier_graph.add_edge(START, "query_classifier_follow_up")
query_classifier_graph.add_edge("query_classifier_follow_up", END)

graph = query_classifier_graph.compile()


if __name__ == "__main__":
    output_path = Path("query_classifier_graph.png")
    graph.get_graph().draw_mermaid_png(output_file_path=output_path)
    logger.info("Graph exported to %s", output_path.resolve())
