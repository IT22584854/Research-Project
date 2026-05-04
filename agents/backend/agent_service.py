"""Agent service — pure Python bridge to the LangGraph supervisor graph."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so agent imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # … / research-project-y4-data-science
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage

# ---------------------------------------------------------------------------
# Lazy-load the compiled graph (heavy import — only done once)
# ---------------------------------------------------------------------------
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        from agents.src.agent.system import agent_supervisor_graph
        _graph = agent_supervisor_graph
    return _graph


# ---------------------------------------------------------------------------
# Source extraction helper (ported from streamlit_app/utils/agent_runner.py)
# ---------------------------------------------------------------------------

def _normalize_sources(raw_sources: List[Dict]) -> List[Dict]:
    normalized = []
    for idx, raw in enumerate(raw_sources, start=1):
        label = raw.get("source_reference") or raw.get("source") or "Unknown Source"
        entry: Dict[str, Any] = {"id": raw.get("id", idx), "source": label}
        url = raw.get("url")
        if not url and isinstance(label, str) and label.startswith(("http://", "https://")):
            url = label
        if url:
            entry["url"] = url
        if raw.get("chunk_index") is not None:
            entry["chunk_index"] = raw["chunk_index"]
        if raw.get("excerpt"):
            entry["excerpt"] = raw["excerpt"]
        normalized.append(entry)
    return normalized


def extract_sources(messages: list) -> List[Dict]:
    """Extract citation metadata from agent messages."""
    # Prefer metadata embedded on the assistant response
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            metadata_sources = getattr(msg, "additional_kwargs", {}).get("sources")
            if metadata_sources:
                return _normalize_sources(metadata_sources)

    sources: List[Dict] = []
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "tool":
            content = getattr(msg, "content", str(msg))
            try:
                data = json.loads(content)
                if "documents" in data:
                    parsed = [
                        {
                            "id": len(sources) + i + 1,
                            "source_reference": doc.get("source_reference", "Unknown Source"),
                            "chunk_index": doc.get("chunk_index"),
                            "excerpt": (doc.get("content", "") or "")[:160].strip(),
                        }
                        for i, doc in enumerate(data["documents"])
                    ]
                    sources.extend(parsed)
            except (json.JSONDecodeError, KeyError):
                if content and len(content) > 50:
                    sources.append({"id": len(sources) + 1, "source": "Medical Knowledge Base"})

    return _normalize_sources(sources)


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

def run_agent(user_message: str, session_id: str) -> Tuple[str, List[Dict]]:
    """
    Invoke the LangGraph agent and return (response_text, sources).

    Args:
        user_message: The user's input text.
        session_id:   Conversation thread identifier (used as LangGraph thread_id).

    Returns:
        (response_text, sources) tuple.
    """
    graph = get_graph()

    graph_input = {"messages": [HumanMessage(content=user_message)], "session_id": session_id}
    config = {"configurable": {"thread_id": session_id}}

    graph.invoke(graph_input, config=config)

    snapshot = graph.get_state(config)
    messages = snapshot.values.get("messages", [])

    sources = extract_sources(messages)

    response_text = ""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            if message.content and not message.tool_calls:
                response_text = message.content
                break
            elif message.content:
                response_text = message.content
                break

    if not response_text:
        response_text = (
            "I apologize, but I couldn't generate a response. "
            "Please try rephrasing your question."
        )

    return response_text, sources
