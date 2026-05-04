"""Linear triage -> medical information workflow with supervisor orchestration."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List
from typing_extensions import Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langsmith import traceable

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.src.graph.state import AgentState, AgentInputState
from agents.src.agent.intent_classifier import graph as intent_classifier_graph
from agents.src.agent.query_classifier import graph as query_classifier_graph
from agents.src.agent.medical_info import medical_info_graph
from agents.src.utils import setup_logger

# ===== LOGGING =====
logger = setup_logger("agent_supervisor")

# ===== DATABASE LOGGING =====
try:
    from database.agent_logging import AgentStateLogger
    DB_LOGGING_ENABLED = True
except ImportError:
    DB_LOGGING_ENABLED = False
    logger.warning("Database logging not available (database module not found)")


def _copy_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    return list(messages) if messages else []


def _new_subgraph_messages(state: AgentState, result_state: AgentState) -> List[BaseMessage]:
    """Return only messages appended by a subgraph invocation."""
    start_idx = len(state.get("messages", []))
    all_subgraph_msgs = result_state.get("messages", [])
    if len(all_subgraph_msgs) <= start_idx:
        return []
    return list(all_subgraph_msgs[start_idx:])


def _build_run_config(state: AgentState, run_name: str) -> Dict[str, Any]:
    """Create a LangSmith config block with consistent metadata."""
    session_id = state.get("session_id", "unknown-session")
    return {
        "run_name": f"{run_name}-{session_id}",
        "metadata": {
            "session_id": session_id,
            "active_agent": state.get("active_agent"),
            "intent_classifier_turns": state.get("intent_classifier_turns"),
            "has_rag_query": bool(state.get("rag_query")),
        },
    }


ROUTER_STATE_KEYS = (
    "route_language",
    "route_intent",
    "route_keywords",
    "route_summary",
    "router_model",
    "router_base_url",
    "router_payload",
    "router_latency_ms",
)


@traceable(name="intent_classifier_agent")
def intent_classifier_agent(state: AgentState):
    """Run the intent classifier router and decide the next agent."""
    logger.info("Invoking intent classifier router sub-graph")
    
    try:
        result_state = intent_classifier_graph.invoke(state, config=_build_run_config(state, "intent_classifier_graph"))
        updates = {}
        _sentinel = object()
        
        new_messages = _new_subgraph_messages(state, result_state)
        if new_messages:
            updates["messages"] = new_messages

        for key in ("rag_query", "active_agent", *ROUTER_STATE_KEYS):
            value = result_state.get(key, _sentinel)
            if value is not _sentinel:
                updates[key] = value
        updates["intent_classifier_turns"] = state.get("intent_classifier_turns", 0) + 1

        active_agent = result_state.get("active_agent")
        rag_query = result_state.get("rag_query")

        if active_agent == "query_classifier":
            logger.info("Intent router needs query clarification")
            updates["rag_query"] = None
            return Command(goto="query_classifier_agent", update=updates)

        if active_agent in {"emergency_response", "non_medical_response"}:
            logger.info("Intent router produced direct final response: %s", active_agent)
            updates["new_message"] = True
            updates["rag_query"] = None
            updates["last_active_agent"] = active_agent
            updates["last_rag_query"] = None
            updates["turn_type"] = "final"
            return Command(goto="finalize_response", update=updates)

        if rag_query:
            logger.info(f"Intent router complete, routing to medical info with query: {rag_query[:50]}...")
            updates["last_active_agent"] = "medical_info"
            updates["last_rag_query"] = rag_query
            updates["turn_type"] = "final"
            return Command(goto="medical_info_agent", update=updates)

        # Fallback: end the run without advancing if no RAG query was produced
        logger.warning("Router produced no RAG query or direct response; routing to query classifier")
        return Command(
            goto="query_classifier_agent",
            update={
                **updates,
                "active_agent": "query_classifier",
                "rag_query": None,
            },
        )
        
    except Exception as e:
        logger.error(f"Error in intent_classifier_agent: {e}")
        return Command(
            goto="query_classifier_agent",
            update={
                "new_message": True,
                "active_agent": "query_classifier",
                "last_active_agent": "intent_classifier",
                "last_rag_query": None,
                "turn_type": "clarification",
            },
        )


@traceable(name="query_classifier_agent")
def query_classifier_agent(state: AgentState):
    """Run the query classifier sub-graph to ask a follow-up question."""
    logger.info("Invoking query classifier sub-graph")

    try:
        result_state = query_classifier_graph.invoke(state, config=_build_run_config(state, "query_classifier_graph"))
        updates = {}
        _sentinel = object()

        new_messages = _new_subgraph_messages(state, result_state)
        if new_messages:
            updates["messages"] = new_messages

        for key in ("rag_query", "active_agent", *ROUTER_STATE_KEYS):
            value = result_state.get(key, _sentinel)
            if value is not _sentinel:
                updates[key] = value
        updates["query_classifier_turns"] = state.get("query_classifier_turns", 0) + 1
        updates["new_message"] = True
        updates["rag_query"] = None
        updates["last_active_agent"] = "query_classifier"
        updates["last_rag_query"] = None
        updates["turn_type"] = "clarification"
        return Command(goto="finalize_response", update=updates)
        
    except Exception as e:
        logger.error(f"Error in query_classifier_agent: {e}")
        return Command(
            goto="finalize_response",
            update={
                "new_message": True,
                "last_active_agent": "query_classifier",
                "last_rag_query": None,
                "turn_type": "clarification",
            },
        )


@traceable(name="medical_info_agent")
def medical_info_agent(state: AgentState) -> Command[Literal["finalize_response"]]:
    """Invoke the retrieval-augmented medical information agent."""
    logger.info("Invoking medical info sub-graph")
    
    try:
        rag_query = state.get("rag_query")
        med_state = state

        if rag_query:
            synthetic_message = HumanMessage(
                content=rag_query,
                name="intent_classifier_summary",
                additional_kwargs={"source": "intent_classifier_rag"}, 
            )
            med_state = dict(state)
            med_messages = _copy_messages(state.get("messages", []))
            med_messages.append(synthetic_message)
            med_state["messages"] = med_messages
        
        result_state = medical_info_graph.invoke(med_state, config=_build_run_config(state, "medical_info_graph"))
        updates = {}
        _sentinel = object()
        
        for key in ("symptom_json",):
            value = result_state.get(key, _sentinel)
            if value is not _sentinel:
                updates[key] = value

        # Critical: Only propagate NEW messages to avoid duplication
        # The subgraph was initialized with: existing history + 1 synthetic message
        # So new messages start after len(existing_history) + 1
        start_idx = len(state.get("messages", []))
        all_subgraph_msgs = result_state.get("messages", [])
        
        new_messages = []
        if len(all_subgraph_msgs) > start_idx:
            # Check if the message at start_idx is the synthetic one we added
            potential_new = all_subgraph_msgs[start_idx:]
            
            for msg in potential_new:
                # Filter out the synthetic trigger message if it appears
                if (isinstance(msg, HumanMessage) and 
                    msg.additional_kwargs.get("source") == "intent_classifier_rag"):
                    continue
                new_messages.append(msg)
                
        if new_messages:
            updates["messages"] = new_messages
        updates["last_active_agent"] = "medical_info"
        updates["last_rag_query"] = rag_query
        updates["turn_type"] = "final"
        updates["rag_query"] = None
        updates["active_agent"] = None
        
        logger.info("Medical info processing complete")
        return Command(goto="finalize_response", update=updates)
        
    except Exception as e:
        logger.error(f"Error in medical_info_agent: {e}")
        return Command(
            goto="finalize_response",
            update={
                "rag_query": None,
                "active_agent": None,
                "last_active_agent": "medical_info",
                "last_rag_query": state.get("rag_query"),
                "turn_type": "final",
            },
        )


@traceable(name="finalize_response")
def finalize_response(state: AgentState):
    """Deliver the final answer along with any safety reminder."""
    logger.info("Finalizing response")
    messages = _copy_messages(state.get("messages", []))
    return {
        "messages": messages,
        "active_agent": None,
        "rag_query": None,
        "last_active_agent": state.get("last_active_agent"),
        "last_rag_query": state.get("last_rag_query"),
        "turn_type": state.get("turn_type"),
        "new_message": True,
        **{key: state.get(key) for key in ROUTER_STATE_KEYS},
    }


# ===== BUILD WORKFLOW =====
workflow = StateGraph(AgentState, input_schema=AgentInputState)
workflow.add_node("intent_classifier_agent", intent_classifier_agent)
workflow.add_node("query_classifier_agent", query_classifier_agent)
workflow.add_node("medical_info_agent", medical_info_agent)
workflow.add_node("finalize_response", finalize_response)

workflow.add_edge(START, "intent_classifier_agent")
from langgraph.checkpoint.memory import MemorySaver

# Initialize MemorySaver (In-Memory Persistence)
# This is stable and works with async/sync without complex context management
checkpointer = MemorySaver()

agent_supervisor_graph = workflow.compile(
    checkpointer=checkpointer
)

# Guard graph visualization - only run when script is executed directly
# Guard graph visualization - only run when script is executed directly
if __name__ == "__main__":
    output_path = Path("system.png")
    agent_supervisor_graph.get_graph().draw_mermaid_png(output_file_path=output_path)
    logger.info(f"Graph exported to {output_path.resolve()}")

    # Initialize database logger if available
    db_logger = None
    if DB_LOGGING_ENABLED:
        try:
            db_logger = AgentStateLogger()
            logger.info("✅ Database logging enabled")
        except Exception as e:
            logger.warning(f"Could not initialize database logger: {e}")
            db_logger = None
    else:
        logger.info("ℹ️  Database logging disabled (not configured)")

    # Use a fixed session ID for the demo
    session_id = "demo-session"
    config = {"configurable": {"thread_id": session_id}}
    
    logger.info(f"Starting Session: {session_id}")
    logger.info("Intent Classifier/Medical Info workflow ready. Type 'quit' to exit.")
    
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            # End conversation in database
            if db_logger:
                db_logger.end_conversation(session_id)
                logger.info("📊 Conversation saved to database")
            break
        
        graph_input = {
            "messages": [HumanMessage(content=user_input)],
            "session_id": session_id  # Add session_id to state
        }

        # Use invoke for simple synchronous execution in CLI
        result = agent_supervisor_graph.invoke(graph_input, config=config)
        
        snapshot = agent_supervisor_graph.get_state(config)
        messages = snapshot.values.get("messages", [])
        
        reply_message = None
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                continue
            reply_message = message
            break

        if reply_message is None:
            logger.info("Agent: (no response)")
        else:
            logger.info(f"Agent: {reply_message.content}")
            
            # Log interaction to database
            if db_logger:
                try:
                    db_logger.log_interaction(
                        state=result,
                        session_id=session_id,
                        category="general",  # or auto-detect from content
                        language="en"
                    )
                    logger.info("💾 Interaction logged to database")
                except Exception as e:
                    logger.error(f"Failed to log to database: {e}")
