import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import NotRequired, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    session_id: NotRequired[str]
    rag_query: NotRequired[str]

class SymptomData(TypedDict, total=False):
    chief_complaint: str
    duration: str
    severity: str
    age_group: str
    location: str
    other_symptoms: List[str]

def merge_symptoms(existing: SymptomData, updates: SymptomData) -> SymptomData:
    #keeps old data and only updates/adds what the sub-agent just found
    return {**(existing or {}), **(updates or {})}


def overwrite_active(_: Optional[str], new_value: Optional[str]) -> Optional[str]:
    return new_value


def overwrite_text(_: Optional[str], new_value: Optional[str]) -> Optional[str]:
    return new_value


def overwrite_flag(_: Optional[bool], new_value: Optional[bool]) -> Optional[bool]:
    return new_value


def overwrite_any(_: Any, new_value: Any) -> Any:
    return new_value

class AgentState(MessagesState):
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    active_agent: Annotated[Optional[str], overwrite_active] = "intent_classifier"
    new_message: Annotated[Optional[bool], overwrite_flag] = True
    rag_query: Annotated[Optional[str], overwrite_text]
    # Stable audit fields kept for downstream persistence and evaluation.
    last_active_agent: Annotated[Optional[str], overwrite_text] = None
    last_rag_query: Annotated[Optional[str], overwrite_text] = None
    turn_type: Annotated[Optional[str], overwrite_text] = None
    intent_classifier_turns: int = 0
    query_classifier_turns: int = 0
    route_language: Annotated[Optional[str], overwrite_text] = None
    route_intent: Annotated[Optional[str], overwrite_text] = None
    route_keywords: Annotated[Optional[List[str]], overwrite_any] = None
    route_summary: Annotated[Optional[str], overwrite_text] = None
    router_model: Annotated[Optional[str], overwrite_text] = None
    router_base_url: Annotated[Optional[str], overwrite_text] = None
    router_payload: Annotated[Optional[Dict[str, Any]], overwrite_any] = None
    router_latency_ms: int = 0
    rewrite_attempts: int = 0  # max 3
    critique_attempts: int = 0  # max 2
    critique_feedback: Annotated[Optional[str], overwrite_text] = None  # Feedback from critique
    grounding_context: Annotated[Optional[str], overwrite_text] = None
    grounding_sources: Annotated[Optional[List[Dict[str, Any]]], overwrite_any] = None

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    follow_up_question: Optional[str] = Field(
        description="Clarifying question to ask from user",
    )
    intent_summary: str = Field(
        description="Either 'CONVERSATIONAL:<response>' for social messages, or 2-3 sentences describing what the user needs from RAG",
    )
  

class gatheredSymptomInfo(BaseModel):
    """Schema for gathered symptom information from the user."""

    chief_complaint: Optional[str] = Field(
        default=None,
        description="The main symptom or complaint reported by the user.",
    )
    duration: Optional[str] = Field(
        default=None,
        description="The duration or onset of the symptoms.",
    )
    severity: Optional[str] = Field(
        default=None,
        description="The severity or location of the symptoms.",
    )
    age_group: Optional[str] = Field(
        default=None,
        description="The age group or any chronic conditions of the user.",
    )
    location: Optional[str] = Field(
        default=None,
        description="The location of the symptoms.",
    )
    other_symptoms: Optional[List[str]] = Field(
        default=None,
        description="Any other symptoms reported by the user.",
    )
