from typing import Literal

from langchain.agents import AgentState, create_agent
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import NotRequired


# 1. Define state with active_agent tracker
class MultiAgentState(AgentState):
    active_agent: NotRequired[str]

# 2. Create handoff tool for triage agent
@tool
def handoff_to_triage_agent(
    runtime: ToolRuntime,
) -> Command:
    """Transer to the triage agent."""
    last_ai_message = next(
        msg
        for msg in reversed(runtime.agent_state["messages"])
        if isinstance(msg, AIMessage)
    )
    transfer_message = ToolMessage(
        content="Transferring to triage agent.",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="triage_agent",
        update={
            "active_agent": "triage_agent",
            "messages": [last_ai_message, transfer_message]
        },
        graph=Command.PARENT, 
    )



# 2. Create handoff tool for medical_information agent
@tool
def handoff_to_medical_information_agent(
    runtime: ToolRuntime,
) -> Command:
    """Transer to the medical_information agent."""
    last_ai_message = next(
        msg
        for msg in reversed(runtime.agent_state["messages"])
        if isinstance(msg, AIMessage)
    )
    transfer_message = ToolMessage(
        content="Transferring to medical_information agent.",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="medical_information_agent",
        update={
            "active_agent": "medical_information_agent",
            "messages": [last_ai_message, transfer_message]
        },
        graph=Command.PARENT, 
    )


# 3. Create handoff tool for validation agent
@tool
def handoff_to_validation_agent(
    runtime: ToolRuntime,
) -> Command:
    """Transer to the validation agent."""
    last_ai_message = next(
        msg
        for msg in reversed(runtime.agent_state["messages"])
        if isinstance(msg, AIMessage)
    )
    transfer_message = ToolMessage(
        content="Transferring to validation agent.",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="validation",
        update={
            "active_agent": "validation agent",
            "messages": [last_ai_message, transfer_message]
        },
        graph=Command.PARENT, 
    )



# 4. Create handoff tool for supervisor agent
@tool
def handoff_to_supervisor_agent(
    runtime: ToolRuntime,
) -> Command:
    """Transer to the supervisor agent."""
    last_ai_message = next(
        msg
        for msg in reversed(runtime.agent_state["messages"])
        if isinstance(msg, AIMessage)
    )
    transfer_message = ToolMessage(
        content="Transferring to supervisor agent.",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="supervisor",
        update={
            "active_agent": "supervisor agent",
            "messages": [last_ai_message, transfer_message]
        },
        graph=Command.PARENT, 
    )