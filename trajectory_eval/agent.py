"""
LangGraph agent used for trajectory evaluation.

Provides two compiled graph variants:
  - agent:       Full execution (for end-to-end and trajectory evals)
  - test_agent:  Stops before tools (for single-step inspection)
"""
from __future__ import annotations
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool


# ── Tools ─────────────────────────────────────────────────────────────────────────────────────
@tool
def search_web(query: str) -> str:
    """Search the web for up-to-date information about a topic or event."""
    return f"[Web search results for: '{query}']"


@tool
def lookup_database(table: str, filters: dict) -> str:
    """Query an internal database table with optional key/value filters."""
    return f"[Database results from table='{table}' filters={filters}]"


@tool
def send_notification(user_id: str, message: str, channel: str = "email") -> str:
    """Send a notification to a user via the specified channel."""
    return f"[Notification sent to user_id='{user_id}' via {channel}]"


tools = [search_web, lookup_database, send_notification]

# ── State ─────────────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# ── Nodes ──────────────────────────────────────────────────────────────────────────────────────
_llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)


def call_llm(state: AgentState) -> AgentState:
    response = _llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route to tools if tool calls are pending, otherwise end."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


_tool_node = ToolNode(tools)

# ── Graph ──────────────────────────────────────────────────────────────────────────────────────
_builder = StateGraph(AgentState)
_builder.add_node("llm", call_llm)
_builder.add_node("tools", _tool_node)
_builder.set_entry_point("llm")
_builder.add_conditional_edges("llm", should_continue)
_builder.add_edge("tools", "llm")

_memory = MemorySaver()

# Full agent: runs to completion
agent = _builder.compile(checkpointer=_memory)

# Test agent: stops before tool execution for trajectory inspection
test_agent = _builder.compile(
    checkpointer=_memory,
    interrupt_before=["tools"],
)
