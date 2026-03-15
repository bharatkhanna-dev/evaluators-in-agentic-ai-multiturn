"""
LangGraph agent with middleware patterns for harness engineering.

Demonstrates:
  - LoopDetectionMiddleware: prevent doom loops on repeated resource edits
  - PreCompletionChecklistMiddleware: force verification before exit
  - LocalContextMiddleware: inject environment context on startup
"""
from __future__ import annotations
import operator
from collections import defaultdict
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool


class LoopDetectionMiddleware:
    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self._counts: dict[str, int] = defaultdict(int)

    def on_tool_call(self, tool_name: str, args: dict) -> str | None:
        resource = args.get("path") or args.get("filename") or args.get("resource")
        if resource and tool_name in ("edit_file", "write_file", "update_resource"):
            self._counts[resource] += 1
            if self._counts[resource] >= self.threshold:
                return (
                    f"[LOOP DETECTION] You have modified '{resource}' "
                    f"{self._counts[resource]} times. Consider reconsidering your approach."
                )
        return None

    def reset(self) -> None:
        self._counts.clear()


class PreCompletionChecklistMiddleware:
    _CHECKLIST = (
        "Before completing this task, confirm:\n"
        "1. Did you address ALL parts of the request?\n"
        "2. Have you verified your solution against the requirements?\n"
        "3. Are edge cases handled?\n"
        "4. Is the output in the exact format requested?\n"
        "Only proceed if ALL answers are YES."
    )

    def should_inject(self, state: dict) -> bool:
        messages = state.get("messages", [])
        if not messages:
            return False
        last = messages[-1]
        content = getattr(last, "content", "").lower()
        signals = ["complete", "finished", "done", "here is", "i have completed"]
        return (
            not getattr(last, "tool_calls", None)
            and any(s in content for s in signals)
        )

    def build_injection(self) -> HumanMessage:
        return HumanMessage(content=self._CHECKLIST)


@tool
def edit_file(path: str, content: str) -> str:
    """Edit the content of a file at the given path."""
    return f"[File edited: {path}]"


@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path."""
    return f"[File content of {path}]"


@tool
def run_tests(test_path: str = ".") -> str:
    """Run the test suite and return the results."""
    return "[Tests: 5 passed, 0 failed]"


tools = [edit_file, read_file, run_tests]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    loop_warnings: list[str]
    verification_injected: bool


_llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)
_loop_mw = LoopDetectionMiddleware(threshold=3)
_checklist_mw = PreCompletionChecklistMiddleware()


def call_llm(state: AgentState) -> dict:
    messages = state["messages"]
    verification_injected = state.get("verification_injected", False)

    if (
        not verification_injected
        and len(messages) > 2
        and _checklist_mw.should_inject(state)
    ):
        messages = messages + [_checklist_mw.build_injection()]
        verification_injected = True

    response = _llm.invoke(messages)
    return {
        "messages": [response],
        "verification_injected": verification_injected,
        "loop_warnings": state.get("loop_warnings", []),
    }


def execute_tools(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    injected_warnings = []
    tool_node = ToolNode(tools)

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            warning = _loop_mw.on_tool_call(tc["name"], tc.get("args", {}))
            if warning:
                injected_warnings.append(warning)

    result = tool_node.invoke(state)
    new_messages = result.get("messages", [])

    for w in injected_warnings:
        new_messages.append(SystemMessage(content=w))

    return {
        "messages": new_messages,
        "loop_warnings": state.get("loop_warnings", []) + injected_warnings,
        "verification_injected": state.get("verification_injected", False),
    }


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_agent(interrupt_before_tools: bool = False):
    builder = StateGraph(AgentState)
    builder.add_node("llm", call_llm)
    builder.add_node("tools", execute_tools)
    builder.set_entry_point("llm")
    builder.add_conditional_edges("llm", should_continue)
    builder.add_edge("tools", "llm")
    memory = MemorySaver()
    kwargs = {}
    if interrupt_before_tools:
        kwargs["interrupt_before"] = ["tools"]
    return builder.compile(checkpointer=memory, **kwargs)


agent = build_agent(interrupt_before_tools=False)
test_agent = build_agent(interrupt_before_tools=True)
