"""
Trajectory evaluators for the LangGraph agent.

Demonstrates:
  - Single-step eval: interrupt before tools, inspect proposed call
  - Full-trajectory eval: check that required tools appeared in execution
  - Tool argument correctness checking

Run via pytest:
    pytest trajectory_eval/eval.py -v
"""
from __future__ import annotations
import uuid
import pytest
import langsmith
import langsmith.testing
from langchain_core.messages import HumanMessage

from agent import agent, test_agent, AgentState


# ── Helpers ───────────────────────────────────────────────────────────────────────────────────
def _make_config() -> dict:
    """Generate a unique thread_id config for each test run."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _get_all_tool_calls(state: AgentState) -> list[dict]:
    """Extract all tool call records from the full message history."""
    calls = []
    for msg in state["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            calls.extend(msg.tool_calls)
    return calls


# ── Single-step trajectory tests ────────────────────────────────────────────────────────────────────
class TestSingleStepTrajectory:
    """
    These tests run the agent for ONE step (interrupt_before=["tools"])
    and assert that the proposed tool call is correct.
    No tool execution occurs — cheap, fast, and side-effect-free.
    """

    @pytest.mark.langsmith
    def test_proposes_search_for_current_events(self):
        """For a current-events question the agent should propose search_web."""
        question = "What happened at the AI Safety Summit last month?"
        langsmith.testing.log_inputs({"question": question})

        config = _make_config()
        state = test_agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )

        last_msg = state["messages"][-1]
        assert getattr(last_msg, "tool_calls", None), (
            "Agent did not propose any tool call for a current-events question"
        )
        proposed_tool = last_msg.tool_calls[0]["name"]
        assert proposed_tool == "search_web", (
            f"Expected 'search_web', got '{proposed_tool}'"
        )
        langsmith.testing.log_outputs({"proposed_tool": proposed_tool})
        langsmith.testing.log_feedback(key="correct_tool", score=1.0)

    @pytest.mark.langsmith
    def test_proposes_db_lookup_for_account_query(self):
        """For internal account data the agent should use lookup_database."""
        question = "Look up the account details for user ID 1042"
        langsmith.testing.log_inputs({"question": question})

        config = _make_config()
        state = test_agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )

        last_msg = state["messages"][-1]
        assert getattr(last_msg, "tool_calls", None), "No tool call proposed"

        proposed_tool = last_msg.tool_calls[0]["name"]
        assert proposed_tool == "lookup_database", (
            f"Expected 'lookup_database', got '{proposed_tool}'"
        )

        args_str = str(last_msg.tool_calls[0].get("args", {})).lower()
        assert "1042" in args_str, f"User ID '1042' missing from tool arguments: {args_str}"

        langsmith.testing.log_outputs({"proposed_tool": proposed_tool, "args": args_str})
        langsmith.testing.log_feedback(key="correct_tool", score=1.0)
        langsmith.testing.log_feedback(key="correct_args", score=1.0)


# ── Full-trajectory tests ──────────────────────────────────────────────────────────────────────────────────
class TestFullTrajectory:
    """
    These tests run the agent to completion and check the full trajectory.
    More expensive than single-step tests; run on PRs and nightly.
    """

    @pytest.mark.langsmith
    def test_search_appears_in_trajectory(self):
        """search_web should appear somewhere in the full execution trajectory."""
        question = "What is the latest version of Python as of today?"
        langsmith.testing.log_inputs({"question": question})

        config = _make_config()
        state = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )

        all_calls = _get_all_tool_calls(state)
        called_names = {tc["name"] for tc in all_calls}

        langsmith.testing.log_outputs({
            "trajectory": list(called_names),
            "final_answer": state["messages"][-1].content,
        })

        assert "search_web" in called_names, (
            f"Expected search_web in trajectory. Got: {called_names}"
        )
        langsmith.testing.log_feedback(key="search_in_trajectory", score=1.0)

    @pytest.mark.langsmith
    def test_no_notification_sent_without_explicit_request(self):
        """Agent should NOT send notifications unless explicitly asked to."""
        question = "What year was Python created?"
        langsmith.testing.log_inputs({"question": question})

        config = _make_config()
        state = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )

        all_calls = _get_all_tool_calls(state)
        called_names = [tc["name"] for tc in all_calls]

        langsmith.testing.log_outputs({"trajectory": called_names})

        assert "send_notification" not in called_names, (
            f"Agent sent a notification without being asked. Trajectory: {called_names}"
        )
        langsmith.testing.log_feedback(key="no_spurious_notifications", score=1.0)
