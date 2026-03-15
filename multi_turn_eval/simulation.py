"""
Multi-turn user simulation harness.

Provides:
  - SimulatedUser: LLM agent playing the user role in a conversation
  - TurnEvaluator: scores individual turns for helpfulness/accuracy/tone
  - SimulationResult: typed result container with aggregated metrics
  - run_simulation(): orchestrates the full loop
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Callable

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ── Prompts ─────────────────────────────────────────────────────────────────────────────────────
_USER_SYSTEM = """\
You are simulating a user with the following persona:
{persona}

Your goal in this conversation:
{task}

Rules:
1. Stay fully in character — respond as this user would, naturally.
2. Adapt your follow-ups based on what the AI assistant just said.
3. If your goal has been FULLY accomplished, respond with exactly: TASK_COMPLETE
4. If you are stuck, frustrated, or the agent is clearly failing, respond with: TASK_FAILED: <brief reason>
5. Do NOT reveal you are a simulation or AI.
6. Keep messages concise (1-3 sentences) unless the task requires more detail."""

_TURN_SCORE_PROMPT = """\
You are evaluating a single turn in a customer support conversation.

User's goal: {task}
Agent's latest response: {agent_response}
Conversation context: {context}

Rate the agent's response on three criteria (each 0.0–1.0):
- helpfulness: Did this response advance the user toward their goal?
- accuracy: Was the information correct and complete?
- tone: Was the response professional, empathetic, and appropriate?

Respond with ONLY valid JSON: {{"helpfulness": 0.0, "accuracy": 0.0, "tone": 0.0}}"""


# ── Data classes ──────────────────────────────────────────────────────────────────────────────────
@dataclass
class TurnRecord:
    turn_number: int
    user_message: str
    agent_response: str
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationResult:
    scenario_id: str
    task_completed: bool
    task_failed: bool
    failure_reason: str | None
    turns: list[TurnRecord]
    total_turns: int

    @property
    def mean_helpfulness(self) -> float:
        scores = [t.scores.get("helpfulness", 0.0) for t in self.turns if t.scores]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def mean_accuracy(self) -> float:
        scores = [t.scores.get("accuracy", 0.0) for t in self.turns if t.scores]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def completion_score(self) -> float:
        return 1.0 if self.task_completed else 0.0

    def summary(self) -> str:
        status = "COMPLETED" if self.task_completed else f"FAILED ({self.failure_reason})"
        return (
            f"[{self.scenario_id}] {status} | "
            f"turns={self.total_turns} | "
            f"helpfulness={self.mean_helpfulness:.2f} | "
            f"accuracy={self.mean_accuracy:.2f}"
        )


# ── Simulated user ──────────────────────────────────────────────────────────────────────────────────
class SimulatedUser:
    def __init__(self, persona: str, task: str, model: str = "gpt-4o"):
        self._llm = ChatOpenAI(model=model, temperature=0.6)
        self.persona = persona
        self.task = task
        self._history: list = []
        self._system = _USER_SYSTEM.format(persona=persona, task=task)
        self._last_signal: str = ""

    def generate_initial_message(self) -> str:
        response = self._llm.invoke([
            SystemMessage(content=self._system),
            HumanMessage(content="Start the conversation. Send your opening message to the assistant."),
        ])
        msg = response.content.strip()
        self._history.append(HumanMessage(content=msg))
        return msg

    def respond_to(self, agent_response: str) -> str | None:
        self._history.append(AIMessage(content=agent_response))
        response = self._llm.invoke([
            SystemMessage(content=self._system),
            *self._history,
        ])
        content = response.content.strip()
        self._last_signal = content

        if content == "TASK_COMPLETE" or content.startswith("TASK_FAILED:"):
            return None

        self._history.append(HumanMessage(content=content))
        return content

    @property
    def completed(self) -> bool:
        return self._last_signal == "TASK_COMPLETE"

    @property
    def failure_reason(self) -> str | None:
        if self._last_signal.startswith("TASK_FAILED:"):
            return self._last_signal[len("TASK_FAILED:"):].strip()
        return None


# ── Turn evaluator ──────────────────────────────────────────────────────────────────────────────────
class TurnEvaluator:
    def __init__(self, model: str = "gpt-4o"):
        self._llm = ChatOpenAI(model=model, temperature=0)

    def score(self, task: str, agent_response: str, context: str) -> dict[str, float]:
        prompt = _TURN_SCORE_PROMPT.format(
            task=task,
            agent_response=agent_response,
            context=context,
        )
        raw = self._llm.invoke([HumanMessage(content=prompt)]).content.strip()
        try:
            data = json.loads(raw)
            return {k: max(0.0, min(1.0, float(v))) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError, KeyError):
            return {"helpfulness": 0.5, "accuracy": 0.5, "tone": 0.5}


# ── Simulation runner ────────────────────────────────────────────────────────────────────────────────
def run_simulation(
    agent_callable: Callable[[str], str],
    scenario_id: str,
    persona: str,
    task: str,
    max_turns: int = 8,
    turn_evaluator: TurnEvaluator | None = None,
) -> SimulationResult:
    user = SimulatedUser(persona=persona, task=task)
    evaluator = turn_evaluator or TurnEvaluator()
    records: list[TurnRecord] = []
    context_log = f"Task: {task}"

    current_user_msg = user.generate_initial_message()

    for turn_num in range(1, max_turns + 1):
        agent_response = agent_callable(current_user_msg)
        context_log += f"\nTurn {turn_num} — agent: {agent_response[:120]}"

        scores = evaluator.score(task, agent_response, context_log)
        records.append(TurnRecord(
            turn_number=turn_num,
            user_message=current_user_msg,
            agent_response=agent_response,
            scores=scores,
        ))

        next_msg = user.respond_to(agent_response)
        if next_msg is None:
            return SimulationResult(
                scenario_id=scenario_id,
                task_completed=user.completed,
                task_failed=not user.completed,
                failure_reason=user.failure_reason,
                turns=records,
                total_turns=turn_num,
            )

        current_user_msg = next_msg
        context_log += f"\nTurn {turn_num} — user: {next_msg[:120]}"

    return SimulationResult(
        scenario_id=scenario_id,
        task_completed=False,
        task_failed=True,
        failure_reason=f"Max turns ({max_turns}) reached without task completion",
        turns=records,
        total_turns=max_turns,
    )
