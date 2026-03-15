"""
Run multi-turn simulation scenarios and report results.

Usage:
    python run_eval.py
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from simulation import run_simulation, TurnEvaluator, SimulationResult

_llm = ChatOpenAI(model="gpt-4o", temperature=0)
_SYSTEM = "You are a helpful and friendly customer support assistant for a software product."

def customer_support_agent(user_message: str) -> str:
    response = _llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=user_message),
    ])
    return response.content.strip()


SCENARIOS = [
    {
        "id": "password_reset",
        "persona": "A non-technical office worker in their 50s who forgot their account password. Not very comfortable with technology. Slightly anxious but polite.",
        "task": "Get help resetting my account password so I can log back in and access my files.",
    },
    {
        "id": "billing_dispute",
        "persona": "A business owner who was charged twice on their credit card for the same subscription. Professional but increasingly frustrated after waiting a week for a resolution.",
        "task": "Get confirmation that the duplicate charge has been reversed and a refund is on its way.",
    },
    {
        "id": "api_integration_help",
        "persona": "A junior developer trying to integrate the product's API into a Python application. Has read the documentation but is stuck on authentication.",
        "task": "Find out the correct way to pass an API key for OAuth2 authentication and get a working example in Python.",
    },
    {
        "id": "adversarial_confused_user",
        "persona": "A user who is extremely confused about what the product does. They think this is a food delivery service and keep asking about their pizza order.",
        "task": "Find out if my pizza order from 30 minutes ago has been dispatched.",
    },
]


def push_to_langsmith(result: SimulationResult, project_name: str = "multi_turn_sim") -> None:
    try:
        from langsmith import Client
        import uuid
        client = Client()
        run_id = str(uuid.uuid4())
        client.create_run(
            name=f"simulation_{result.scenario_id}",
            run_type="chain",
            inputs={"scenario_id": result.scenario_id},
            outputs={"completed": result.task_completed, "total_turns": result.total_turns},
            id=run_id,
            project_name=project_name,
        )
        client.update_run(run_id, end_time=__import__("datetime").datetime.utcnow())
        client.create_feedback(run_id=run_id, key="task_completed", score=result.completion_score)
        client.create_feedback(run_id=run_id, key="mean_helpfulness", score=result.mean_helpfulness)
    except Exception as e:
        print(f"  [warn] Could not push to LangSmith: {e}")


def main() -> None:
    evaluator = TurnEvaluator()
    results: list[SimulationResult] = []

    print("Running multi-turn simulations...\n")

    for scenario in SCENARIOS:
        print(f"  ▶ {scenario['id']}")
        result = run_simulation(
            agent_callable=customer_support_agent,
            scenario_id=scenario["id"],
            persona=scenario["persona"],
            task=scenario["task"],
            max_turns=8,
            turn_evaluator=evaluator,
        )
        results.append(result)
        print(f"    {result.summary()}")
        if os.getenv("LANGCHAIN_API_KEY"):
            push_to_langsmith(result)

    completed = sum(1 for r in results if r.task_completed)
    mean_help = sum(r.mean_helpfulness for r in results) / len(results)
    mean_acc = sum(r.mean_accuracy for r in results) / len(results)

    print(f"\n── Summary ({completed}/{len(results)} tasks completed) ──")
    print(f"  Mean helpfulness: {mean_help:.2f}")
    print(f"  Mean accuracy:    {mean_acc:.2f}")
    print(f"  Completion rate:  {completed / len(results):.0%}")

    if (completed / len(results)) < 0.6:
        print(f"\n  FAIL: Completion rate below 60% threshold")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
