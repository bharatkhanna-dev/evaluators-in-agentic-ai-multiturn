"""
Single-turn evaluator example.

Tests a simple Q&A chain against a LangSmith dataset using:
  - A heuristic evaluator (non-empty, format check)
  - An LLM-as-judge evaluator (factual correctness with rubric)

Run:
    python eval.py

Requires:
    LANGCHAIN_API_KEY, OPENAI_API_KEY, LANGCHAIN_TRACING_V2=true
    A dataset named 'agent_qa_v2' in LangSmith (run datasets/create_and_push.py first)
"""
from __future__ import annotations
import os
import re
from dotenv import load_dotenv

load_dotenv()

from langsmith import evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Application under test ───────────────────────────────────────────────────────────────────────
_llm = ChatOpenAI(model="gpt-4o", temperature=0)
_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise, factual assistant. Answer in one sentence."),
    ("human", "{question}"),
])
_chain = _prompt | _llm | StrOutputParser()


def target(inputs: dict) -> dict:
    """The function under evaluation. Must accept dict, return dict."""
    q = inputs.get("question", "").strip()
    if not q:
        return {"answer": "Please provide a question."}
    return {"answer": _chain.invoke({"question": q})}


# ── Heuristic evaluators ────────────────────────────────────────────────────────────────────────────────
def is_non_empty(run, example):
    """Fail if the agent returned an empty response."""
    answer = run.outputs.get("answer", "")
    return {"key": "non_empty", "score": int(len(answer.strip()) > 0)}


def contains_number_when_expected(run, example):
    """If the reference answer is a year or number, check the output contains one."""
    reference = example.outputs.get("answer", "")
    answer = run.outputs.get("answer", "")
    if re.fullmatch(r"\d{3,4}", reference.strip()):
        has_number = bool(re.search(reference.strip(), answer))
        return {"key": "number_present", "score": int(has_number)}
    return {"key": "number_present", "score": 1}


# ── LLM-as-judge evaluator ──────────────────────────────────────────────────────────────────────────────
_judge = ChatOpenAI(model="gpt-4o", temperature=0)

_RUBRIC = """\
You are a rigorous evaluator assessing factual correctness.

Question: {question}
Reference Answer: {reference}
Actual Answer: {answer}

Scoring rubric:
  1.0 — Completely correct per the reference
  0.7 — Mostly correct; minor error or omission
  0.3 — Partially correct; missing key facts
  0.0 — Incorrect, irrelevant, or harmful

Rules:
- Do not reward length over accuracy
- A correct fact stated differently still scores 1.0
- Respond with ONLY a decimal number between 0.0 and 1.0"""

_judge_prompt = ChatPromptTemplate.from_messages([("human", _RUBRIC)])


def llm_correctness(run, example):
    """LLM-as-judge evaluator scoring factual correctness against a reference."""
    question = example.inputs.get("question", "")
    reference = example.outputs.get("answer", "")
    answer = run.outputs.get("answer", "")

    if not reference or reference in ("requires_search",):
        # No reference available — skip LLM judge
        return {"key": "correctness", "score": None}

    response = _judge.invoke(
        _judge_prompt.format_messages(
            question=question,
            reference=reference,
            answer=answer,
        )
    )
    try:
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))
    except ValueError:
        score = 0.0

    return {"key": "correctness", "score": score}


# ── Main ──────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = evaluate(
        target,
        data="agent_qa_v2",
        evaluators=[is_non_empty, contains_number_when_expected, llm_correctness],
        experiment_prefix="single_turn_eval",
        num_repetitions=1,
        max_concurrency=4,
    )

    df = results.to_pandas()
    print("\n── Single-Turn Eval Results ──")
    print(f"  Non-empty rate:    {df['feedback.non_empty'].mean():.0%}")
    if "feedback.correctness" in df.columns:
        valid_scores = df["feedback.correctness"].dropna()
        if len(valid_scores) > 0:
            print(f"  Mean correctness:  {valid_scores.mean():.2f}")
    print(f"\nView full results in LangSmith ↗")
