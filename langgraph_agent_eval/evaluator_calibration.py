"""
Evaluator drift detection: run your LLM judge against human-labeled anchors
and flag if agreement drops below the threshold.

Usage:
    python evaluator_calibration.py

Set CALIBRATION_THRESHOLD env var to override default of 0.85.
"""
from __future__ import annotations
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "llm_as_judge"))
from judge import StructuredJudge, calibration_test  # noqa: E402


def main():
    threshold = float(os.getenv("CALIBRATION_THRESHOLD", "0.85"))
    print(f"Running evaluator calibration test (threshold={threshold:.0%})...\n")
    judge = StructuredJudge(model="gpt-4o")
    result = calibration_test(judge, threshold=threshold)

    print(f"Calibration agreement: {result['agreement']:.0%}")
    print(f"Status: {'PASS \u2713' if result['passed'] else 'FAIL \u2014 evaluator drift detected \u2717'}")

    print("\nPer-example details:")
    for d in result["details"]:
        mark = "\u2713" if d["agreed"] else "\u2717"
        print(f"  {mark}  Q: {d['question'][:42]:<42} human={d['human_score']:.1f}  judge={d['judge_score']:.2f}")

    if not result["passed"]:
        print(f"\n  Action required: Judge agreement {result['agreement']:.0%} < {threshold:.0%}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
