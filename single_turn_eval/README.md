# Single-Turn Evaluator

Tests a Q&A chain using LangSmith's `evaluate()` API with both heuristic and LLM-as-judge evaluators.

## What it demonstrates

- Heuristic evaluators (format, number detection)
- LLM-as-judge with rubric injection
- `langsmith.evaluate()` API usage
- Handling missing/non-applicable reference answers

## Run

```bash
python single_turn_eval/eval.py
```

## Expected output

```
── Single-Turn Eval Results ──
  Non-empty rate:    100%
  Mean correctness:  0.83
```

## Prerequisites

- Dataset `agent_qa_v2` must exist in LangSmith (run `datasets/create_and_push.py` first)
