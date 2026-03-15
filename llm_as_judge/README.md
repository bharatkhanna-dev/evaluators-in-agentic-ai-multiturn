# LLM-as-Judge

Structured rubric-based judge with calibration testing.

## Run

```bash
python llm_as_judge/judge.py
```

## Key features

- `StructuredJudge`: reusable judge with `.score()` and `.as_langsmith_evaluator()`
- Pre-defined rubrics: `factual_correctness`, `response_completeness`, `tool_call_necessity`, `conversation_goal_completion`, `safe_refusal`
- `calibration_test()`: validate judge agreement against human-labeled anchors
- `with_structured_output(JudgeVerdict)` for reliable score parsing
