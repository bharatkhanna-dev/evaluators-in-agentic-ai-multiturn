# Trajectory Evaluators

Tests a LangGraph agent at two levels:
- **Single-step:** interrupt before tool execution, assert the proposed tool call
- **Full trajectory:** run to completion, assert the set of tools used

## Run

```bash
pytest trajectory_eval/eval.py -v
```

## Key files

| File | Description |
|---|---|
| `agent.py` | LangGraph agent with `agent` and `test_agent` (interrupt_before) |
| `eval.py` | pytest tests using single-step and full-trajectory inspection |
