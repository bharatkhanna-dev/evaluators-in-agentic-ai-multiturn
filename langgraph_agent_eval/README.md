# LangGraph Agent Evaluation

Advanced evaluation patterns for LangGraph agents: middleware patterns, automated trace analysis, and evaluator calibration.

## Run

```bash
# Middleware agent demo
python langgraph_agent_eval/agent_with_middleware.py

# Evaluator calibration check
python langgraph_agent_eval/evaluator_calibration.py

# Analyze experiment failures
python langgraph_agent_eval/trace_analyzer.py --experiment my-experiment-name
```

## Key files

| File | Description |
|---|---|
| `agent_with_middleware.py` | LangGraph agent with loop detection and pre-completion checklist |
| `trace_analyzer.py` | Fetch failing runs and synthesize improvement suggestions |
| `evaluator_calibration.py` | Test judge alignment against human-labeled anchors |
