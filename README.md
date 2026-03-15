# evaluators-in-agentic-ai-multiturn

Code examples for the article: [Continuously Improving Agent Quality Using Evaluators Across Single-Turn, Trajectory, and Multi-Turn Interactions](https://bharatkhanna.dev/ai/evaluators-in-agentic-ai-multiturn/)

Covers the full evaluation stack for LangGraph agents: single-turn unit evals, trajectory scoring, LLM-as-judge, multi-turn user simulation, LangGraph-native patterns, and a pytest regression harness.

## Quick start

```bash
git clone https://github.com/bharatkhanna-dev/evaluators-in-agentic-ai-multiturn.git
cd evaluators-in-agentic-ai-multiturn
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your LANGCHAIN_API_KEY and OPENAI_API_KEY
```

Requires Python 3.11+, a LangSmith account, and an OpenAI API key.

## Structure

```
single_turn_eval/        # Layer 1: heuristic + LLM-as-judge unit evals
trajectory_eval/         # Layer 2: single-step and full-trajectory pytest tests
llm_as_judge/            # Structured rubric judge with calibration testing
multi_turn_eval/         # Simulated user conversations with per-turn scoring
langgraph_agent_eval/    # Middleware patterns + automated trace analysis
datasets/                # Create and push LangSmith evaluation datasets
pytest_regression/       # CI-ready pytest quality gate with LangSmith scoring
```

## Running the examples

```bash
# Create LangSmith datasets first
python datasets/create_and_push.py

# Single-turn eval
python single_turn_eval/eval.py

# Trajectory evals
pytest trajectory_eval/eval.py -v

# LLM judge calibration
python llm_as_judge/judge.py

# Multi-turn simulation
python multi_turn_eval/run_eval.py

# Evaluator drift check
python langgraph_agent_eval/evaluator_calibration.py

# Full regression suite
pytest pytest_regression/ -v
```

## License

MIT
