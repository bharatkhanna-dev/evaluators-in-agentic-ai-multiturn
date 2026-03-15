# Datasets

Creates and pushes evaluation datasets to LangSmith.

## Run

```bash
# Create datasets (run once before running evals)
python datasets/create_and_push.py

# Run an experiment
python datasets/run_experiment.py --version v2 --dataset agent_qa_v2
```

## Datasets

| Dataset | Examples | Purpose |
|---|---|---|
| `agent_qa_v2` | 10 | Factual QA with edge cases and adversarial inputs |
| `trajectory_dataset` | 4 | Tool routing and trajectory evaluation |
