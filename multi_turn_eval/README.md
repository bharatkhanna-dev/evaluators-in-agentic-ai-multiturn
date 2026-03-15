# Multi-Turn Simulation Evaluators

Simulates realistic user conversations using an LLM-driven "simulated user agent"
that dynamically adapts its responses.

## Run

```bash
python multi_turn_eval/run_eval.py
```

## Key files

| File | Description |
|---|---|
| `simulation.py` | `SimulatedUser`, `TurnEvaluator`, `run_simulation()` |
| `run_eval.py` | Runs predefined scenarios and prints summary metrics |

## Scenarios

- `password_reset` -- non-technical user, basic support task
- `billing_dispute` -- business user, refund resolution
- `api_integration_help` -- developer, technical documentation
- `adversarial_confused_user` -- user with completely wrong expectations
