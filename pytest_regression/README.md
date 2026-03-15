# pytest Regression Suite

CI-ready quality gate tests for agent correctness, safety, and response quality.

## Run

```bash
# All tests
pytest pytest_regression/ -v

# With LangSmith quality gate
export LANGCHAIN_PROJECT=pytest_regression_v1
pytest pytest_regression/ -v

# Only safety tests
pytest pytest_regression/ -k "Safety"
```

## Test classes

| Class | What it tests |
|---|---|
| `TestFactualCorrectness` | Python year, Australia capital, WWW inventor, binary search |
| `TestSafetyBehavior` | Prompt injection, system prompt leak, empty input |
| `TestResponseQuality` | Substantive responses, code generation |

## Quality gate

When `LANGCHAIN_PROJECT` is set, `conftest.py` fetches mean scores from LangSmith
after the run and fails CI if any metric drops below its threshold.
