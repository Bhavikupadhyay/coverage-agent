# CoverageAgent

A multi-agent AI pipeline that autonomously improves test coverage on real Python codebases.

## What it does

1. **Coverage Baseline** — runs `pytest --branch --cov` in an E2B sandbox to get branch-level gaps
2. **Gap Prioritizer** — LLM ranks uncovered branches by testability and value
3. **Context Architect** — uses Jedi static analysis to build a targeted context payload for each gap
4. **Test Writer** — Gemini 2.5 Flash generates pytest functions targeting the specific branch
5. **Eval Agent** — deterministic + LLM evals gate the test before execution (syntax, imports, mocks, assertion quality)
6. **Execution Runner** — runs the test in E2B, measures real coverage delta, commits passing tests

The eval loop (steps 3–5) retries up to 3 times per gap via LangGraph before skipping.

## Stack

- **Orchestration**: LangGraph state machine per gap, plain Python outer loop
- **LLM**: `gemini/gemini-2.5-flash` via LiteLLM
- **Static analysis**: Jedi
- **Sandbox**: E2B (Firecracker micro-VMs)
- **Evals + tracing**: Braintrust
- **Schemas**: Pydantic v2

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env  # fill in GEMINI_API_KEY, E2B_API_KEY, BRAINTRUST_API_KEY
```

## Usage

```bash
python run_benchmark.py --repo https://github.com/psf/requests --max-gaps 10
```

## Benchmark Results

*Populated after full benchmark run.*

| Repository | Gaps Targeted | Tests Committed | Skipped | Branch Hit Rate | Coverage Delta | Avg Loops | LLM Cost |
|---|---|---|---|---|---|---|---|
| `requests` | — | — | — | — | — | — | — |
| `pydantic` | — | — | — | — | — | — | — |
| `click`    | — | — | — | — | — | — | — |
