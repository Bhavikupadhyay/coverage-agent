# CoverageAgent

A multi-agent pipeline that finds uncovered Python branches in your PR, writes targeted pytest tests for them, and verifies every test by actually executing it before it reaches you.

![CI](https://github.com/Bhavikupadhyay/coverage-agent/actions/workflows/ci.yml/badge.svg) ![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue) ![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-lightgrey)

A test is kept only when three deterministic gates agree: pytest passes on every repetition, the coverage data proves the target arc was executed, and the repo's full existing suite still passes. The LLM never judges its own work.

## Use it on pull requests

```yaml
# .github/workflows/coverage-agent.yml
name: coverage-agent
on: pull_request
permissions:
  pull-requests: write

jobs:
  coverage-agent:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0          # the diff base must be resolvable
      - uses: astral-sh/setup-uv@v5

      # Install YOUR project and put its venv first on PATH — generated
      # tests execute in your environment, with your dependencies.
      - name: Install project
        run: |
          uv sync
          echo "$PWD/.venv/bin" >> "$GITHUB_PATH"

      - uses: Bhavikupadhyay/coverage-agent@main
        with:
          llm-api-key: ${{ secrets.LLM_API_KEY }}
          model: groq/llama-3.3-70b-versatile   # free tier works
```

On every PR it diffs against the base branch, finds new or changed code that no test executes, generates tests for it, verifies them on the runner, and posts one comment with the results and the test files as diffs. The comment is updated in place on later pushes, never duplicated. Add `commit-mode: commit` (plus `contents: write`) to have accepted tests committed to the PR branch instead.

[PR #2 on this repo](https://github.com/Bhavikupadhyay/coverage-agent/pull/2) is a live example: the PR added a module with no tests, and the Action generated five verified tests for it and posted them — using this repo's own workflow file, on a free Groq key.

## Measured results

Every number below comes from a real run. Raw `RunReport` JSON for final runs lives in [`benchmarks/results/`](benchmarks/results/); nothing is projected.

| Run | Model (free tier) | Result |
|---|---|---|
| Known-answer benchmark, diff scope | Groq llama-3.3-70b | 3/3 gaps accepted |
| Known-answer benchmark, full scope | Groq llama-3.3-70b | 9/9 gaps accepted |
| Known-answer benchmark, both scopes | Gemini 2.5 Flash | 10/12 gaps accepted |
| This repo's own codebase, full scope | Groq llama-3.3-70b | 5 gaps accepted in one run, incl. a 5-arc cluster; full suite stayed green |
| Live PR ([#2](https://github.com/Bhavikupadhyay/coverage-agent/pull/2)) | Groq llama-3.3-70b | 5 verified tests posted; comment upserted across 2 runs |

The known-answer benchmark is a synthetic package with exactly 12 uncovered arcs whose covering tests are known to exist; it runs against a mock LLM in CI on every push and against real keys before releases. The tests under [`tests/generated/`](tests/generated/) were written by the agent for this repo and are now part of the suite.

Two design choices matter for free-tier keys. Gaps in the same function are clustered into one conversation, so LLM requests scale with functions touched rather than arcs found (a 10-gap run on this repo used 3 conversations). And the writer must verify its draft with its `run_candidate` tool before submitting, which is where most of the acceptance rate comes from.

## Pipeline

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	context_architect(context_architect)
	gap_filter(gap_filter)
	test_writer(test_writer)
	eval_agent(eval_agent)
	execution_runner(execution_runner)
	accept(accept)
	skip(skip)
	__end__([<p>__end__</p>]):::last
	__start__ --> context_architect;
	context_architect --> gap_filter;
	eval_agent -. &nbsp;CONTEXT_ARCHITECT&nbsp; .-> context_architect;
	eval_agent -. &nbsp;EXECUTION_RUNNER&nbsp; .-> execution_runner;
	eval_agent -. &nbsp;SKIP&nbsp; .-> skip;
	eval_agent -. &nbsp;TEST_WRITER&nbsp; .-> test_writer;
	execution_runner -. &nbsp;ACCEPT&nbsp; .-> accept;
	execution_runner -. &nbsp;SKIP&nbsp; .-> skip;
	execution_runner -. &nbsp;TEST_WRITER&nbsp; .-> test_writer;
	gap_filter --> test_writer;
	test_writer --> eval_agent;
	accept --> __end__;
	skip --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

| Node | What it does |
|---|---|
| **context_architect** | Jedi traversal: assembles target function source + callee signatures into a `ContextPayload` (≤15k tokens) |
| **gap_filter** | Marks gap difficulty (`easy`/`hard`) based on IO-heavy symbols and context token count |
| **test_writer** | ReAct agent: litellm function-calling with `read_source`, `find_symbol`, `find_usages`, and `run_candidate` — it must execute its own draft and see the target hit before submitting |
| **eval_agent** | Deterministic gate: `ast.parse` syntax check + import plausibility; routes to EXECUTE, REWRITE, or RECONTEXTUALIZE |
| **execution_runner** | Runs `coverage run --branch -m pytest`; verifies each cluster arc in the coverage data; repeats `flaky_runs` times |
| **accept / skip** | Terminal nodes; a failed execution feeds its stderr back to the writer as the next critique |

One LLM agent, many deterministic actors: only `test_writer` calls a model. Verification, routing, and regression checks are all machinery the model cannot influence.

## CLI

```bash
git clone https://github.com/Bhavikupadhyay/coverage-agent.git
cd coverage-agent
uv sync

export GROQ_API_KEY=<your-key>          # or GEMINI_API_KEY etc.
export COVERAGE_AGENT_MODEL=groq/llama-3.3-70b-versatile

# Run on the current checkout (auto-runs coverage if no .coverage file found)
coverage-agent run --scope full --max-gaps 5

# Only target the changes since a ref (a branch, a SHA, or a commit range base)
coverage-agent run --scope diff --base origin/main
coverage-agent run --scope diff --base HEAD~3

# Write a RunReport JSON, pretty-print it, list models
coverage-agent run --scope full --max-gaps 5 --output report.json
coverage-agent report report.json
coverage-agent models
```

## Configuration

Drop a `.coverage-agent.yml` in your repo root. All fields are optional.

```yaml
version: 1
model: groq/llama-3.3-70b-versatile   # any litellm model string
scope: diff                            # diff | full
max_gaps: 10
test_command: pytest -q
tests_dir: tests/generated
flaky_runs: 3
test_timeout: 60
exclude:
  - "**/migrations/**"
  - "**/conftest.py"
  - "tests/**"
  - "test_*.py"
```

## Models

Any model in [`coverage_agent/models.json`](coverage_agent/models.json) — Gemini, OpenAI, Anthropic, Groq, xAI, Cerebras, Mistral. Pass `--model <id>` to override. The ReAct tool loop activates for models with `tool_calling: true`; others fall back to single-shot generation. Free-tier notes from real runs: Groq's free tier handled every benchmark in this README; Gemini's free tier is capped at 20 requests per day, enough for small PRs only.

## Tests

```bash
uv run pytest -q
# 158 tests, ~20 seconds, no network calls, no keys
python benchmarks/run_acceptance.py --mock-llm   # the known-answer benchmark, also key-free
```

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
