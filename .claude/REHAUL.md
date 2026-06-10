# CoverageAgent Rehaul

This document supersedes `.agent/DESIGN.md`. It is the single source of truth for the v1 rebuild.

---

## Why the rehaul

The project has never produced a working result: benchmark commit rate is **0% across every run** (v1–v3 on `psf/requests`). The code is mostly fine — credential threading, Pydantic contracts, the sandbox-stderr-as-critique retry loop, jedi context, and deterministic eval all hold up — but the product framing is wrong, and every pain point flows from it:

- The "paste a repo URL" web demo forces the unsolvable problem: building an environment for an arbitrary repo from scratch. That's the E2B dependency-install cost/storage problem, and the reason the sandbox degrades to raw host execution when E2B isn't configured — arbitrary code execution on the server.
- Whole-repo gap hunting on mature OSS repos is the hardest possible target — huge, old, deeply contextual gaps. That's why commits sit at 0%.
- No private repos, no incremental/diff mode, no CI story. The tool has no place to live where anyone would actually use it.

The fix is a form-factor change, not a code polish. This is also a portfolio project for showcasing multi-agent workflows — the rehaul strengthens that showcase rather than stripping it.

---

## Decisions

| Question | Decision | Why |
|---|---|---|
| Form factor | **CI-first: GitHub Action + installable Python CLI core.** Web UI becomes a thin results dashboard (v1.5, optional). | In CI, the environment already exists — the user's workflow installed the deps. The sandbox/install problem evaporates. Private repos work via the workflow's own `GITHUB_TOKEN`. Diff scope means small, fresh, winnable gaps. |
| Execution/sandbox | **CI-native: generated tests run inside the user's own GitHub Actions job** (runner VM = isolation boundary). Local CLI runs in the user's own checkout, which is trusted by definition. **E2B deleted.** | Zero compute cost, no template lifecycle, no paper sandbox. |
| Cost model | **BYOK LLM key via repo secret**, routed through litellm. I pay nothing for other people's usage. | Standard for CI AI tools — the professional model, not a compromise. |
| Orchestration | **LangGraph stays** as the per-gap StateGraph (typed contracts, conditional routing, retry cycles). **TestWriter becomes a ReAct agent with tools** (see Engine). | The multi-agent showcase is a first-class goal. One genuinely agentic loop — a ReAct test engineer with execution tools, orchestrated by LangGraph, gated by deterministic verification — beats seven thin wrapper agents. |
| Model strategy | **Model picker backed by a registry** (`coverage_agent/models.json`): provider, key prefix, tool-calling support, pricing, free-tier flag per model. Users pick per run or set a default in config. **Product default: `gemini/gemini-2.5-flash`** (free tier, tool-calling). Anthropic, OpenAI, and Groq entries maintained alongside. | Cheap by default; the choice belongs to the user. A small data file powers config validation, the `coverage-agent models` CLI listing, and the Action's `model` input. |
| Scope | **Coverage stays the v1 wedge — on merit, not the name.** It's the one test-generation target with deterministic ground truth (a branch is hit or it isn't), and diff/patch coverage in CI is a proven category. v2+ extension path: mutation-testing gaps, regression-repro tests from diffs. | Name stays **CoverageAgent**. |
| Repo | **In-place rehaul**: new layout, aggressive deletion, port the keepers. **Before any demolition: annotated tag `pre-rehaul` + branch `legacy/web-demo`** at current HEAD (pushed if a remote exists). | A checkpoint to return to. Enough code survives that a greenfield repo would only add a copy step. |
| Hosted demo | **Not in v1.** v2 may add a constrained demo on allowlisted repos. | The hosted demo is exactly the piece that forces arbitrary-repo execution on my infra. |

---

## Target architecture

### Engine — LangGraph StateGraph + ReAct TestWriter

```
GapFinder (det.) → select → per gap, LangGraph StateGraph:
   ContextBuilder (jedi) ──→ TestWriter (ReAct agent, LLM + tools) ──→ Validator (det.)
        ↑ unknown imports → depth+1 (max 2) ←──────────────┘ (REWRITE w/ critique)
   → Executor (det. verification gate: pytest subprocess in the CALLER'S env,
               3-run flakiness, per-gap target-hit via coverage API)
        → pass + targets hit → accept | fail → stderr critique → TestWriter (≤ max_retries)
→ RegressionGuard (full suite once, bisect offenders) → Reporter (PR comment / patch / commit)
```

**The ReAct TestWriter is the centerpiece.** Instead of a single-shot completion, it runs a tool-using loop (litellm function-calling, works across all registry vendors):

- `read_source(path, start, end)` — pull any file beyond the jedi-seeded context
- `find_symbol(name)` / `find_usages(name)` — jedi-backed navigation
- `run_candidate(test_code)` — execute the draft in the CI env; returns pass/fail, stderr, and whether the gap's target lines/branches were hit
- Budget-capped: `max_tool_calls` plus token/`budget_usd` ceilings; every step traced into the `RunReport`

The agent probes the repo, runs its own draft, observes the failure, and fixes it — the ReAct loop subsumes most rewrite churn. The **Executor remains a separate deterministic gate** (the agent's own runs don't count): 3-run flakiness check plus target-hit verification via the coverage API (`coverage.Coverage(data_file=...)` arcs) and junit XML + exit codes. Never regex on pytest stdout.

LangGraph's conditional edges (RECONTEXTUALIZE / REWRITE / EXECUTE / accept / skip) and the executor→writer retry edge carry over from the current `pipeline.py`; the critique construction at `pipeline.py:155-180` ports verbatim. The compiled graph's Mermaid render goes in the README.

Coverage delta is per-gap: of this gap's N target lines/branches, how many did this test newly cover.

### Diff-gap algorithm (`gaps/diff.py`)

1. Base = `git merge-base origin/$GITHUB_BASE_REF HEAD` (local: `--base` flag, default merge-base with main/master).
2. `git diff -U0 <base>...HEAD -- '*.py'` → changed-line map; exclude tests, excluded globs, pure renames, deletions.
3. Map to coverage data: changed line ∈ `missing_lines` → `line` gap (grouped per enclosing function); a missing branch touching changed code → `branch` gap; **a file absent from coverage entirely (new file)** → `function` gaps from AST. Changed-but-covered files get surfaced as good news in the PR comment.
4. Deterministic selection: function gaps in new files first, then branch, then line; the IO-difficulty heuristic demotes rather than skips; cap at `max_gaps`. No LLM ranking.

### Model registry (`coverage_agent/models.json`)

```json
{"models": [
  {"id": "gemini/gemini-2.5-flash", "provider": "gemini", "key_prefix": "AIza",
   "tool_calling": true, "free_tier": true, "default": true},
  {"id": "anthropic/claude-opus-4-8", "provider": "anthropic", "key_prefix": "sk-ant-",
   "tool_calling": true, "free_tier": false, "pricing_per_mtok": [5, 25]},
  {"id": "anthropic/claude-sonnet-4-6", "...": "..."},
  {"id": "openai/...", "...": "..."},
  {"id": "groq/llama-3.3-70b-versatile", "...": "..."}
]}
```

Consumed by: config validation + the key/model coupling check (ported from `credentials.py`), the `coverage-agent models` CLI listing, the Action's `model` input, and the README model table. Unknown model IDs warn but pass through to litellm as an escape hatch. Free-tier vs paid commit-rate deltas get **measured** in Phase 5, never guessed.

### Config (`.coverage-agent.yml`, pydantic-validated, every field overridable by flag/env)

```yaml
version: 1
test_command: "pytest -q"        # fallback baseline + regression guard
coverage_file: ".coverage"       # or coverage.json / coverage.xml; auto-detected if omitted
source: []                       # default: auto-detect from pyproject/setup.cfg
scope: diff                      # diff | full | paths
paths: []
exclude: ["**/migrations/**", "**/conftest.py"]
tests_dir: "tests/generated"
commit_mode: comment             # comment | commit | pr
model: "gemini/gemini-2.5-flash" # registry-validated; key from LLM_API_KEY env
max_gaps: 10
max_retries: 3
max_tool_calls: 12               # per-gap ReAct budget
flaky_runs: 3
test_timeout: 60
budget_usd: 1.00                 # hard stop
dashboard_url: ""                # optional; DASHBOARD_TOKEN env for auth
```

Coverage acquisition order: (1) `--coverage-file` flag / Action input → (2) `coverage_file` from config if present → (3) run `test_command` under `coverage run --branch` → (4) hard error with a copy-pasteable workflow fix. Documented default posture: run your own tests with coverage in CI and hand over the file — it's free, since the tests run anyway, and it guarantees env compatibility.

### GitHub Action (composite, `action.yml` at repo root)

- Installs the pinned package via `uvx`. **Agent deps live isolated in the uvx env, but pytest/coverage subprocesses run through the job's python** (the user's venv on PATH). This is the one subtle packaging invariant — pinned with a test.
- Inputs: `llm-api-key` (required), `model`, `coverage-file`, `scope`, `commit-mode`, `config-path`, `version`, `dashboard-url`, `github-token`. Outputs: `tests-added`, `gaps-found`, `report-path`.
- PR comment upserted idempotently via a `<!-- coverage-agent:report -->` marker.
- Permissions: `pull-requests: write` always; `contents: write` only for commit mode.
- Self-trigger guard: exit 0 if the head commit message starts with `coverage-agent:` or the diff touches only `tests_dir`.

### Acceptance vs. delivery (terminology — do not conflate)

Two distinct concepts; the legacy code overloaded "commit" for both. In this codebase:

- **Accept** (engine decision, no git): a generated test passes every gate — pytest green across all `flaky_runs`, the gap's target lines/**logical coverage branches** (`from_line → to_line` arcs, nothing to do with git branches) verifiably hit, and later the RegressionGuard. Accepted tests are written to `tests/generated/`. The metric is `tests_accepted`.
- **Deliver** (git action, governed by `commit_mode`): what happens to already-accepted tests — `comment` (default: unified diff in a `<details>` block of the PR comment, zero git writes), `commit` (git-commit to the PR branch), `pr` (stacked PR).

Pipeline: per-test gate (all flaky runs pass + targets hit) → accepted tests accumulate in `tests/generated/` → RegressionGuard runs the full `test_command`; on new failures, bisect by removing accepted files and drop the offenders → delivery per `commit_mode`. Never touch any file outside `tests_dir`. In code and docs, use "accept/accepted" for the engine gate and reserve "commit" strictly for git.

### Contracts (`contracts.py`, single module)

Keep `BranchGap`, `ContextPayload`, `DraftTest`, `RegressionResult`. Changes:

- `CoverageGap` gains `kind: branch|line|function` and `origin: diff|full`; loses `priority_score`.
- `EvalResult` → `ValidationResult`: drop the vestigial `assertion_score` and `mock_complete`.
- `ExecutionResult.coverage_delta` → `targets_hit` / `targets_total`.
- New `AgentTrace`: ReAct steps (tool calls, observations, token counts).
- New `RunReport`: one schema consumed by the CLI's JSON output, the PR comment renderer, and dashboard ingest.

### Dashboard (Phase 4, optional, ~1,200 lines total)

`dashboard/` with its own dependencies: FastAPI `POST /api/runs` (bearer token) + `GET /api/repos/{owner}/{repo}/runs`, SQLite on a Fly.io free-tier volume, one static page (run history, accepted/found ratio, cost, per-gap agent traces). The Action posts fire-and-forget (5s timeout, warn-only) — a dead dashboard never fails a run.

---

## Demolition and ports (Phase 0 — only after the checkpoint)

**Checkpoint first:**

```bash
git tag -a pre-rehaul -m "Last commit of the web-demo architecture"
git branch legacy/web-demo
# push both if a remote exists
```

**Delete:** `app.py`, `public/`, `Dockerfile`, `docker-compose.yml`, `run_benchmark.py`; from `coverage_agent/`: `orchestrator.py`, `run_engine.py`, `rate_limiter.py`, `tpm_throttle.py`, `preflight.py`, `db.py`, `error_mapper.py`, `recommendations.py` (branch-miss guidance text folds into the reporter), `cost_tracker.py` (replaced by a small tally inside the writer), `sandbox/` (both runners), `agents/gap_prioritizer.py`, `agents/gap_filter.py` (heuristic absorbed into `gaps/select.py`), `agents/context_architect.py` (`jedi_graph` called directly), `agents/result_summarizer.py`, `evals/braintrust_logger.py`. Drop deps: e2b, braintrust, fastapi/uvicorn (fastapi moves to `dashboard/`). **langgraph stays.**

**Port:**

| From | To | Notes |
|---|---|---|
| `credentials.py` | `credentials.py` | trimmed: drop demo/e2b/braintrust fields; keep frozen dataclass, provider/key coupling, `should_commit`, offline mode |
| `contracts/schemas.py` | `contracts.py` | schema changes above |
| `context/jedi_graph.py`, `context/branch_conditions.py` | `context/` | as-is |
| `context/coverage_parser.py` | `gaps/coverage_data.py` | add `.coverage` data-file and Cobertura XML loaders |
| `pipeline.py` | `engine/graph.py` | LangGraph topology + critique logic kept; nodes rewired |
| `agents/test_writer.py` | `engine/writer.py` | seed; then the ReAct upgrade |
| `agents/eval_agent.py` (deterministic half) | `engine/validator.py` | |
| `agents/execution_runner.py` + local_runner pytest parsing | `engine/executor.py` | coverage-API parsing, no stdout regex |
| `agents/regression_guard.py` | `engine/regression.py` | |
| fixtures; `test_pipeline_offline.py`, `test_credentials_isolation.py`, `test_branch_conditions.py`, validator/executor agent tests | `tests/` | adapted |

---

## Target layout

```
coverage-agent/
├── action.yml
├── pyproject.toml              # pydantic, litellm, langgraph, jedi, coverage[toml], pyyaml, ruff, typer
├── coverage_agent/
│   ├── cli.py                  # typer: run, ci-run, models, report
│   ├── config.py               # .coverage-agent.yml loader
│   ├── models.json             # model registry
│   ├── credentials.py
│   ├── contracts.py            # schemas + AgentTrace + RunReport
│   ├── gaps/{coverage_data,diff,select}.py
│   ├── context/{jedi_graph,branch_conditions}.py
│   ├── engine/{graph,writer,tools,validator,executor,regression}.py
│   ├── report/{run_report,markdown,github,dashboard}.py
│   └── fixtures/
├── dashboard/                  # Phase 4: app.py, db.py, static/, fly.toml
├── benchmarks/{fixture_repo/, run_acceptance.py, naive_baseline.py}
└── tests/
```

---

## Mandatory validation milestone (explicit step, after the workflow is complete)

Once the agent workflow is actually ready — engine, diff mode, and Action all built — there is an **explicit, non-skippable validation step before anything is released or claimed**. The previous build died precisely because this step never happened: the system was "done" on paper and never proven end-to-end. Concretely:

1. **The full system gets run for real, as a dedicated milestone.** Real LLM key, real repo, real PR. The run must show sureshot results: tests accepted, their target lines/branches verifiably covered afterward, suite still green. The output (`RunReport` JSON) is saved to `benchmarks/results/` as the proof artifact. Until this run exists, the project does not "work" — no README claims, no PyPI release, no Marketplace listing.
2. **Validation order:** offline fixtures (fast, zero keys) → the synthetic `fixture_repo` with known answers → dogfood on this repo itself via the Action. The last one is the bar that counts.
3. **Measured numbers only.** Anything published comes from actual runs — never projected, never fabricated, including costs.
4. **Improving the odds comes later, and honestly.** Once it genuinely works, we can deliberately pick repos where catching gaps is most likely — actively developed Python repos with mid-range coverage (40–80%), pure-logic modules, light IO — to build a credible public track record. Selection criteria get documented next to the results. This is an amplifier for proven capability, never a substitute for it.
5. **Ship-vs-research tripwire.** The product shell (Action packaging, reporting, dashboard) is only built on top of a proven engine — Phase 1's dogfood gate and Phase 2's known-answer fixture come first precisely so failure is cheap and early. If the fixture benchmark can't reach its acceptance bar after the ReAct upgrade and one round of prompt/model iteration, stop building and diagnose with the per-step `AgentTrace` data — do not proceed to Phase 3 on hope. The definition of shipped is Phase 3's gate: a stranger goes from README to a working PR comment in ~10 minutes with one secret and one workflow block.

## Phases — each ends with a verifiable artifact (the anti-0% discipline)

- **Phase 0 — Checkpoint + demolition + ports (~½ day).** Tag `pre-rehaul` + branch `legacy/web-demo` first. Accept: tag/branch exist; `uv run pytest` green on the slimmed suite; `git grep -l 'e2b\|braintrust'` returns nothing.
- **Phase 1 — Core engine, full scope, dogfood (~3 days).** Rebuild the LangGraph graph with a single-shot writer first, then the ReAct upgrade (`engine/tools.py`, budget caps, AgentTrace). Accept: `coverage run --branch -m pytest && coverage-agent run --scope full --max-gaps 5` **on this repo itself** accepts ≥2 tests, suite stays green, target branches verifiably covered afterward; the `RunReport` JSON (with agent traces) committed to `benchmarks/results/phase1.json`.
- **Phase 2 — Diff mode + fixture benchmark (~2 days).** Build `benchmarks/fixture_repo/` (synthetic package, ~8 known uncovered branches including a new-file case) and `run_acceptance.py` (git-init the fixture, create a branch adding a function with 3 known-uncovered branches, run `--scope diff`, assert gaps_found == expected, tests_accepted ≥ 2, suite green). Accept: passes with a real key locally AND in offline-fixture mode in CI (offline fixtures include canned ReAct tool-call sequences).
- **Phase 3 — Action + reporting (~2 days).** Publish to PyPI, tag v1. Accept: this repo's own workflow uses `uses: ./`; a real PR gets the comment; a second PR with `commit-mode: commit` lands a green commit; editing the PR updates — not duplicates — the comment.
- **Phase 4 — Dashboard (~1–2 days, optional).** Accept: Phase 3's real run renders on the hosted page; killing the dashboard does not fail the Action.
- **Phase 5 — Hardening + honesty (ongoing).** Nightly acceptance run appends commit-rate to `benchmarks/results/history.jsonl`, broken out per registry model (measured free-vs-paid delta). README rewritten with measured numbers only. Naive single-shot baseline rerun — now also against the ReAct writer, which is the ablation worth publishing.

---

## Risks

1. User CI produces line-only coverage (no `--cov-branch`) → degrade to line/function gaps and say so in the comment, with the one-line fix.
2. Container jobs / tox make "the job's python" ambiguous → `test_command` is the documented escape hatch.
3. The free-tier default (Gemini Flash) may underperform on hard gaps → registry tiers + measured per-model numbers make upgrading a one-line config change; `run_candidate` feedback compensates for raw model weakness.
4. ReAct loop cost/latency blowup → `max_tool_calls` + `budget_usd` hard caps; every step traced.
5. Generated tests can be mock-heavy or overfit → prompt rules + regression guard; documented v1 limit.
6. Slow suites make RegressionGuard expensive → `regression: affected|full|off` knob in Phase 5.
7. Self-trigger loops in commit mode → guard keys on the commit-message prefix, not paths alone.
