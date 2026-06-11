# CoverageAgent — Design & Roadmap

Single source of truth for the rebuild. `.agent/DESIGN.md` is superseded; the legacy web-demo architecture survives only on the `pre-rehaul` tag and the `legacy/web-demo` branch, and nothing from it gets reintroduced.

---

## Why this shape

The original system (web demo: paste a repo URL, watch agents run) never accepted a single test on a real repo. The failure was structural, not a tuning problem:

- A "run any repo" demo forces building an environment for an arbitrary codebase from scratch — the E2B dependency-install cost/storage problem, degrading to raw host execution when E2B wasn't configured.
- Whole-repo gap hunting on mature OSS repos is the hardest possible target: huge, old, deeply contextual gaps.
- No private repos, no diff mode, no CI story — no place where the tool would actually be used.

The fix is the form factor. In CI, the user's environment already exists (their workflow installed the deps), private repos work via the workflow's own token, and a PR diff yields small, fresh, winnable gaps.

---

## Decisions (final)

| Question | Decision | Why |
|---|---|---|
| Form factor | **CI-first: GitHub Action + installable Python CLI core.** A read-only results dashboard may come later; it never executes runs. | The environment problem evaporates in CI; private repos via `GITHUB_TOKEN`; diff scope = winnable gaps. |
| Execution | **Generated tests run in the caller's own environment** — the user's GitHub Actions job (runner VM = isolation boundary) or their local checkout (trusted by definition). No remote sandbox. | Zero compute cost, no template lifecycle, no paper isolation. |
| Cost model | **BYOK LLM key via repo secret**, routed through litellm. The maintainer pays nothing for others' usage. | Standard for CI AI tools. |
| Orchestration | **LangGraph StateGraph per gap** (typed contracts, conditional routing, retry cycles) with the **TestWriter as a ReAct agent with tools**. | One genuinely agentic loop — a ReAct test engineer with execution tools, gated by deterministic verification — beats many thin wrapper agents. |
| Models | **Registry-driven picker** (`coverage_agent/models.json`): provider, key prefix, key env var, tool-calling flag per model. Default `gemini/gemini-2.5-flash` (free tier, tool-calling). Unknown IDs warn but pass through to litellm. | Cheap by default; the choice belongs to the user; one small data file drives validation, the `models` CLI listing, and docs. |
| Scope | **Coverage gaps are the v1 wedge — on merit.** Deterministic ground truth (an arc is hit or it isn't); diff/patch coverage in CI is a proven category. Extension path later: mutation-testing gaps, regression-repro tests from diffs. | Name stays CoverageAgent. |
| Branch/commit targeting | The CLI runs in the current checkout only. Whole-branch = check out the branch (in CI, the workflow's `ref`) and run `--scope full`. A commit or commit range = `--scope diff --base <ref|SHA>` (e.g. `--base HEAD~3`). No `--repo` clone flag; clone-and-run convenience lives only in the benchmark harness. | Covers every targeting need without running third-party code on the host by accident. |
| Keys | **Everything is built and verified key-free.** Real LLM keys appear exactly once, at the final validation ("key day"). | Cheap iteration; the proof run happens against a finished system. |
| Hosted demo | Not in v1. | It is exactly the piece that forces arbitrary-repo execution on maintainer infra. |

## Removed by design — do not reintroduce

E2B / any sandbox-as-a-service; Braintrust or any eval-logging SaaS (observability = `AgentTrace` + `RunReport` JSON); web UI as an execution surface, demo mode, rate limiters, TPM throttles, preflight services, SQLite run history; offline/`is_offline` mode in production code (tests mock `litellm.completion`); the `--repo` clone flag; LLM-judged eval scores. Each of these has already been removed at least once. If work seems to require one of them, the work is misdirected — stop and log a blocker.

---

## Architecture

### Engine — LangGraph StateGraph + ReAct TestWriter

```
GapFinder (det.) → select → per gap, LangGraph StateGraph:
   ContextBuilder (jedi) ──→ TestWriter (ReAct agent, LLM + tools) ──→ Validator (det.)
        ↑ unknown imports → depth+1 (max 2) ←──────────────┘ (REWRITE w/ critique)
   → Executor (det. verification gate: pytest subprocess in the CALLER'S env,
               flaky_runs repetitions, per-gap target-hit via coverage API)
        → pass + targets hit → accept | fail → stderr critique → TestWriter (≤ max_retries)
→ RegressionGuard (full suite once, bisect offenders) → Reporter (PR comment / patch / commit)
```

The **ReAct TestWriter** is the centerpiece: a tool-using loop (litellm function-calling, works across registry vendors) with `read_source`, `find_symbol`, `find_usages`, and `run_candidate` (executes the draft and reports pass/fail, stderr, and whether the gap's target arcs were hit). The agent probes the repo, runs its own draft, observes the failure, and fixes it. Budgets: `max_tool_calls` per gap plus token/`budget_usd` ceilings; every step is recorded as an `AgentTrace`.

The **Executor remains a separate deterministic gate** — the agent's own runs don't count. Verification reads the coverage API (`coverage.Coverage(data_file=...)` arcs) and junit XML + exit codes; never regex on pytest stdout. Coverage delta is per-gap: of this gap's N target lines/arcs, how many this test newly covered.

### Acceptance vs. delivery (terminology — do not conflate)

- **Accept** (engine decision, no git): the test passes every gate — pytest green across all `flaky_runs`, the gap's target lines/**logical coverage arcs** (`from_line → to_line`, nothing to do with git branches) verifiably hit, RegressionGuard clean. Accepted tests are written to `tests/generated/`. Metric: `tests_accepted`.
- **Deliver** (git action, `commit_mode`): `comment` (default — unified diff in a `<details>` block of the PR comment, zero git writes), `commit` (git-commit to the PR branch), `pr` (stacked PR). Never touch any file outside `tests_dir`. Reserve the word "commit" for git.

### Diff-gap algorithm (`gaps/diff.py`)

1. Base = `--base <ref|SHA>` if given, else merge-base with `origin/main`/`origin/master` (in CI: `origin/$GITHUB_BASE_REF`).
2. `git diff -U0 <base>...HEAD -- '*.py'` → changed-line map; excludes tests, excluded globs, pure renames, deletions.
3. Map to coverage: changed line ∈ `missing_lines` → `line` gap (grouped per enclosing function); a missing arc touching changed code → `branch` gap; a file absent from coverage entirely (new file) → `function` gaps from AST. Changed-but-covered files are surfaced as good news in the report.
4. Deterministic selection: function gaps in new files first, then branch, then line; IO-difficulty heuristic demotes rather than skips; cap at `max_gaps` with substitution from the tail when a gap is skipped. No LLM ranking.

### Config (`.coverage-agent.yml`, pydantic-validated, flag/env overridable)

`test_command`, `coverage_file` (default posture: the user runs their own tests with `coverage run --branch` and hands over the file; fallback: run `test_command` under coverage; hard error with a copy-pasteable fix), `source`, `scope: full|diff`, `paths`, `exclude`, `tests_dir: tests/generated`, `commit_mode: comment|commit|pr`, `model` (registry-validated), `max_gaps`, `max_retries`, `max_tool_calls`, `flaky_runs`, `test_timeout`, `budget_usd` (hard stop), `dashboard_url` (optional).

### GitHub Action (composite, `action.yml` at repo root)

- Installs the pinned package via `uvx`. **Agent deps live isolated in the uvx env; pytest/coverage subprocesses run through the job's python** (the user's venv on PATH) — the one subtle packaging invariant, pinned with a test.
- Inputs: `llm-api-key` (required), `model`, `coverage-file`, `scope`, `commit-mode`, `config-path`, `version`, `dashboard-url`, `github-token`. Outputs: `tests-added`, `gaps-found`, `report-path`.
- PR comment upserted idempotently via a `<!-- coverage-agent:report -->` marker. Permissions: `pull-requests: write` always; `contents: write` only for commit mode.
- Self-trigger guard: exit 0 if the head commit message starts with `coverage-agent:` or the diff touches only `tests_dir`.

### Contracts (`coverage_agent/contracts.py`)

`BranchGap`, `CoverageGap` (`kind: branch|line|function`, `origin: diff|full`), `ContextPayload`, `DraftTest`, `ValidationResult`, `ExecutionResult` (`targets_hit`/`targets_total`), `RegressionResult`, `AgentTrace` (ReAct steps: tool calls, observations, tokens), `RunReport` — one schema consumed by the CLI JSON output, the PR comment renderer, and (later) dashboard ingest.

### Dashboard (optional, after v1)

`dashboard/` with its own deps: FastAPI `POST /api/runs` (bearer token) + `GET /api/repos/{owner}/{repo}/runs`, SQLite on a free-tier volume, one static page (run history, accepted/found ratio, cost, per-gap agent traces). The Action posts fire-and-forget (5s timeout, warn-only); a dead dashboard never fails a run.

---

## Verification rules

1. **Nothing is claimed, released, or listed until the finished system has been run for real** — real LLM key, real repo, real PR — and the run shows tests accepted, their target arcs verifiably covered afterward, and the suite still green. The `RunReport` JSON is saved to `benchmarks/results/` as the proof artifact. `benchmarks/results/` is gitignored: interim run outputs never leave the machine; only final real-key proof artifacts get published, deliberately, with `git add -f` on key day.
2. **Proof order:** mock-LLM harness in CI (zero keys, every push) → synthetic fixture repo with known answers → dogfood on this repo itself via the Action. The last one is the bar that counts.
3. **Measured numbers only.** Anything published comes from actual runs — never projected, never fabricated, including costs.
4. **A deliverable is done only when its acceptance criteria below have been run and passed.** Code-complete is not done.
5. If the fixture benchmark cannot reach its bar after the ReAct loop plus one round of prompt/model iteration, stop building and diagnose with the `AgentTrace` data — do not continue on hope.
6. Improving the odds comes later, honestly: once it genuinely works, deliberately target repos where wins are likely (actively maintained Python, 40–80% coverage, logic-heavy, light IO) to build a public track record, with the selection criteria documented next to the results.

---

## Roadmap (remaining work, in order — everything before "Key day" is key-free)

**1. Verification harness.**
- `benchmarks/fixture_repo/`: small synthetic package with ~8 known uncovered arcs incl. a new-file case (plain directory; the harness git-inits it).
- `benchmarks/run_acceptance.py`: git-inits the fixture, creates a branch adding a function with 3 known-uncovered arcs, runs `--scope diff --base main` and `--scope full`. Asserts gaps_found == expected, tests_accepted ≥ 2, fixture suite green. Two modes: `--mock-llm` (patches `litellm.completion` with hand-written canned ReAct sequences whose final tests genuinely cover the fixture arcs) and real-key mode (same harness, used on key day). Repo-cloning convenience for ad-hoc benchmarks lives here, not in the CLI.
- `.github/workflows/ci.yml`: pytest + mock-LLM acceptance on every push.
- *Acceptance:* the harness passes in CI with zero secrets configured.

**2. Action & reporting.**
- `coverage_agent/report/markdown.py`: PR comment body (marker, results table, diff in `<details>`, arc-miss guidance).
- `coverage_agent/report/github.py`: idempotent comment upsert, commit/push, stacked PR — unit-tested against recorded REST fixtures, plus a dry-run print mode.
- `cli.py ci-run`: reads `GITHUB_*` env, resolves base from `GITHUB_BASE_REF`, self-trigger guard, writes `$GITHUB_OUTPUT`.
- `action.yml` as specified above.
- *Acceptance:* all tests green; in CI, a mock-LLM `ci-run --dry-run` renders a correct comment body end-to-end. Still zero secrets.

**3. Key day — the only step requiring secrets, one sitting.**
- Real-key acceptance run on the fixture repo (same harness, no mock).
- Dogfood: `coverage run --branch -m pytest && coverage-agent run --scope full --max-gaps 5` on this repo — ≥2 tests accepted, suite green, target arcs verifiably covered; `RunReport` artifact published with `git add -f` — the only results that ever go public are final real-key runs.
- Real PR through the Action: comment mode, then commit mode; verify the comment updates (not duplicates) on PR edit.
- Measured numbers → README scorecard; PyPI publish; tag v1; Marketplace listing.

**4. Afterward, optional:** dashboard; per-model measured comparison (free tier vs paid) appended to `benchmarks/results/history.jsonl` by a nightly acceptance run; ReAct-vs-single-shot ablation for the README.

---

## Risks

1. User CI produces line-only coverage (no `--cov-branch`) → degrade to line/function gaps and say so in the comment, with the one-line fix.
2. Container jobs / tox make "the job's python" ambiguous → `test_command` is the documented escape hatch.
3. The free-tier default may underperform on hard gaps → registry + measured per-model numbers make upgrading a one-line config change; `run_candidate` feedback compensates for raw model weakness.
4. ReAct loop cost/latency blowup → `max_tool_calls` + `budget_usd` hard caps; every step traced.
5. Generated tests can be mock-heavy or overfit → prompt rules + RegressionGuard; documented v1 limit.
6. Slow suites make RegressionGuard expensive → `regression: affected|full|off` knob later.
7. Self-trigger loops in commit mode → guard keys on the commit-message prefix, not paths alone.
