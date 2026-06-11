# Post-v1 Infra Improvements

Secondary plan. Nothing here gets built before key day — v1 ships on the current engine, then these land in roughly this order. Each item names the code it changes so the motivation stays checkable.

The recurring theme: today, one coverage arc buys one full LLM conversation, serially, with no reuse between arcs, retries, or runs. That's fine for a 5-gap PR diff and wasteful everywhere else.

---

## 1. Gap clustering — prompts scale with functions, not arcs

Today `engine/writer.py` builds one conversation per arc ("cover the following uncovered branch", singular), even when several uncovered arcs share one enclosing function. One test frequently covers sibling arcs for free; nothing exploits that. A full-scope run on this repo finds 1,599 gaps — at one conversation per arc that's untenable.

Change: cluster selected gaps by enclosing function (then file) in `gaps/select.py`. One prompt asks for tests covering all K sibling arcs. The Executor already verifies per-arc; extend acceptance to be partial — keep the test, re-queue only the missed arcs with a targeted critique.

## 2. Conversation continuity on retry

Today a retry rebuilds `messages` from scratch with only a critique string carried over (`writer.py::_build_user_prompt` retry_section). The agent's prior tool observations — everything it learned reading source and running candidates — are thrown away and repurchased.

Change: keep the ReAct conversation alive across the rewrite loop. The Executor's verdict (stderr, which arcs missed) becomes the next user turn. Pairs directly with item 3: the growing prefix becomes cache reads instead of repeat full-price tokens.

## 3. Prompt-cache-aware layout + registry capability flag

The system prompt is identical for every gap, every retry, every run, and is re-sent at full price. Layout the prompt stable-prefix-first (system prompt, then file/context block; the gap-specific ask last), process gaps file-by-file so sibling conversations share the file prefix, and add a `prompt_caching` flag per model in `models.json`, setting vendor cache markers where supported.

## 4. Real cost ledger (pull this one into key day)

The budget check in `writer.py` is dead code in practice: it reads `getattr(response, "cost", 0)` — not a field litellm populates — and compares a single response's cost against the whole-run `budget_usd`. The budget is effectively unenforced.

Change: a run-level `CostLedger` — usage tokens × registry pricing (fallback: litellm's computed cost), shared across all gaps, enforced cumulatively, spend reported in `RunReport`. Key day is the first time money is real, so this lands before it.

## 5. Concurrency

The gap loop in `cli.py` is strictly serial with sync `litellm.completion`. Gaps are independent until acceptance. Move to `litellm.acompletion` + K async gap workers; serialize only writes to `tests_dir` and the RegressionGuard. Per-vendor `max_concurrency` lives in the registry. CI minutes are the user's money too.

## 6. Executor economics

- If the agent's final `run_candidate` (in `engine/tools.py`) executed the exact same test bytes, count it as repetition #1 of the flaky check instead of starting over.
- Use coverage dynamic contexts to attribute arcs per test in one run where possible, instead of one coverage subprocess per candidate.
- `regression: affected|full|off` config knob — `affected` reruns only test files importing changed modules.

## 7. Run lifecycle — resume + cross-commit memory

Full-repo runs should be a resumable queue, not one monster run. A local ledger keyed by gap fingerprint (file + symbol + arc + content hash of the enclosing function):

- skip gaps whose accepted test still exists and whose code hasn't changed;
- back off gaps that failed repeatedly on unchanged code (don't repay the same failure every commit);
- let "entire repo" mode drain across multiple CI runs.

Also memoize jedi context per (file, symbol, depth) within a run — it's rebuilt per gap today.

## 8. Model cascade

Default cheap model first; escalate a gap to a stronger registry model only after the cheap model exhausts its retries. Calibrate against the nightly per-model numbers rather than intuition.

## 9. Trace flywheel

`AgentTrace`s from real successful runs become canned mock-LLM fixtures for the acceptance harness — the mocks stop being hand-written and start being harvested. A small analysis script over `history.jsonl`: acceptance rate by model, cost per accepted test, tool-call patterns that predict success.

## 10. Multi-user — deferred on purpose

BYOK already isolates per-user cost and rate limits; nothing aggregates on the maintainer. This only becomes a problem if a hosted surface ships, at which point it needs per-installation queues and token buckets. Recorded here so it doesn't get re-derived from scratch later.
