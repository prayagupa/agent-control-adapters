# Validation evidence

Executed locally on October 2, 2026 UTC with Python 3.13.14, published
CrewAI 1.15.23 and agent-hooks-sdk 0.1.0a5. No sibling framework fork was installed.

| Check | Result |
|---|---|
| Locked `uv` environment setup | Passed |
| Ruff lint and formatting | Passed |
| Strict Pyright, package source and examples | 0 errors, 0 warnings |
| Adapter deterministic tests | 175 passed |
| Example I/O safety tests | 19 passed |
| Unmodified SDK reference-host vectors | 47 passed |
| Combined deterministic suite | **241 passed** |
| Required actual-Qwen example e2e | **6 passed**, none skipped |
| Bandit, package source and examples | Passed |
| Independent wheel and sdist build | Passed |
| Local Markdown links | No missing targets |
| Mermaid rendered-image validation | Not run; no renderer provisioned |
| Dependency audit | **Blocked: four Chroma advisories**, see [SECURITY](../SECURITY.md) |

The SDK reference-host vectors are **not** a full adapter-conformance claim.
Deprecation warnings from the pinned CrewAI APIs remain visible; passing tests do
not imply that upgrading the framework is safe without a new compatibility run.
Only Python 3.13 was exercised locally; CI is configured for 3.11 but has not
been run on GitHub by this task.

## Regressions from review

The final deterministic suite includes 37 added regressions for:

- Exhaustion boundaries at `max_iter=1,2,3`, including actual repeated ReAct and
  native tool calls on both released executors. No unhooked forced-final request
  is allowed.
- Initially empty native tool arguments on the legacy executor: a changed
  transform blocks before tool defaults execute. Valid nonempty transforms are
  still tested on both executors.
- Startup authorization of tools declared only on a task.
- Rejecting crew skills before loading, and rejecting ungoverned summarization,
  function-calling repair and structured-output paths before effects.

Six additional regressions cover reused executors: four stale summarization/
function-calling settings reject before effects, and two correctly configured
retained executors still run. The stale-setting tests failed before the repair
and pass afterward on both released executors.

Tests also cover forced emitter errors, malformed transforms, sticky denial,
agent-policy isolation, parallel tool-call correlation, bounded audit retention,
concurrent admission, noncooperative callback cleanup failure and worker shutdown.
Model responses in deterministic tests are explicitly stubbed; these tests do
not claim to demonstrate real model behavior.

## Actual model evidence

The [six receipts](example-runs/) come from actual example subprocesses calling
local Qwen through a no-proxy, no-redirect HTTP client. They cover allowed,
blocked, approved, rejected, EOF and multi-agent cases.
Approval input was piped and is labeled as such—not claimed human participation.
Each receipt records model digest, Ollama version, request count and SDK decisions.

## Review status

Two read-only architecture/code/security review rounds drove the enforcement
regressions above. The second review confirmed the iteration guards, legacy
argument handling, task-tool inventory and crew-skill rejection; it then identified
the reused-executor setting gap. That final admission repair passed the full
deterministic suite and all six actual-model cases, but **has not received another
independent review**. No production/release approval is claimed. Both that review
step and the documented upstream dependency audit must be cleared before release.

## Reproduce

```bash
uv sync --locked --group dev
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync pytest -m 'not e2e'
LOCAL_MODEL_REQUIRED=1 uv run --no-sync pytest -m e2e
uv run --no-sync bandit -r packages/crewai/src examples -q
uv run --no-sync pip-audit
uv build --package agent-hooks-crewai
```

`pip-audit` is expected to fail until the documented upstream dependency
advisories have a compatible remediation. Do not suppress that result to
represent this local alpha as release-approved.
