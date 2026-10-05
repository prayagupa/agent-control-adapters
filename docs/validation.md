# Validation evidence

## Private source-checkpoint preflight - October 5, 2026

The full local non-e2e suite passed **503 tests**, with the 16 model cases
deliberately deselected, in 9.29 seconds. Whole-repository Ruff lint/format
(39 Python files), strict Pyright, Bandit, offline lock consistency and both
packages' wheel/sdist builds passed. Real-model evidence is recorded in the
rename and folder-layout sections below; it was not replaced with mocked output.

The fresh installed-environment audit still fails on exactly the four known
Chroma 1.1.1 advisories, with no fixed versions. The two unpublished local
packages receive source/static review rather than a claimed PyPI audit pass.
The user approved a private source checkpoint to the existing `origin/main`,
which GitHub resolves to `prayagupa/agent-control-adapters`; this is not
production/release approval or an audit waiver. CI's security check remains
unsuppressed. Exact-commit remote CI must be reported separately after the push.

## Multi-agent LLM-hook names - October 5, 2026

The native [LLM-hook example](../examples/crewai/agent_hooks/multi_agent_llm_hooks.py)
and [ACS counterpart](../examples/crewai/acs/multi_agent_llm_hooks.py) are named
`multi_agent_llm_hooks.py`: they enforce 1,600/800-character per-agent LLM response
limits, not a tool-call workflow. Their application and policy bodies are unchanged.
Module commands, imports, recorder choices and fresh receipt names use the new names;
historical captured receipts retain their original names.

The rename recheck passed **84 focused safety/integration tests** and **4 required
real-Qwen e2e cases**, none skipped, covering both examples as modules and copied
scripts/folders. The e2e selector deliberately deselected 12 unaffected cases and
finished in 80.37 seconds. Scoped Ruff/format and strict CLI Pyright passed.
The earlier full 16-case folder-layout run below predates this filename change.

```bash
uv run --no-sync --offline --no-python-downloads pytest -q tests/test_example_safety.py
LOCAL_MODEL_REQUIRED=1 LOCAL_MODEL=qwen2.5:latest \
  uv run --no-sync --offline --no-python-downloads pytest -q -m e2e \
  tests/test_examples_e2e.py -k multi_agent_llm_hooks
```

## Example folders and runner naming - October 5, 2026

The four native Python-policy scripts are now in
[agent_hooks](../examples/crewai/agent_hooks); the two ACS scripts are in
[acs](../examples/crewai/acs), beside their YAML manifests and Rego bundle.
All six use `agent_control` for the runner returned by `with_agent_controls`.
Module commands, direct-file links, the recorder and copied-folder tests use
the new paths. Old root-level scripts were removed rather than duplicated.

| Check | Layout-change result |
|---|---|
| Focused example safety/integration tests | **84 passed**, none skipped |
| Required real-Qwen e2e | **16 passed**, none skipped, in 298.93 seconds |
| Actual modes and outcomes | Module and copied-file/folder modes; 8 allowed, 8 blocked |
| Ruff lint and formatting | Passed; 12 scoped Python files formatted |
| Strict CLI Pyright | 0 errors/warnings on package sources, examples and affected tests/recorder |
| Bandit on examples | Passed |
| YAML/Rego asset integrity | All five files exactly match the reviewed implementation snapshot |
| Local documentation links | Passed |

The layout regression failed before the moves and passed afterward. The source
checks also require the `agent_control` variable name. Copied ACS examples locate
sibling manifests from `Path(__file__).parent`, independent of the working
directory. All 16 real-model receipts identify the new qualified example names;
the recorder verified the model artifact and runtime before and after each run.
Editor test discovery still returned no tests, so provisioned `pytest` supplied
the actual test evidence.

```bash
uv run --no-sync --offline --no-python-downloads pytest -q tests/test_example_safety.py
LOCAL_MODEL_REQUIRED=1 LOCAL_MODEL=qwen2.5:latest \
  uv run --no-sync --offline --no-python-downloads pytest -q -m e2e tests/test_examples_e2e.py
```

This slice did not change model prompts, framework configuration, policy assets,
package implementation or dependency pins. The implementation-level checks and
reviews below belong to the earlier snapshot, not a fresh review of these example
paths. Package builds and the unchanged dependency audit were not rerun for this
layout change; the known Chroma release blocker remains.

## Native and ACS controls - October 5, 2026

Executed locally with Python 3.13.14, CrewAI 1.15.23,
agent-hooks-sdk 0.1.0a5 and optional agent-control-spec 0.4.0a4.
The core and CrewAI adapter remain separate packages; no sibling source fork,
legacy-context shim, or alternate evaluator was installed.

| Check | Implementation result before the layout change |
|---|---|
| Focused deterministic gate | **389 passed**, none skipped |
| Covered selectors | Core contracts, portable API, actual ACS/Rego controls, all six example safety checks, worker/admission, unmodified SDK reference vectors |
| Ruff lint and formatting | Passed; 37 Python files formatted |
| Strict CLI Pyright | 0 errors/warnings, including changed source, examples, tests and recorder |
| Required real-Qwen e2e | **16 passed**, none skipped; module and copied-file/asset modes |
| Actual example outcomes | 8 allowed, 8 blocked; no denied demonstration tool body executed |
| Independent package builds | Both wheels and sdists passed |
| Bandit | Passed on both package sources and examples |
| Lock consistency | Passed; 167 packages, only ACS added |
| Installed-environment dependency audit | **Failed: four existing Chroma 1.1.1 advisories**, no fixed versions reported |
| Current remote CI/publication | Not run; this change is local, uncommitted and unpushed |
| Independent architecture/code review | Architecture sound; code scope approved, no blocker/major defect |
| Independent security review | No new findings; local stage GO WITH CONDITIONS, inherited release hold retained |

The deterministic command was a focused gate, **not a claim that every existing
non-e2e test was rerun**. Existing enforcement/admission/worker tests and
dependency reference-host vector assertions were retained. Editor test discovery
reported no tests, so the provisioned repository `pytest` command supplied the
actual evidence. Pylance's editor view returned a stale pre-change signature;
it is not treated as current compatibility proof. CLI strict typing and
executed alias/native-client tests validate the changed surface.

The code reviewer ran on host-confirmed `claude-sonnet-5.5`, which the user
explicitly accepted. Reasoning effort/context remain unverified; this is not a
max-effort-qualified release gate. The two read-only reviews assessed the exact
materialized implementation snapshot, not the prior plan. Optional feedback
clarified absolute file paths and the copied ACS asset requirement without
changing runtime behavior. Existing empty shared-list admission timing was
retained for compatibility. The contract-only clarification passed its 95 tests
and the core package was rebuilt. No review grants commit or publication approval.
The bounded reviewer follow-up closed the requested lifecycle API, dependency
metadata and test-evidence questions, and preserved code-scope approval and the
architecture-sound verdict. It did not repeat the full gate.

At the user's request, the Chroma remediation check reran the installed-environment
audit: the same four advisories remain, all with empty fixed-version lists.
Authoritative PyPI metadata reports CrewAI `1.15.23` requires `chromadb~=1.1.0`;
Chroma's latest `1.5.9` remains affected by all four current GitHub advisory ranges.
The official `crewai-core` package was also checked: it supplies shared utilities,
not the `crewai` agent/crew runtime, so it cannot replace the required framework.
No dependency change, audit suppression or safe-remediation claim was made.

The real model was `qwen2.5:latest`, artifact digest
`845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`,
on Ollama 0.34.0 at `http://localhost:11434/v1`. Requests used temperature 0,
max_tokens 256, native timeout 60 seconds and zero retries. The selected artifact
and runtime were checked before and after every example. Human decisions were
explicitly piped test input, not claimed human participation. Current receipts
were captured separately from the historical checked-in `example-runs/`.
Reproduce them with the complete command below rather than treating old receipts
as proof of the new scripts.

Real ACS cases cover all four shared/agent source combinations, same-role
agent identity isolation, configuration before manifest I/O, genuine engine
failure and destructive-tool denial, transforms, approvals, exact response
limits, native full-snapshot limits, source collection snapshots, concurrent
admission, audit caps and cleanup. Missing artifacts are tested both at startup
and after an earlier agent/model request. Shutdown is tested as audit-only.
Cedar's released legacy-envelope expectation remains an explicit fail-closed
limitation, not a supported success path.

```bash
uv sync --locked --group dev --group acs
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync pytest -q \
  packages/core/tests/test_contracts.py \
  packages/crewai/tests/test_portable_api.py \
  packages/crewai/tests/test_acs_controls.py \
  tests/test_example_safety.py \
  packages/crewai/tests/test_worker.py \
  packages/crewai/tests/test_admission.py \
  tests/test_sdk_conformance.py
LOCAL_MODEL_REQUIRED=1 LOCAL_MODEL=qwen2.5:latest \
  uv run --no-sync pytest -q -m e2e tests/test_examples_e2e.py
uv run --no-sync bandit -r packages/core/src packages/crewai/src examples -q
uv lock --check
uv run --no-sync pip-audit
uv build --package agent-hooks-adapters
uv build --package agent-hooks-crewai
```

Only Python 3.13 and this local aarch64 source-build environment were executed.
CI configuration installs the ACS group on Python 3.11; no current remote CI,
all-platform certification, Mermaid image rendering or production/release
approval is claimed. The unsuppressed dependency audit remains blocked; see
[SECURITY](../SECURITY.md#known-dependency-release-blocker).

## Historical baseline - October 2, 2026

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
