# agent-hooks-adapters

Small, independently packaged integrations between
[agent-hooks](https://github.com/responsibleai/agent-hooks),
[Agent Control Specification (ACS)](https://pypi.org/project/agent-control-spec/),
and agent frameworks. Use native Python interceptors or YAML/Rego controls
without changing the framework integration. Frameworks do not need to change
their core.

**First adapter:** `agent-hooks-crewai` (local alpha; not published by this workspace).
Only CrewAI is implemented. Future LangGraph, Google ADK, and OpenAI Agents SDK
adapters should be separate packages, not dependencies of this one.

**Release status:** an upstream CrewAI/Chroma dependency audit is currently
blocked. See [the security notice](SECURITY.md#known-dependency-release-blocker);
working local examples are not a production security approval.

```mermaid
flowchart LR
    App["with_agent_controls: your existing crew"] --> Adapter["agent-hooks-crewai"]
    Adapter --> CrewAI["CrewAI public hooks"]
    Adapter --> SDK["agent-hooks-sdk"]
    SDK --> Policy["Python interceptors or native ACS interceptor"]
    Policy --> YAML["Optional YAML manifest and Rego bundle"]
```

## Start here

Requires Python 3.11-3.13 and `uv`. Run from this directory:

```bash
uv sync --group dev
uv run --no-sync python -m examples.crewai.agent_hooks.llm_hooks
```

The examples require an **already running** Ollama on `localhost:11434` with
`qwen2.5:latest` installed. They never download a model, start a service, or fall
back to a remote provider. See [the example guide](examples/crewai/README.md).
For ACS examples and the complete dual-source development gate, install the
optional group with `uv sync --locked --group dev --group acs`.

| Example | What it demonstrates | Command |
|---|---|---|
| [LLM hooks](examples/crewai/agent_hooks/llm_hooks.py) | Sanitize a real model response using an SDK transform | `uv run --no-sync python -m examples.crewai.agent_hooks.llm_hooks` |
| [Tool hooks](examples/crewai/agent_hooks/tool_hooks.py) | Block a model-requested tool before execution | `uv run --no-sync python -m examples.crewai.agent_hooks.tool_hooks` |
| [Human approval](examples/crewai/agent_hooks/human_approval.py) | Approve or reject a model-requested message preview | `uv run --no-sync python -m examples.crewai.agent_hooks.human_approval` |
| [Multi-agent LLM hooks](examples/crewai/agent_hooks/multi_agent_llm_hooks.py) | Researcher and editor have separate LLM response-limit policies | `uv run --no-sync python -m examples.crewai.agent_hooks.multi_agent_llm_hooks` |
| [ACS tool controls](examples/crewai/acs/tool_hooks.py) | The same tool denial, configured with real YAML/Rego | `uv run --no-sync python -m examples.crewai.acs.tool_hooks` |
| [ACS multi-agent LLM hooks](examples/crewai/acs/multi_agent_llm_hooks.py) | Shared YAML plus separate researcher/editor LLM response limits | `uv run --no-sync python -m examples.crewai.acs.multi_agent_llm_hooks` |

Each example includes the full application using CrewAI's built-in local-model
provider, policies, and explicit agent/crew setup. The four Python-policy examples
in `agent_hooks/` are standalone files; copy the entire `acs/` folder to keep its
Python examples, manifests and Rego bundle together. All use the shared
`with_agent_controls(...).run(RunRequest(...))` API. Their scenarios replace
CrewAI's native LLM/tool-hook examples with SDK policies, not a new workflow
framework. See the [native-hook migration guide](examples/crewai/README.md).
Model provenance and detailed receipts are collected by the tests.

## Application API

Install the shared API and CrewAI backend with
`pip install 'agent-hooks-adapters[crewai]'`, or select `[crewai,acs]` for YAML
controls. These are distribution extras; this workspace is still unpublished.
Given your existing `crew`,
`researcher`, and framework-neutral policy instances:

Configure agents with `respect_context_window=False` and `max_iter >= 2`, and
use text-output tasks. Automatic summarization and structured-output repair can
make model calls outside public hooks, so this alpha refuses those paths.

```python
from agent_hooks_adapters import AgentPolicy, Backend, RunRequest, with_agent_controls

agent_control = with_agent_controls(
    backend=Backend.CREWAI,
    runtime=crew,
    interceptors=[shared_policy],
    agents={"researcher": researcher},
    agent_policies=[AgentPolicy("researcher", (researcher_policy,))],
)
result = agent_control.run(RunRequest(inputs={"question": "Explain unit tests."}))
output = result.output
records = result.records
```

To switch the policy source, replace `interceptors=[shared_policy]` with
`policy_file="controls.yaml"`. For an agent, replace
`AgentPolicy("researcher", (researcher_policy,))` with
`AgentPolicy("researcher", policy_file="researcher.yaml")`. The rest of the
integration stays the same. Each assignment accepts exactly one source, and
shared/agent sources may be mixed. YAML files use the native ACS manifest
grammar and may reference Rego files; they are not a second Python interceptor
implementation. The [complete ACS examples](examples/crewai/README.md#acs-yamlrego-controls)
include every required asset.

Shared policies run before the selected agent's policies. Bind policies to the
trusted `agents` mapping, not a model-supplied role name or request field.
Application labels stay stable while each backend resolves actual native agent
objects; SDK identities are not replaced by those labels. An adapter is single-use;
create a new one for the next run. Call **`agent_control.run()`**, not `crew.kickoff()`,
to enter the governed execution scope. Governance denial raises the shared
`AdapterBlocked` error instead of returning a successful result.

Only the CrewAI backend is implemented in this change. An OpenAI Agents backend
can implement the same request/result, policy assignment and error contracts,
but native workflow setup and hook coverage must be verified separately.
`Backend.CREWAI` is currently the only enum member; raw strings and invalid
backend selectors are rejected. The existing
`agent_hooks_crewai.CrewAIAdapter.kickoff()` API remains compatible for callers
that need the native `CrewOutput`. `with_agent_hooks` and `create_adapter` are
exact compatibility aliases of `with_agent_controls`.

ACS is optional and pinned to `agent-control-spec==0.4.0a4`; the existing SDK
stays at `0.1.0a5`. Manifest parsing is eager, but Rego artifacts load lazily at
their first bound event. Earlier agent/model/tool effects are not rolled back.
The released Cedar engine expects legacy metadata and fails closed with this
adapter's canonical contexts; use the tested YAML/Rego profile. See
[ACS coverage and limitations](docs/coverage.md#acs-policy-source).

**Support is intentionally explicit:** synchronous sequential crews on the
pinned CrewAI release. The adapter rejects unsupported execution modes instead
of silently claiming coverage. It is not a sandbox and does not make arbitrary
tools or Python policies safe. Read [coverage and limitations](docs/coverage.md)
before using it for enforcement.

## Repository layout

```text
packages/crewai/        Installable adapter, typed source, unit/integration tests
packages/core/          Shared run/request/result API and backend selection
examples/crewai/agent_hooks/  Four complete native Python-policy examples
examples/crewai/acs/          Two complete YAML/Rego examples and all policy assets
tests/                 Example e2e and dependency conformance checks
docs/                  Architecture, coverage, and actual example receipts
```

No empty framework scaffolds or shared runtime abstraction are included. Add
another package only when implementing another adapter.

## Develop

```bash
uv sync --locked --group dev --group acs
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync pytest -m 'not e2e'
LOCAL_MODEL_REQUIRED=1 uv run --no-sync pytest -m e2e
uv build --package agent-hooks-crewai
uv build --package agent-hooks-adapters
```

Default CI runs deterministic checks without downloading models. Local-model
tests are a separate required evidence tier; a skip is not a pass.
Native-only development can omit `--group acs` and select
`pytest -m 'not e2e and not acs'`; that is not the complete dual-source gate.
ACS 0.4.0a4 has a Linux x86_64 wheel requiring glibc 2.34 or newer. Other
platforms need its Rust source build; see [contributing](CONTRIBUTING.md).

See [architecture](docs/architecture.md), [contributing](CONTRIBUTING.md), and
[validation evidence](docs/validation.md). Read the [security scope](SECURITY.md)
before deployment. Licensed under [MIT](LICENSE).
