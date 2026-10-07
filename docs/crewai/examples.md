# CrewAI examples

Six complete applications keep native CrewAI setup in the script and add
`with_agent_controls` at the execution boundary. Policies use either native
Agent Hooks or the published ACS runtime; neither folder depends on a sibling
example.

**These are runnable source examples, not current acceptance receipts.** The
latest real-model refresh is incomplete, including a human-approval timeout.
Read [validation](../validation.md) and the
[dependency warning](../../SECURITY.md#known-dependency-release-blocker).

## Choose an example

| Behavior | Agent Hooks | ACS |
|---|---|---|
| Shared model controls | [llm_hooks.py](../../examples/crewai/agent_hooks/llm_hooks.py) | Not included |
| Shared/per-agent model controls | [multi_agent_llm_hooks.py](../../examples/crewai/agent_hooks/multi_agent_llm_hooks.py) | [multi_agent_llm_hooks.py](../../examples/crewai/acs/multi_agent_llm_hooks.py) |
| Tool decisions and validated transforms | [tool_hooks.py](../../examples/crewai/agent_hooks/tool_hooks.py) | [tool_hooks.py](../../examples/crewai/acs/tool_hooks.py) |
| Explicit human approval | [human_approval.py](../../examples/crewai/agent_hooks/human_approval.py) | Not included |

Applications bind policy labels to actual agents. Shared controls precede
agent-specific controls. Tools remain ordinary trusted CrewAI tools, not
adapter-defined workflow abstractions.

## Run locally

Use the [source installation](README.md#install-from-source), then an
operator-managed Ollama service at `http://localhost:11434/v1` with the existing
`qwen2.5:latest` model.

```bash
uv run --no-sync python examples/crewai/agent_hooks/llm_hooks.py
uv run --no-sync python examples/crewai/agent_hooks/tool_hooks.py
uv run --no-sync python examples/crewai/agent_hooks/human_approval.py
uv run --no-sync python examples/crewai/agent_hooks/multi_agent_llm_hooks.py

uv run --no-sync python examples/crewai/acs/tool_hooks.py
uv run --no-sync python examples/crewai/acs/multi_agent_llm_hooks.py
```

Scripts intentionally use fixed local provider settings and bounded transports.
They do not substitute canned model responses. Do not claim a tool-control
example passed unless the real model proposed a valid call and the expected
governed effect occurred. Follow operator-approved run limits; do not silently
change models, prompts, timeouts, or assertions to manufacture acceptance.

The human-approval example uses an **unauthenticated loopback demo service**.
It is not production approval infrastructure. For deployment, use a trusted,
authenticated resolver and never treat a timeout or missing answer as approval.

## Complete ACS assets

Keep the actual files rather than copying policy excerpts from this guide:

| Asset | Purpose |
|---|---|
| [tool_hooks.yaml](../../examples/crewai/acs/tool_hooks.yaml) | Tool-control manifest |
| [shared.yaml](../../examples/crewai/acs/shared.yaml) | Shared lifecycle, input/output, model, and tool bindings |
| [researcher.yaml](../../examples/crewai/acs/researcher.yaml) | Researcher model/tool bindings |
| [editor.yaml](../../examples/crewai/acs/editor.yaml) | Editor model/tool bindings |
| [controls.rego](../../examples/crewai/acs/policy/controls.rego) | Referenced native Rego policies |

Missing ACS bindings deny; shutdown is audit-only. The scripts construct native
`AcsInterceptor` instances from paths relative to `Path(__file__).parent`, then
pass them through `interceptors`. This differs from the convenience
`policy_file` loader's pre-load validation; see [shared API](../api.md).

To copy an example outside the repository:

- Install core, CrewAI, and the optional ACS dependencies in the destination
  environment.
- Copy the selected script; for ACS, copy the **whole ACS folder**, including
  every YAML file and the `policy` directory.
- Keep filenames/layout intact. No `examples` or `tests` imports are required.
- Keep application policies, tool implementations, and approval configuration
  trusted. The adapter is not a sandbox.

## What changes from native hooks

Build the same native Crew, agents, tasks, tools, and local model client. Instead
of registering process-global policies or calling bare `Crew.kickoff`, create a
single-use governed adapter and call its `run`. Stable labels are supplied by the
application, not inferred from role text.

On success, inspect `RunResult.output`, actual `records`, and `records_dropped`.
On denial, handle `AdapterBlocked`; do not parse a blocked message as successful
output. Handle `AdapterCleanupError` separately and inspect `cleanup_error`.
Completed effects are not rolled back.

See the [supported profile and hook gaps](README.md#supported-profile) before
adding features. In particular, automatic context summarization, configured
function-calling repair models, structured-output repair, and the native
forced-final iteration are not permitted execution shortcuts.

## Evidence

[Historical example receipts](../example-runs/README.md) record real local-model
runs on earlier snapshots. They are not approval for the current source.
Deterministic tests may mock only the model boundary; those tests do not prove
current provider compatibility or production safety.
