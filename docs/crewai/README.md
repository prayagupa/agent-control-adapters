# CrewAI adapter

Govern a native CrewAI workflow with Agent Hooks interceptors or ACS YAML/Rego
policies. The adapter uses CrewAI's public hooks; it does not replace the agents,
tasks, tools, or model client.

**Unpublished alpha.** CrewAI's Chroma dependency has unresolved critical/high
advisories. See the [security notice](../../SECURITY.md#known-dependency-release-blocker).
Current real-model acceptance is also incomplete; see [validation](../validation.md).

## Install from source

Requires Python **3.11-3.13**, CrewAI **1.15.23**, and Agent Hooks SDK **0.1.0a5**.
ACS support uses **agent-control-spec 0.4.0a4**.

```bash
git clone https://github.com/prayagupa/agent-control-adapters.git
cd agent-control-adapters
uv sync --frozen --group acs
```

Use this environment only for core/CrewAI. Other adapters have independent
environments. Distributions are not yet available on PyPI.

## Use the shared API

Create your Crew normally, then pass it to `with_agent_controls` with
`Backend.CREWAI`. Choose exactly one shared policy source: `interceptors` or
`policy_file`. The [shared API](../api.md) describes assignments, requests,
results, and errors; [complete examples](examples.md) provide runnable programs.

- `agents` binds stable application labels to actual agents in the Crew.
  Policies never authorize by role text or model-provided identifiers.
- Shared controls run first; an agent's extra controls follow on model/tool events.
  Crew lifecycle, input, and final output use shared controls.
- Every adapter is **single-use**, including failed or denied runs. Create a new
  adapter for another run, and use independent Crews for concurrent execution.
- `run(RunRequest(...))` returns normalized text and actual SDK records.
  The framework-specific `CrewAIAdapter.kickoff()` retains `CrewOutput`.
- A denial raises `AdapterBlocked`, never a success-shaped replacement answer.
  Records remain inspectable after failure.

The direct `CrewAIAdapter` accepts native interceptor sequences and native
agent-object policy bindings. It shares the same admission, execution guard,
record retention, and cleanup scope as the portable factory.

## Supported profile

Only synchronous `Crew.kickoff()` on ordinary sequential Crew, Agent, and Task
objects is covered. Both released executors, `AgentExecutor` and
`CrewAgentExecutor`, are supported. Native parallel tool batches have serialized
governance emissions. Model history must contain text messages or JSON-compatible
native function calls.

**Set `respect_context_window=False` on every Agent.** Automatic summarization
makes unhooked model requests. Leave Crew/Agent `function_calling_llm` unset and
Task `output_json`, `output_pydantic`, and `response_model` unset: repair/conversion
calls also bypass these hooks. Retained executors are checked separately;
clearing an Agent setting does not clear stale executor configuration.

Admission rejects streaming, resumption/checkpoints, kickoff/step/task/agent
callbacks, planning/reasoning, memory, knowledge, task/agent guardrails, human task
input, delegation, async tasks, hierarchical crews, custom runtime subclasses,
code execution, platform apps, MCP, skills, agent execution-timeout wrappers, and
task output files. Construction effects that happened before wrapping are not
governed.

**One iteration is reserved for safe stopping.** Execution blocks before
`iterations >= max_iter - 1` enters CrewAI's unhooked forced-final request.
`max_iter=1` allows no model request; `2` allows at most one; `3` at most two per
task attempt. Exhaustion raises `AdapterBlocked`, not an extra generated answer.
The actual executor, counter, and unchanged configured limit are checked before
each request. Host refusals are diagnostics, not invented SDK records.

## Hook mapping

| Native boundary | SDK point | Governed write-back |
|---|---|---|
| Start | `agent_startup` | Declared tools; registration changes are rejected |
| Input | `input` | Strict JSON input object; role remains `user` |
| Before model | `pre_model_call` | Validated text-message list |
| After model | `post_model_call` | Text, not invented structured calls |
| Before tool | `pre_tool_call` | JSON arguments |
| After tool | `post_tool_call` | Text result |
| Final output | `output` | Raw text and the final nonempty task raw |
| End | `agent_shutdown` | Audit-only; no write-back or approval |

Disable caching on both Crew and Agent when transforming tool arguments.
Cached lookups may already use the original arguments, so changed arguments are
rejected with an attached cache. Other tool decisions and post-tool text
transforms remain supported.

For `CrewAgentExecutor`, initially empty tool arguments cannot be transformed:
the native runner passes a different dictionary to the tool. Nonempty transforms
work on both executors; empty-argument transforms work on `AgentExecutor`.

Structured native tool-call responses bypass CrewAI's post-model hook. Their
subsequent pre-tool boundary is governed, but complete structured post-model
coverage and full SDK conformance are **not** claimed. Task/flow steps are not
mapped. Unexpected structured final output cannot be transformed into stale
alternate views.

## Deadlines, records, and cleanup

Policy checks default to `timeout=5`, the whole SDK emission to
`emit_timeout=30`, and retained records to `max_records=1000`. Governance uses
bounded worker admission and a persistent owned async bridge; denial is sticky
even if CrewAI swallows a hook abort. Overload, invalid payloads/transforms, SDK
errors, and governance timeouts fail closed.

These limits do not impose a total native model/tool deadline or kill Python
callbacks. Configure transport/tool limits and process isolation separately.
Timed-out callbacks retain their worker slot until they finish. Cleanup cancels
and drains owned async work; unfinished work raises `AdapterCleanupError` and is
also available through `cleanup_error`. `records_dropped` counts bounded-retention
overflow. SDK shutdown is an audit notification, not rollback.

For ownership, threat boundaries, and alternatives, see the
[architecture](architecture.md), [shared coverage](../coverage.md), and
[security policy](../../SECURITY.md).
