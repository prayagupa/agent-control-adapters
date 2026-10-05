# agent-hooks-crewai

An opt-in, execution-scoped adapter for **CrewAI 1.15.23** and the published
**agent-hooks-sdk 0.1.0a5**. Requires Python 3.11-3.13. It does not patch CrewAI,
register global hooks, or require a local SDK fork.

## Public API

Prefer the [shared API](../core): import `Backend`, `with_agent_hooks`, `RunRequest`,
`AgentPolicy`, `AdapterBlocked`, and `AdapterCleanupError` from
`agent_hooks_adapters`. Select `backend=Backend.CREWAI`, pass the actual Crew as
`runtime`, and bind application agent labels through the trusted `agents` map.
Call `adapter.run(request)` for normalized text and audit records.
`create_adapter` remains a compatible alias. The
[complete hook examples](../../examples/crewai) adapt CrewAI's native
response-sanitization, tool-safety, approval and agent-scoping scenarios.

For backward compatibility, import `CrewAIAdapter`, native `AgentPolicy`,
`AdapterBlocked`, and `AdapterCleanupError` from `agent_hooks_crewai`.
The native policy class still accepts an actual CrewAI Agent, and `kickoff()`
still returns its original `CrewOutput`. Native error classes also inherit the
shared error categories, preserving CrewAI's `HookAborted` behavior.

```text
CrewAIAdapter(
    crew: Crew,
    *,
    interceptors: Sequence[Interceptor],
    agent_policies: Sequence[AgentPolicy] = (),
    resolver: ApprovalResolver | None = None,
    timeout: float = 5,
    emit_timeout: float = 30,
    max_records: int = 1000,
)

AgentPolicy(agent: Agent, interceptors: tuple[Interceptor, ...])

adapter.kickoff(inputs: dict[str, object] | None = None) -> CrewOutput
adapter.run(request: RunRequest | None = None) -> RunResult
adapter.records -> tuple[InterceptionRecord, ...]
adapter.records_dropped -> int
adapter.cleanup_error -> AdapterCleanupError | None
```

The signatures above describe the interface, not a runnable model example.
See the repository's [real local-model examples](../../examples/crewai) and
[architecture documentation](../../docs) for complete applications.

Each adapter is **single-use**, including failed admission or a denied execution.
`run()` and `kickoff()` share the same execution guard and cleanup scope.
Create a new adapter for the next run. Use independent Crew objects for concurrent
runs; concurrently reusing one Crew is unsupported. Shared interceptors must be
nonempty. Agent bindings must be nonempty tuples, unique, and reference actual
agents in the crew. Role strings are never used for authorization.

Crew-level events use shared controls. Model/tool events use the shared controls
followed by that agent's controls. SDK `sequential/run_all` composition ensures an
approval request cannot skip a later control. A final deny cannot be lifted by the
approval resolver. A failing SDK transform can short-circuit the remaining controls
with a denial, as specified by the SDK.

## Supported execution profile

Only synchronous `Crew.kickoff()` on an ordinary sequential Crew with ordinary
Agent and Task objects is supported. Both released executor implementations are
supported: `AgentExecutor` and `CrewAgentExecutor`. Native parallel tool batches
are supported, with serialized governance emissions. Only text model messages
and JSON-compatible native function-call history are supported.

Admission rejects streaming, checkpoints/resumption, kickoff/step/task/agent
callbacks, planning/reasoning, memory, knowledge retrieval, task/agent guardrails,
human task input, delegation, asynchronous tasks, hierarchical crews, custom
crew/agent/task/executor subclasses, code execution, platform apps, MCP, skills,
agent execution-timeout wrappers, and task output files. These options have
additional execution or output paths outside this initial profile. Configuring
an object can itself have effects before the adapter receives it; admission only
guards the subsequent kickoff.

**Set `respect_context_window=False` on every Agent.** The default automatic
summarization makes internal model calls outside the public model hooks. Configured
`function_calling_llm` on either the Crew or an Agent is also rejected because its
tool-argument repair calls are uncovered. Tasks must leave `output_json`,
`output_pydantic`, and `response_model` unset: structured conversion and repair
likewise bypass the governed model path.
Admission also checks these two forbidden settings on any retained agent executor:
clearing the Agent/Crew settings does not refresh the executor's old summarization
or `function_calling_llm` configuration. A retained executor with neither setting
enabled remains supported.

**One iteration is reserved for safe stopping.** Before each model request, the
adapter verifies the actual admitted executor, its iteration counter, and the
unchanged configured `max_iter`. It blocks when `iterations >= max_iter - 1`,
before the pinned runners can enter their unhooked forced-final request.
Thus `max_iter=1` permits no model request, `max_iter=2` permits at most one, and
`max_iter=3` permits at most two per task attempt. Nonfinal answers and tool loops
that exhaust this governed allowance raise `AdapterBlocked`; they do not receive
an extra model-generated final answer. These host refusals produce a diagnostic,
not a fabricated SDK model-call record.

Admission/configuration errors raise `ValueError` or `TypeError`. Denials,
malformed hook payloads or transforms, SDK errors, admission saturation during
execution, and policy timeouts raise the constant-message `AdapterBlocked`
(a CrewAI `HookAborted` subclass). Denial is sticky for the run, even when CrewAI
swallows a pre-tool or parallel-tool abort. No replacement "blocked" output is
returned as a successful CrewOutput.

| CrewAI seam | SDK point | Supported target/write-back |
| --- | --- | --- |
| Execution start | `agent_startup` | Agent- and task-owned declared tools; registration changes are rejected |
| Input | `input` | `$target.content`: JSON input object; role stays `user` |
| Before model | `pre_model_call` | `$target`: validated text-message list, updated in place |
| After model | `post_model_call` | `$target.content`: text; no invented structured calls |
| Before tool | `pre_tool_call` | `$target`: JSON arguments, updated in place |
| After tool | `post_tool_call` | `$target`: text result |
| Output | `output` | `$target.content`: raw text; retains CrewOutput and updates the final **nonempty** task raw |
| Execution end | `agent_shutdown` | Notification/audit only; no target write-back or approval |

Disable tool caching when transforming tool arguments, for example with
`cache=False` on both the Agent and Crew. If a tool cache is attached, argument
changes are rejected: the native runner may already have looked up a result
using the original arguments. Other tool decisions and post-tool text transforms
remain supported with caching.

The legacy `CrewAgentExecutor` passes a different argument dictionary to its
native tool when the original arguments are empty. The adapter therefore rejects
any changed pre-tool target for initially empty arguments on that executor,
before tool execution. Nonempty argument transforms remain supported on both
executors; empty-argument transforms remain supported on `AgentExecutor`.

Structured-output tasks are rejected at admission, rather than claiming coverage
of their internal repair calls. Defensive write-back checks also reject changes
to any unexpected JSON/Pydantic/model-structured output to avoid stale alternate
views. CrewAI's structured native model tool-call response **bypasses the
post-model hook**. The subsequent pre-tool boundary is governed; this adapter
does not claim complete structured post-model or full SDK conformance coverage.
Task/flow step events are not mapped to SDK events.

## Isolation, deadlines, and audit

The wrapper installs public `scoped_hooks` only for the kickoff lifetime, preserves
existing scoped hooks, and restores the previous scope on exit. Every mapped
event must carry the admitted Crew object; model/tool events must also carry an
admitted Agent. Missing or foreign identities, including nested crews, fail closed.

One private daemon thread owns one persistent asyncio loop, unique SDK builders
and emitters for the crew and every agent, and at most one running emission.
There are at most eight admitted waiting caller threads. Reentry on the policy
worker fails immediately. `emit_timeout` is a wall-clock bound including waiting
for the worker; `timeout` is the SDK timeout for each interceptor/resolver call.
Use finite positive values; `max_records` must be between 1 and 100,000.

A wall timeout poisons the worker: later emissions are rejected and late results
are never applied. Cooperative async callbacks are cancelled and drained.
**Python cannot terminate a noncooperative callback thread.** Cleanup joins for
at most `min(1 second, emit_timeout)`, retains the occupied worker until actual
exit, and reports `AdapterCleanupError` if it remains alive. If a kickoff exception
already exists, it is preserved, the cleanup error is available through
`cleanup_error`, and a constant cleanup note is attached to the original exception.
Otherwise the cleanup error is raised. Use process isolation for untrusted or
noncooperative policy code.

Audit storage is a bounded snapshot of actual SDK records; older records are
evicted and counted in `records_dropped`. Emissions have unique per-binding session
UUIDs and per-session sequences. A common host-owned `trace.trace_id` correlates
the adapter's SDK records and structured diagnostic logs. User input keys such as
`session` remain nested content and cannot overwrite the trusted envelope.

Pre/post tool and model IDs are paired by host thread, actual agent, and tool/model
name, never by role or reused context object. Missing post events are not invented.
Unmatched posts receive an explicitly `unpaired-`-prefixed ID and a host diagnostic.
Correlation storage is capped at 1,024 pending keys and cleared after every run.

Each emission crosses a strict, detached JSON boundary: cycles, arbitrary Python
objects, nonfinite floats, integers outside the interoperable 53-bit range,
unknown envelope/message keys, depth over 32, more than 10,000 nodes, and more
than 1 MiB of UTF-8 string content are rejected. This bound also applies to the
complete SDK context, which includes copies of the target.

Shutdown verdicts are recorded without undoing a completed action. Shutdown
engine failures preserve any original exception. If timeout poisoning prevents
shutdown, `adapter_shutdown_not_emitted` is logged; no synthetic SDK record fills
that evidence gap. A host-side mapping refusal can follow an SDK permit record:
the diagnostic, not a fabricated SDK decision, explains that the host refused
to apply it. Logs contain event categories, trace IDs, points, and error types,
not prompts or tool results.

## Security boundary

This is a governance integration, **not a sandbox or an authoritative last
governor**. Trusted host hooks can run before or after it, and arbitrary Python
callbacks/tools can use side channels. Already-running native parallel tools
cannot be rolled back when another call is denied. Output interception does not
erase original content from framework telemetry, internal history, or caches.
Policies, tools, and custom model clients need separate trust and process/network
isolation appropriate to the application.

Unit and integration tests use real CrewAI kickoff and the released SDK emitter;
only model network I/O is stubbed. Live Ollama examples and e2e coverage belong
to the repository-level example/test suite, not this package's deterministic tests.
