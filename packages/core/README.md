# agent-hooks-adapters

A small, typed execution API for agent-hooks backends. It does not define agents,
tasks, tools, models, or workflows. Applications keep that setup in their native
framework and pass the resulting runtime into `with_agent_hooks`.

Install the CrewAI backend with `pip install 'agent-hooks-adapters[crewai]'`.
Import `Backend` from `agent_hooks_adapters` and select `Backend.CREWAI`, the only
implemented member. The factory requires an enum member and raises
`UnsupportedBackendError` for raw strings, invalid selectors, or a missing backend
installation, rather than falling back to another framework.

## Public contract

```text
with_agent_hooks(
    *,
    backend: Backend,
    runtime: object,
    interceptors: Sequence[Interceptor],
    agents: Mapping[str, object] | None = None,
    agent_policies: Sequence[AgentPolicy] = (),
    resolver: ApprovalResolver | None = None,
    timeout: float = 5,
    emit_timeout: float = 30,
    max_records: int = 1000,
) -> Adapter

Backend.CREWAI  # value: "crewai"
AgentPolicy(agent_id: str, interceptors: tuple[Interceptor, ...])
RunRequest(inputs: dict[str, object] | None = None)
adapter.run(request: RunRequest | None = None) -> RunResult
RunResult(output: str, records: tuple[InterceptionRecord, ...], records_dropped: int)
```

This is an interface description, not a model-output demonstration. See the
[complete real-model examples](../../examples/crewai).
`create_adapter` remains an alias with the same signature and behavior.
Wrapping does not mutate the native runtime or register process-global hooks;
call the returned adapter's `run` method to enter its governed scope.

`agents` is trusted configuration mapping stable application labels such as
`researcher` to actual native agent objects. Policies refer to those labels;
backends validate ownership and bind the controls to the objects, never to role
text or request/model-provided IDs. Labels are 1-128 ASCII letters, digits,
underscores, dots, or dashes. Unknown labels, foreign agents and duplicate
bindings are rejected.

Shared policies run before an agent's additional policies on agent-scoped
events. Event coverage remains a backend capability, not a promise that all
frameworks expose the same hooks. CrewAI currently applies agent policies to
model/tool events and shared policies to crew lifecycle, input and final output.

Adapters are single-use. `RunRequest` captures detached strict JSON inputs with
the existing depth, node, number and one-MiB UTF-8 bounds. A success contains
normalized text and actual SDK audit records. A governance denial raises
`AdapterBlocked`; it is never represented as successful output. Audit records
remain available on the adapter after failure. `AdapterCleanupError` and
`cleanup_error` expose unfinished governance work.

Backend selection and initialization are the only framework-dependent layer.
Native workflow construction and unsupported feature handling are not hidden
or automatically translated. Framework-neutral policies should not depend on
backend-specific role text or the exact native agent UUID.
