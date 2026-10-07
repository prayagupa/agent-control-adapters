# Shared control API

A small, typed execution API for native Agent Hooks and YAML-based ACS controls.
It does not define agents,
tasks, tools, models, or workflows. Applications keep that setup in their native
framework and pass the resulting runtime into `with_agent_controls`.

Install from source using the [adapter guide](crewai/README.md).
The `agent-hooks-adapters` distribution declares `[crewai]` and `[acs]` extras;
`[crewai,acs]` selects both the backend and the optional, pinned
`agent-control-spec==0.4.0a4` runtime. Adapter distributions are not yet published
to PyPI.
Import `Backend` from `agent_hooks_adapters` and select `Backend.CREWAI` for this source checkpoint. The factory requires an enum member and raises
`UnsupportedBackendError` for raw strings, invalid selectors, or a missing backend
installation, rather than falling back to another framework.

## Public contract

```text
with_agent_controls(
    *,
    backend: Backend,
    runtime: object,
    interceptors: Sequence[Interceptor] | None = None,
    policy_file: str | Path | None = None,
    agents: Mapping[str, object] | None = None,
    agent_policies: Sequence[AgentPolicy] = (),
    resolver: ApprovalResolver | None = None,
    timeout: float = 5,
    emit_timeout: float = 30,
    max_records: int = 1000,
) -> Adapter

Backend.CREWAI  # value: "crewai"
AgentPolicy(
    agent_id: str,
    interceptors: tuple[Interceptor, ...] = (),
    policy_file: str | Path | None = None,
)
RunRequest(inputs: dict[str, object] | None = None)
agent_control.run(request: RunRequest | None = None) -> RunResult
RunResult(output: str, records: tuple[InterceptionRecord, ...], records_dropped: int)
```

This is an interface description, not a model-output demonstration. See the
[complete real-model examples](crewai/examples.md).
`with_agent_hooks` and `create_adapter` remain exact aliases with the same
signature and behavior.
Wrapping does not mutate the native runtime or register process-global hooks;
call the returned adapter's `run` method to enter its governed scope.

`agents` is trusted configuration mapping stable application labels such as
`researcher` to actual native agent objects. Policies refer to those labels;
backends validate ownership and bind the controls to the objects, never to role
text or request/model-provided IDs. Labels are 1-128 ASCII letters, digits,
underscores, dots, or dashes. Unknown labels, foreign agents and duplicate
bindings are rejected.

Choose exactly one shared source: `interceptors=[policy]` or
`policy_file="controls.yaml"`. An agent assignment likewise accepts a nonempty
native tuple or a policy file, never both/neither. All four shared/agent source
combinations are supported. The file is a native ACS manifest, with real Rego
resources where referenced; this package does not implement YAML parsing or
policy evaluation.

File paths accept only `str` or `Path`, reject empty strings and NULs, and are
captured as absolute paths without resolving symlinks or lexical `..`.
Choose them in trusted application configuration, never from request/model
content. With `policy_file=`, invalid backend, runtime, run limits or agent
assignments fail before manifest loading. Missing/incompatible optional installations raise
`ACSUnavailableError`; native manifest `ValueError` and builder `RuntimeError`
propagate instead of falling back to Python controls.

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

## Native ACS semantics

The convenience path constructs the published `AcsInterceptor` and uses the
same CrewAI hook mapper, SDK emitter and bounded worker as native policies.
Manifests are parsed at construction; Rego bundles compile lazily at the first
bound evaluation. A broken later-agent/tool policy may therefore fail after
earlier effects. Keep manifests and referenced resources immutable for the run;
there is no hot reload or transactional rollback.

Shared YAML should bind all eight emitted points, including `agent_shutdown`.
Agent YAML needs the four model/tool points. Missing bindings deny rather than
being skipped; shutdown is audit-only and its actual denial cannot retroactively
block a completed action. Each crew/agent binding has its own SDK session, so a
shared interceptor sees different session IDs for crew-level and agent-level
events. Native SDK summaries retain `name=None` because this
host registers no explicit names; passing `AcsInterceptor(name=...)` does not
change that. Ordered application configuration and actual verdict reasons
provide context, not fabricated source labels.

The pinned Cedar engine requires legacy `snapshot.envelope.agent.id` and fails
closed with the canonical SDK context; no translation shim is supplied.
YAML approval resolver settings are not host configuration: only the explicit
`resolver` argument and SDK approval rules apply. Review the
[ACS limits and trust boundary](coverage.md#acs-policy-source).
Applications may also pass a trusted prebuilt `AcsInterceptor` through
`interceptors=`, including per-agent tuples. The complete ACS examples use
this explicit form. Native construction happens before adapter validation
and uses the application's installed runtime; the convenience loader's path
and version checks do not apply to prebuilt objects. Custom native limits or
cache reuse do not add service-wide admission control.

Backend selection and initialization are the only framework-dependent layer.
Native workflow construction and unsupported feature handling are not hidden
or automatically translated. Framework-neutral policies should not depend on
backend-specific role text or the exact native agent UUID.
