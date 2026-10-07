# Shared control API

A small, typed execution API for native Agent Hooks and YAML-based ACS controls.
It does not define agents,
tasks, tools, models, or workflows. Applications keep that setup in their native
framework and pass the resulting runtime into `with_agent_controls`.

Install from source using the selected [adapter guide](../README.md#get-started).
The root workspace contains core and CrewAI only. Google ADK, LangChain,
LangGraph, OpenAI Agents SDK and Microsoft Agent Framework have independently
locked environments; do not combine them into the CrewAI workspace.
The `agent-hooks-adapters` distribution declares `[crewai]` and `[acs]` extras;
`[crewai,acs]` selects both the backend and the optional, pinned
`agent-control-spec==0.4.0a4` runtime. Adapter distributions are not yet published
to PyPI.
Import `Backend` from `agent_hooks_adapters` and select an implemented member:
`Backend.CREWAI`, `Backend.GOOGLE_ADK`, `Backend.LANGCHAIN`,
`Backend.LANGGRAPH`, `Backend.OPENAI_AGENTS` or `Backend.MAF`.
The factory imports only the selected adapter.
It requires an enum member and raises
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
Backend.GOOGLE_ADK  # value: "google-adk"; async-native
Backend.LANGCHAIN  # value: "langchain"; native sync and async
Backend.LANGGRAPH  # value: "langgraph"; native sync and async
Backend.OPENAI_AGENTS  # value: "openai-agents"; async-native
Backend.MAF     # value: "maf"; ACS-only, async-native
AgentPolicy(
    agent_id: str,
    interceptors: tuple[Interceptor, ...] = (),
    policy_file: str | Path | None = None,
)
RunRequest(inputs: dict[str, object] | None = None)
agent_control.run(request: RunRequest | None = None) -> RunResult
await async_agent_control.arun(request: RunRequest | None = None) -> RunResult
RunResult(output: str, records: tuple[InterceptionRecord, ...], records_dropped: int)
```

`AsyncAdapter` extends `Adapter` with
`arun(request: RunRequest | None = None) -> RunResult`. This is a capability
contract for async-native backends, not an executor bridge for CrewAI.
The five new backend selectors return this capability; CrewAI's synchronous
`run` remains unchanged. Google ADK, OpenAI Agents SDK and MAF reject synchronous
`run` before admission, so the same adapter can then be used correctly with
`arun` on the application's existing event loop. LangChain and LangGraph
execute real native `invoke` / `ainvoke`; they do not run synchronous work in
an executor to simulate async support.

All five new adapters accept exactly `RunRequest({"prompt": nonempty_text})`,
with 1-65,536 characters, at least one non-whitespace character, and no
additional runtime, history, identity, or options keys. An admitted failure
consumes the single-use adapter. CrewAI retains its native task-input mapping.

`NativeRuntime(runtime, agents)` is a frozen builder result for adapters whose
native workflow is constructed by the application. It copies a read-only,
bounded mapping of trusted labels to distinct native objects; it does not copy,
own, or close those objects. Backend validation must still prove membership and
complete enforcement coverage before policy loading. It does not itself
register another backend.

LangChain and LangGraph use application-owned builders supplied by their
packages. Their `NativeRuntime` contains the actual compiled workflow and
trusted actor mapping. LangGraph binds every guarded executable node to its
own actual object; model-origin tool correlation does not turn a tool-node
label into authorization. A separately supplied `agents` mapping must agree
exactly with the builder's mapping. See the
[LangChain guide](langchain/README.md) and [LangGraph guide](langgraph/README.md)
for complete construction and supported topology.

This is an interface description, not a model-output demonstration. See the
[complete real-model examples](crewai/examples.md).
`with_agent_hooks` and `create_adapter` remain exact aliases with the same
signature and behavior.
Wrapping does not mutate the native runtime or register process-global hooks;
call the returned adapter's supported `run` or `arun` method to enter its
governed scope. Caller-owned models, clients, and event loops remain caller-owned.

`agents` is trusted configuration mapping stable application labels such as
`researcher` to actual native agent objects. Policies refer to those labels;
backends validate ownership and bind the controls to the objects, never to role
text or request/model-provided IDs. Labels are 1-128 ASCII letters, digits,
underscores, dots, or dashes. Unknown labels, foreign agents and duplicate
bindings are rejected.

Choose exactly one shared source: `interceptors=[policy]` or
`policy_file="controls.yaml"`. An agent assignment likewise accepts a nonempty
native tuple or a policy file, never both/neither. All four shared/agent source
combinations are supported by CrewAI. MAF accepts ACS controls only, either
policy files or trusted prebuilt native `AcsInterceptor` instances; it rejects
arbitrary Python interceptors. It reuses the framework's native Agent Hooks
middleware rather than adding another native-hooks adapter.
Google ADK, LangChain, LangGraph and OpenAI Agents SDK support both policy
sources through the same backend integration. Switching the shared policy
source does not rebuild or translate the native workflow.
The file is a native ACS manifest, with real Rego
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
selected backend's native event mapper, SDK emitter and bounded worker.
For CrewAI this is the same mapper as Python policies; MAF delegates the six
in-run points to its native external-emitter middleware and adds startup/shutdown.
Manifests are parsed at construction; Rego bundles compile lazily at the first
bound evaluation. A broken later-agent/tool policy may therefore fail after
earlier effects. Keep manifests and referenced resources immutable for the run;
there is no hot reload or transactional rollback.

Shared YAML should bind all eight emitted points, including `agent_shutdown`.
CrewAI agent YAML needs the four model/tool points. MAF binds all eight points
to its single governed agent. Missing bindings deny rather than
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
