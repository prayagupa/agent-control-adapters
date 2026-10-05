# CrewAI adapter architecture

## Why a separate adapter?

Applications opt into a small package depending on both CrewAI and agent-hooks.
Neither upstream framework imports the adapter. The initial release pins
`crewai==1.15.23` and `agent-hooks-sdk==0.1.0a5`: these are compatibility boundaries,
not a claim that every future patch is supported.

The shared `agent-hooks-adapters` package owns the small application-facing
request/result contract, logical policy assignments and explicit backend
selection. The CrewAI package owns translation and execution scope. The SDK owns verdict
composition, context identity, approval validation, and interception records.
Applications own policies, actual tools, model clients, and external side effects.

The initial profile also rejects automatic summarization and structured-output
repair. These can issue model calls outside the framework's public hooks. The
last model iteration is reserved and fails closed before an unhooked forced-final
request. This is intentionally narrower than wrapping or patching every model
client; the application receives a clear block rather than an ungoverned answer.

```mermaid
flowchart TB
    App["Trusted application configuration"] --> API["with_agent_hooks + RunRequest"]
    API --> Bind["Resolve application labels to actual agent objects"]
    Bind --> Run["Adapter.run -> RunResult"]
    Run --> Adapter["Existing CrewAIAdapter.kickoff"]
    Adapter --> Scope["Execution-scoped public hooks"]
    Scope --> Crew["CrewAI sequential runner"]
    Crew --> Hook["Model, tool and execution boundaries"]
    Hook --> Validate["Validate and copy untrusted payload"]
    Validate --> Worker["Bounded worker with private event loop"]
    Worker --> Emitter["SDK emitter for selected policy binding"]
    Emitter --> Policy["Shared policies then agent policies"]
    Emitter --> Approval["Optional approval resolver"]
    Emitter --> Records["Bounded audit records"]
    Emitter --> Verdict{"Allow, transform or deny?"}
    Verdict -->|allow| Execute["Continue guarded action"]
    Verdict -->|validated transform| Execute
    Verdict -->|deny or error| Abort["Sticky run denial"]
```

## Public API and ownership

`with_agent_hooks` wraps an existing native runtime; `create_adapter` is a
compatible alias. Neither call registers global hooks or changes the workflow.
The portable `Adapter` protocol exposes `run(RunRequest) -> RunResult`, bounded
SDK audit snapshots and explicit cleanup failure. `RunRequest` captures strict,
bounded JSON and returns detached copies; `RunResult` contains text and actual
records, never a success-shaped denial. The core imports no CrewAI types except
at its lazy composition-root dispatch, when the user explicitly selects that
installed backend with the public `Backend` enum. `Backend.CREWAI` is the only
implemented member. Raw strings and invalid selectors fail before backend loading,
instead of falling back.

Portable `AgentPolicy` names an application-owned label. The trusted `agents`
configuration resolves that label to one actual crew-owned Agent. Missing,
foreign, duplicate or ambiguous bindings are rejected; policies never select an
agent from request content or role text. Labels do not overwrite native SDK
agent UUIDs or other host metadata. The compiled controls are frozen into the
existing native object bindings. This keeps policy assignments reusable without
inventing a universal workflow or changing current event coverage.

`CrewAIAdapter` is single-use and binds one actual `Crew`. Its `run()` delegates
to the existing `kickoff()` method, so both entry points share admission, atomic
single-use state, scoped hooks, denial and cleanup. The `kickoff()` method
installs scoped hooks only for the run and always cleans up its worker. It does
not register process-wide callbacks. `AgentPolicy` binds interceptor instances to
an actual host-owned agent identity. Agents with the same role text do not share
policies unless the application deliberately gives them the same instance.
Native callers retain their original API and native result. Shared blocked and
cleanup error bases let portable callers catch enforcement failures without
depending on CrewAI, while the native abort type, causes and notes are preserved.

Crew execution-boundary events use shared policies. Model/tool events use shared
policies plus the matching agent's policies. Composition is sequential/run-all:
an approved escalation must not bypass a later mandatory policy.

Each binding has its own SDK session and sequence. A single worker owns mutable
SDK state. A bounded group of hook callers may wait for emission serialization
because CrewAI can execute multiple native tool calls in parallel even in a
sequential crew. Tool correlation includes host thread, agent and tool identity;
missing post events are not invented.

## Denial is not a model response

```mermaid
sequenceDiagram
    participant App as Application
    participant Crew as CrewAI runner
    participant Adapter as Scoped adapter
    participant SDK as SDK emitter
    participant Policy as Interceptors
    App->>Adapter: kickoff
    Adapter->>Crew: kickoff within scoped hooks
    Crew->>Adapter: before guarded operation
    Adapter->>SDK: copied context with trusted identity
    SDK->>Policy: intercept
    Policy-->>SDK: deny
    SDK-->>Adapter: InterceptionBlocked
    Adapter->>Adapter: remember run denial
    Adapter-->>Crew: HookAborted
    Note over Crew,Adapter: CrewAI may catch a pre-tool abort and skip that tool
    Crew->>Adapter: next boundary or final return
    Adapter-->>App: AdapterBlocked, never successful CrewOutput
```

Arbitrary callback exceptions are unsafe to rely on because CrewAI's hook
dispatcher catches them. The adapter converts mapping, engine and policy failure
into the framework's supported abort path. Public errors have a safe constant
message; audit records retain policy reasons. A denied post-model/tool value is
not replaced with a success-looking string.

Shutdown after denial is audit-only and must not replace the original exception.
If a failed worker cannot emit shutdown, a host diagnostic reports missing SDK
shutdown evidence rather than synthesizing a successful SDK record.

## Timeouts and cleanup

One worker owns a persistent event loop, one active emission, and bounded
admission. SDK timeouts govern cooperative async callbacks. The caller's
`emit_timeout` also bounds the wait for synchronous policies. Timeout poisons the
adapter: no additional work is admitted and late transforms cannot be applied.

Python cannot kill a running synchronous callback safely. Its slot remains
occupied until it exits. Cleanup joins for a finite interval and reports failure
if a noncooperative callback remains alive. A daemon thread in that case is a
**reported residual resource**, not a claimed successful cancellation. Use process
isolation for untrusted or potentially nonterminating policy code.

The adapter does not own the model service or cancel Ollama's server-side
generation. Model clients, HTTP timeouts, tool deadlines and external side-effect
idempotency remain application responsibilities. No automatic replay is performed
by the adapter after a denial.

## Trust boundaries and threats

| Risk | Control | Remaining limitation |
|---|---|---|
| Prompt claims another agent's identity | Resolve policies from host-bound agent objects | Trusted application code can still alter its own objects |
| Payload overwrites session/policy metadata | Build metadata separately; copy and validate content | SDK/custom Python policies are trusted code, not sandboxed |
| Malformed transform or engine exception | Convert to abort; retain sticky deny | No rollback of an action already completed before its post hook |
| Concurrent policy/audit state races | Worker-owned state, bounded serialized emissions | Shared interceptor objects across adapters require application synchronization |
| Approval bypasses other policies | Run-all composition; SDK identity echo validation | Resolver must authenticate a real approver in production |
| Hung callback or approval | Finite deadlines; bounded residual work and explicit cleanup error | Noncooperative Python needs process isolation |
| Secret disclosure in diagnostics | No payload logging by adapter; safe public denial text | Policy reason strings and framework telemetry must also be designed safely |
| Later hooks undo enforcement | Narrow profile and explicit coverage documentation | Trusted hooks/plugins can bypass guarantees; not an authoritative kernel |

## Alternatives and future packages

Built-in integration would couple CrewAI to the SDK and was not the requested
deployment model. Global registration makes scope and cleanup harder. Monkey
patching could reach missing seams but would depend on private implementation.
The chosen public-hook wrapper keeps the dependency direction explicit and
rejects unsupported paths.

Future adapters belong in independently versioned `packages/langgraph`,
`packages/google-adk`, or `packages/openai-agents`. The approved portability
requirement introduces only shared data/protocol contracts and explicit
dispatch now; there is no generic runner, plugin system or workflow translator.
An OpenAI adapter is not implemented or advertised as interchangeable yet.
Each future backend must demonstrate the same request/result, blocked-error,
policy-ordering and identity-ownership contracts through its actual runner.
Differences in coverage, structured output and concurrency must fail explicitly,
not be hidden behind the common method name.

## Rollout

This workspace does not publish a release. Evaluate using harmless tools and
review audit records before real effects. Widen version pins only after the real
runner tests pass for each candidate version. Removing the wrapper/dependency is
the rollback, and also removes its enforcement. Partner sign-off is required
before a future upstream-facing merge; this task does not contact partners.

Mermaid source is included for GitHub rendering. Local rendered-image validation
is not claimed when a Mermaid renderer is unavailable.
