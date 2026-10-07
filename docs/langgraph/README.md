# LangGraph adapter

`agent-hooks-langgraph` adds Agent Hooks or ACS controls to an application-built
native graph. Namespace: `agent_hooks_langgraph`. Pins: LangGraph **1.2.13**,
prebuilt **1.1.0**, langchain-core **1.6.6**, optional ACS **0.4.0a4**.
It does **not** require top-level LangChain.

**Unpublished alpha.** Current model/tool acceptance remains incomplete.
Read [validation](../validation.md) and [security](../../SECURITY.md).

## Install and check separately

Requires Python **3.11-3.13**. From the repository root:

```bash
uv sync --directory python/packages/langgraph --locked --group dev --extra acs
uv run --directory python/packages/langgraph --no-sync pyright
uv run --directory python/packages/langgraph --no-sync \
  pytest -c pyproject.toml --rootdir . -m 'not e2e'
```

Omit the ACS extra for native Agent Hooks only. This project owns its environment
and lock, references sibling `../core`, and is excluded from root resolution.
Do not select core's CrewAI extra. [Contributing](../../CONTRIBUTING.md) and
[CI](../../.github/workflows/ci.yml) cover lint, audit, builds, and isolated imports.

## Application-owned graph

Pass `Backend.LANGGRAPH` to the [shared factory](../api.md). `runtime` is a
**synchronous builder** accepting `LangGraphControls` and returning `NativeRuntime`.
The application visibly creates `StateGraph(MessagesState)`, its edges, and compile.

- `controls.model_node(label, model, tools=())` creates a guarded native Runnable
  for an explicit `BaseChatModel`.
- `controls.tool_node(label, tools)` creates a guarded Runnable for local
  `StructuredTool` objects. Each tool has a sync function, explicit coroutine,
  and Pydantic argument schema.
- `NativeRuntime.agents` maps every executable guarded node exactly once to its
  actual object. Labels are trusted application configuration.
- Choose `interceptors` or `policy_file`. Shared policies precede that bound
  node's additional `AgentPolicy` controls. Tool nodes have their own identities.
- Guards are inactive during construction; even a caught attempt to execute one
  rejects the builder before policy-file I/O.

## Closed native profile

Only ordinary `MessagesState` with exact native `add_messages` and controls-owned
guarded nodes is supported. Routing uses closed edges or exact native
`tools_condition` with an explicit map to a registered tool node and `END`.

The model's tool set must match its actual destination node. A bounded ledger
captures model origin, immutable call, and expected destination before routing.
The tool node proves identity and unchanged call before charging the shared
budget under the source actor; its own policies still govern execution.
Unknown origin, wrong node, duplicate calls, and cross-actor replay abort.

Reject foreign callables, custom schemas/reducers/routing, `Send`, user `Command`,
subgraphs, delegation, injected state/store/runtime tools, remote tools, rich/
structured output, retries/cache/error handlers, timeout/defer/destination options,
persistence/resume/checkpoints/store/interrupts, and external tracing callbacks.
Trusted application transports and tools remain outside a hard sandbox.

## Transform contract

| SDK boundary | Supported executable replacement |
|---|---|
| Startup | Allow/deny only |
| Input | User content, not role |
| Pre-model | Text; preserve count, roles, IDs, calls, and system placement |
| Post-model | One concrete text `AIMessage`; preserve calls, usage, IDs, and finish metadata |
| Pre-tool | Entire strict JSON argument object; validate original and replacement without coercion/dropped fields |
| Post-tool | Actual JSON/text result rendered into native `ToolMessage` |
| Output | Final assistant content with no pending tool calls |
| Shutdown | Actual audit-only verdict; cannot authorize later effects |

Model output must be text or an actual registered tool call. Null/rich/invalid
native values may fail before post-model emission; no missing record is invented.
Absent native finish metadata is projected as observed `stop`/`tool_calls`
without changing or fabricating the native response metadata.

## Execution and limits

Accept exactly `RunRequest({"prompt": text})`, with 1-65536 nonblank characters;
no history, config, callbacks, identity, or session keys. `run` invokes the graph
on the calling thread; `arun` awaits it on the caller's existing loop.
Admission is atomic and single-use across both modes, including invalid/denied runs.

| Resource | Bound |
|---|---|
| Actors/nodes / edges | 16 / 64 |
| Model effects / tool calls per model output | 10 / 1 |
| Pending route correlations | 64 |
| Non-audit events / recursion / native concurrency | 256 / 30 / 1 |
| Audit retention | `max_records` 1-100000, with exact evictions |
| Async native wall / cancellation drain | 120 / 2 seconds |

The private policy loop, evaluator, audit, and budgets are core-owned; policies
do not run inline on the application loop or in a default-executor bridge.
Cancellation/deadline poisons before drain. Resistant work remains strongly owned
and exposes `cleanup_error`; a timeout does not prove it stopped. Sync transport/
tool timeouts are application-owned; sync preemption is not promised.
Caller clients, loops, and services remain caller-owned.

Native governance errors use a fixed safe message without an SDK-visible cause;
the outer application error retains the private cause. Cleanup failure does not
replace a primary error. Only actual records are exposed. Scoped
`langsmith.tracing_context(enabled=False)` does not alter global settings.

## Pinned inspection

Read-only checks inventory native `StateNodeSpec`, `PregelNode`, builder branches,
compiled channels, writers, bound objects, triggers, defaults, and derived caches.
Exact executable sets/identities are captured and rechecked before effects.
No private native field is patched; concurrent application mutation is unsupported.
Upgrades require new native probes, not relaxed version gates.

Tool binding accepts exact `RunnableBinding` and langchain-core's pinned
`_ChatModelBinding`, not arbitrary subclasses, factories, custom types, changed
kwargs, or non-tool options. Narrow upstream typing gaps are locally annotated;
owned APIs/tests/examples retain strict Pyright without global relaxation.

## Complete examples

Copy [Agent Hooks](../../examples/langgraph/agent_hooks/) or
[ACS](../../examples/langgraph/acs/) in full, including requirements and policy
assets. Install matched core/graph wheels from a prepared wheelhouse:

```bash
uv venv .venv
uv pip install --python .venv/bin/python --find-links /absolute/wheelhouse -r requirements.txt
.venv/bin/python llm_hooks.py
.venv/bin/python tool_hooks.py --mode allow
.venv/bin/python tool_hooks.py --mode deny
```

Scripts use real `qwen2.5:latest` at loopback Ollama, temperature 0, bounded tokens,
60-second transports, zero retries, no proxy/remote fallback, and closed
application-owned clients. Deny mode exits successfully only for the expected
actual pre-tool denial with zero effects and `status: blocked`, not fake output.
The tool example requests real calls and a real final answer; current acceptance
must not be inferred from a successful HTTP response.

Diagnostics default to metadata, audit/drop/effect counts and request timing,
not prompt, arguments, results, or provider bodies. Explicit
`--synthetic-diagnostics local-toy-v1` permits bounded toy request/post-policy
summaries only at the fixed credential-free loopback endpoint: requests are capped
at 4096 bytes before parsing and diagnostic emission at 65536 bytes per process.
The helper never captures response bodies/full native messages. SDK/client buffers
and normal exception formatting are outside that guarantee.

Real copied-folder/wheel gates require an approved exclusive slot,
`LOCAL_MODEL_REQUIRED=1`, `ADAPTER_GRAPH_MODEL_SLOT=approved`, and
`ADAPTER_WHEELHOUSE=/absolute/wheelhouse`, then owning `tests/test_e2e.py -m e2e`.
Unrun/skipped gates are missing evidence. No multi-agent model demo is shipped
without its own real gate; model provenance and failed/missing acceptance remain
in [validation](../validation.md).
