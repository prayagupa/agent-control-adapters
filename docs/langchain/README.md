# LangChain adapter

`agent-hooks-langchain` governs a native `create_agent` graph using one concrete
controls middleware. Namespace: `agent_hooks_langchain`. Pins: LangChain
**1.4.3**, core **1.6.6**, LangGraph **1.2.13**, prebuilt **1.1.0**, optional
ACS **0.4.0a4**.

It depends on [agent-hooks-langgraph](../langgraph/README.md) for shared native
message/tool codecs and graph proof. Dependency direction is LangChain ->
LangGraph; standalone LangGraph does not require top-level LangChain or CrewAI.

**Unpublished alpha.** Real-Qwen tool/copy acceptance is blocked; see
[validation](../validation.md) and [security](../../SECURITY.md).

## Separate installation

Requires Python **3.11-3.13**. From the repository root:

```bash
uv sync --directory python/packages/langchain --locked --group dev --extra acs
uv run --directory python/packages/langchain --no-sync pyright
uv run --directory python/packages/langchain --no-sync \
  pytest -c pyproject.toml --rootdir . -m 'not e2e'
```

Omit the ACS extra for native Agent Hooks only. Keep both sibling local sources:
`../core` and `../langgraph`, including graph tests shared by the conformance
suite. This project owns its environment/lock; do not sync CrewAI/root extras.
See [contributing](../../CONTRIBUTING.md) and [CI](../../.github/workflows/ci.yml)
for exact lint, audit, build, and isolated import checks.

## Application-owned builder

Use `Backend.LANGCHAIN` with the [shared factory](../api.md).
`runtime` is a synchronous `Callable[[LangChainControls], NativeRuntime]`.
Inside it, call `controls.middleware(label)` exactly once, visibly call native
`create_agent(model, tools, middleware=[middleware])`, and return
`NativeRuntime(graph, {label: graph})`: one actual compiled graph actor.

Choose `interceptors` or `policy_file`. Shared controls precede the actual actor's
additional `AgentPolicy` controls. Complete
[Agent Hooks](../../examples/langchain/agent_hooks/) and
[ACS](../../examples/langchain/acs/) folders show models, clients, native setup,
tool bodies, policies, error handling, and cleanup without canned model output.

## Closed profile and native proof

Only the pinned `create_agent` graph with that exact middleware is supported.
Models are explicit `BaseChatModel` objects with sync and async implementations;
local `StructuredTool` objects have both explicit bodies and Pydantic schemas.

Reject arbitrary compiled graphs/bare middleware, custom state/reducers/context,
additional middleware, callbacks/custom trace processors, checkpoint/store/
interrupt/cache/retry/error/timeout modes, remote/delegated tools, structured/
rich output, user `Command`, and extended model responses.

Read-only inspection proves exact native model func/afunc and single wrapper
ownership, tool wrap/awrap ownership, before-agent bindings, default tool-error
handler identity, complete builder/compiled node sets, bound objects, native
routing/destination maps, and pinned mappers/writers/triggers/channels/caches.
The handler must rethrow governance failure instead of returning a recoverable
model-visible error.

With middleware present, native `create_agent` installs an empty `ToolNode` even
for `tools=[]`. Validation follows actual nodes/branches, not tool-list truthiness.
Default trace/stream-transformer identities are checked; custom global trace
policy and streaming execution are unsupported. No private field is patched.
Captured topology/configuration is revalidated before effects; concurrent
application mutation is unsupported. Upgrades require new probes/tests.

## Transform contract

| SDK boundary | Supported executable write-back |
|---|---|
| Startup | Unchanged notification; allow/deny |
| Input | Prompt content; fixed user role |
| Pre-model | Text through native `ModelRequest.override`, including separate system message |
| Post-model | One `AIMessage` inside `ModelResponse`; preserve usage, IDs, calls, finish metadata |
| Pre-tool | Entire strict arguments through `ToolCallRequest.override`; original and replacement round-trip without coercion/dropped fields |
| Post-tool | Actual JSON/text `ToolMessage`; preserve call ID, name and status |
| Output | Final plain assistant content |
| Shutdown | Actual audit-only verdict |

Count, role, system placement, call identity, routing, unknown envelope keys,
rich/null output, and unsupported transforms fail closed. Invalid native values
may fail before a post callback; missing records are not fabricated.
Absent finish metadata uses observed `stop`/`tool_calls` without changing native
metadata. Native before-agent/model and outer input/output coverage are required
before success.

## Lifecycle and limits

Input is exactly `RunRequest({"prompt": text})`, 1-65536 nonblank characters;
no history/config/callback/runtime/identity/session fields. `run` uses native
`invoke` on the caller thread; `arun` uses `ainvoke` on the existing caller loop.
Admission is atomic and single-use across both, including invalid requests.

Shared core owns the private evaluator/worker, session, budgets, and native task
owner; no inline application-loop policies or default-executor bridge.
Limits: one LangChain actor; shared ceilings of 16 actors/nodes, 64 edges,
10 model calls, one tool call per output, 64 pending correlations, 256 non-audit
events, recursion 30, native concurrency 1, and 1-100000 retained records.

Async wall/drain bounds are **120/2 seconds**. Poisoning precedes cancellation
drain. Resistant work remains strongly owned and exposes `cleanup_error`, not
false completion. Sync I/O/tool deadlines and caller client/service/loop cleanup
remain application responsibilities; no sync preemption is promised.

Native governance failures use a fixed safe `AdapterBlocked` from `None`; the
outer boundary retains the private cause. Unrelated native errors propagate,
cleanup does not replace a primary error, and shutdown cannot authorize actions.
Scoped `langsmith.tracing_context(enabled=False)` does not mutate global settings.

## Copy and run real examples

Copy the entire selected folder, including requirements/YAML/Rego. Prepare a
matched **core, graph, and chain** wheelhouse; these alphas are not on PyPI:

```bash
uv venv .venv
uv pip install --python .venv/bin/python --find-links /absolute/wheelhouse -r requirements.txt
.venv/bin/python llm_hooks.py
.venv/bin/python tool_hooks.py --mode allow
.venv/bin/python tool_hooks.py --mode deny
```

Scripts use loopback Ollama, real `qwen2.5:latest`, temperature 0, 128/256 tokens,
60-second transports, zero retries, no proxy/remote fallback, and closed
application-owned clients. Tool choice uses the actual frozen application tool
name with `parallel_tool_calls=False`, then `none` after a real `ToolMessage`.
An absent call fails the unchanged exact-one-effect assertion.
Expected denial prints `status: blocked` and zero effects; other failures do not
become success-shaped output.

Diagnostics default to metadata/audit/drop/effect counts and header timing, not
prompts, arguments/results, provider bodies, or output text.
Only explicit `--synthetic-diagnostics local-toy-v1` permits bounded toy
request/post-policy summaries at the fixed credential-free loopback endpoint:
4096 bytes before request parsing and 65536 emitted bytes per process.
The helper never reads response bodies or serializes full native messages.
SDK/client buffers and normal exception formatting are outside that guarantee.

Historical tool-allow HTTP 200 responses contained no calls and zero effects;
they predate the diagnostic privacy fix and are **not passing model gates**.
The 60-second transport bound is not a completion promise. Required real
wheel/copy tests need an approved exclusive slot, `LOCAL_MODEL_REQUIRED=1`,
`ADAPTER_GRAPH_MODEL_SLOT=approved`, `ADAPTER_WHEELHOUSE=/absolute/wheelhouse`,
and owning `tests/test_e2e.py -m e2e`. Unrun/skipped cases are missing evidence.
See [validation](../validation.md) for model provenance and outstanding gates.
