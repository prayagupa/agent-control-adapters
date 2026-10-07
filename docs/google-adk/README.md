# Google ADK adapter

`agent-hooks-google-adk` governs the installed **Google ADK 2.11.0** runner with
Agent Hooks or optional ACS/Rego. Import namespace: `agent_hooks_google_adk`.
It preserves native ADK execution and uses the published SDK/policy runtime.

**Unpublished alpha.** Final real-model and independent review acceptance are
incomplete; see [validation](../validation.md) and [security](../../SECURITY.md).

## Install separately

Requires Python **3.11-3.13**. From the repository root:

```bash
uv sync --directory python/packages/google-adk --locked --group dev --extra acs
uv run --directory python/packages/google-adk --no-sync pyright
uv run --directory python/packages/google-adk --no-sync \
  pytest -c pyproject.toml --rootdir=. -m 'not e2e'
```

Omit `--extra acs` for native Agent Hooks only. This package owns its lock and
environment, uses sibling `../core`, and pins ACS **0.4.0a4** when selected.
Examples use LiteLLM **1.104.0**. ADK's OpenTelemetry/websockets requirements
conflict with CrewAI: never install this profile in the root CrewAI environment.
See [contributing](../../CONTRIBUTING.md) and [CI](../../.github/workflows/ci.yml)
for lint, audit, build, and noneditable import checks.

## Execution and ownership

Use `with_agent_controls(backend=Backend.GOOGLE_ADK, runtime=runner, ...)` from
the [shared API](../api.md). Choose `interceptors` or `policy_file`, not both.
Shared controls precede per-actor `AgentPolicy` controls bound to actual objects.

- Supply an exact `google.adk.Runner` and exact `LlmAgent` with explicit
  `mode="chat"`, or an exact `SequentialAgent` tree of supported LlmAgents.
  Actor names must be unique and trusted; models must be explicit `BaseLlm`
  instances, not deferred string selections.
- Call `await adapter.arun(RunRequest({"prompt": text}))` on your application
  loop. Input contains exactly one nonempty prompt, at most 65,536 characters;
  no caller history, role, metadata, identity, callbacks, config, or session IDs.
- Every admitted attempt consumes the adapter, including invalid input, denial,
  timeout, and cancellation. `run()` raises before admission and does not create
  an executor or nested loop; correcting the call to `arun` remains possible.
- The adapter owns its private App/Runner/plugin, generated user/session,
  in-memory session, stream, task, and policy lane. It closes/deletes only those.
  Caller runners, models, clients, services, and loops remain caller-owned.

Tools must be async functions or exact `FunctionTool` wrappers with explicit,
keyword-callable scalar `str`/`int`/`float`/`bool` parameters and matching defaults.
No context injection, variadic parameters, or reserved governance metadata.
Returns are bounded strict JSON; ADK normally wraps scalars as `{"result": value}`.

## Supported boundaries

| SDK point | Native seam | Transform authority |
|---|---|---|
| `agent_startup` | First lifecycle callback | Allow/deny only |
| `input` | User-message callback | Content, not role |
| `pre_model_call` | Before-model callback | Text; preserve count, roles, non-text history, call IDs and routing |
| `post_model_call` | Replacement native `LlmResponse` | Plain content; preserve calls, usage, version and finish metadata |
| `pre_tool_call` | Same executable argument dictionary | Strict original schema before native sanitation and after transform |
| `post_tool_call` | Replacement result consumed by ADK | Bounded strict JSON; no rollback |
| `output` | Native after-run callback | Final plain content |
| `agent_shutdown` | After-run or owned failure cleanup | Audit-only |

Model output must be nonempty plain text or tool-call-only with an explicit call
ID. Null, rich/thought/multimodal, mixed text/tool, incomplete, and uncorrelated
outputs fail closed. Source rejection before SDK emission is not a fabricated
policy record.

Denial aborts the whole run, never returning a recoverable error to the model.
Native callback logs receive a fixed safe message; the application's outer
`AdapterBlocked` retains the private cause. Unrelated native/transport errors
propagate. `records`/`records_dropped` are actual SDK records/evictions, not
invented post-tool or shutdown events.

A poisoned lane may refuse shutdown audit. That absence is logged as
`adapter_shutdown_audit_unavailable` and attached privately to the primary
exception. `cleanup_error` separately reports incomplete native/policy cleanup.
Abort signaling precedes cancellation drain; late cancellation-resistant
callbacks propagate cancellation rather than entering ADK's node-error wait.

## Pinned profile and limits

ADK normally changes `mode=None` to `chat`; this adapter instead requires explicit
configuration. ADK's root shallow clone is accepted only at the native callback
with identical model/tools, unchanged other settings, and owned invocation/
session/plugin membership. Foreign/additional clones are rejected. Field and
callback inventories are checked before policy I/O; arbitrary later ADK versions
are not certified.

Reject custom runtime/tool subclasses, delegation subtrees, parallel/loop/custom
workflow graphs, preinstalled plugins/callbacks, toolsets, remote/long-running/
deferred/confirmation or sync tools, state/input/output schemas, structured output,
planners/code execution, retry, durable services, compaction/cache/resume,
streaming/live/rich content, and uncovered transport profiles.

| Resource | Bound |
|---|---|
| Actors / nodes / edges | 16 / 16 / 64; actual tree at most 15 edges |
| Model effects per run | 10; block before the 11th |
| Tool calls per result / concurrency | 1 / 1 |
| Pending correlations | 64, atomically checked/updated |
| Stream events / non-audit SDK emissions | 256 each |
| Shared recursion profile | 30; arbitrary ADK graphs remain rejected |
| Tools per actor / parameters per tool | 64 / 64 |
| Native wall / cancellation drain | 120 / 2 seconds |
| Record retention | Configurable 1-100000, with exact drops |
| JSON | 1 MiB, 10000 nodes, depth 32, finite interoperable numbers |

Wall, policy, and transport timeouts are distinct. Cancellation-resistant work
remains strongly owned and bounded; incomplete cleanup is not described as stopped.

## Complete examples and model evidence

Choose [Agent Hooks](../../examples/google-adk/agent_hooks/) or
[ACS](../../examples/google-adk/acs/). Each contains `llm_hooks.py`, `tool_hooks.py`,
requirements, visible native setup, and caller cleanup; ACS also has both YAML
manifests and `policy/`. Copy the whole selected folder and install matched
core/ADK source packages or wheels in a dedicated environment, not from an
unpublished PyPI release.

Scripts use operator-managed loopback Ollama, `qwen2.5:latest`, temperature 0,
128 output tokens, zero retries, and a caller-owned OpenAI client through
LiteLLM. Effective connect/read/write/pool bounds are **60 seconds** despite the
HTTPX client's configured 3-second connect bound; scalar request timeouts override it.
Scripts do not start/stop/download models. No multi-agent model example is claimed.

Earlier copied-folder passes preceded the layout change. The later 45-second
refresh had four read timeouts; changing numeric bounds to 60 seconds does **not**
prove current success. All four final-layout model gates need a new explicit
exclusive-slot grant.

After that grant only:

```bash
LOCAL_MODEL_REQUIRED=1 ADK_MODEL_SLOT=owned timeout --signal=KILL 480 \
  uv run --directory python/packages/google-adk --no-sync \
  pytest -c pyproject.toml --rootdir=. -m e2e tests/test_e2e.py
```

The suite copies folders and runs real scripts under bounded subprocesses.
Release the owned slot after cleanup; a CLI timeout does not prove daemon-side
generation stopped. Never stop the operator's service. Model digest/version and
remaining acceptance limits are recorded in [validation](../validation.md).
