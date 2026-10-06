# Coverage and compatibility

Initial dependency contract: Python 3.11-3.13, CrewAI **1.15.23**, agent-hooks-sdk
**0.1.0a5**. Runtime execution evidence records the actual interpreter separately;
declaring a Python range does not mean every interpreter/platform was tested.
The SDK includes a compiled native core; moving it outside CrewAI does **not**
remove that dependency for applications installing this adapter.

## Supported profile

Call `with_agent_controls(...).run(...)` on a synchronous, sequential crew.
`with_agent_hooks` and `create_adapter` are exact compatibility aliases.
The legacy `CrewAIAdapter.kickoff()` entry point remains supported. Policies and
approval resolvers are trusted application code. Shared interceptors apply to
crew boundaries; shared plus agent-bound interceptors apply to model/tool
boundaries. Native parallel tool calls are serialized only at policy evaluation,
not at the tool's actual side effects.

| Boundary | Allow / deny | Transformation |
|---|---|---|
| Crew startup | Yes | Registration-changing transforms rejected |
| Crew input | Yes | Valid input mapping |
| Pre-model call | Yes | Valid message list, applied in place |
| Post-model call | Textual response path | Valid textual response envelope |
| Pre-tool call | Yes, whole run remains denied after a block | Valid argument object, applied in place |
| Post-tool call | Yes, no successful output after denial | Textual result |
| Crew output | Yes | Text output preserving the CrewOutput shape |
| Shutdown | Audit-only; cannot undo effects | No action-changing guarantee |

The adapter deliberately does **not** register `PRE_STEP` or `POST_STEP`.

## ACS policy source

Install the optional `[crewai,acs]` extra (or development `acs` group) and use
`policy_file` instead of `interceptors`. The pinned runtime is
`agent-control-spec==0.4.0a4`, using SDK 0.1.0a5. It accepts the native
0.4.0-alpha.1 and 0.5.0-alpha.1 manifest contracts. Tested controls use real
Rego bundles and canonical top-level SDK `agent`, `session`, `target` and
`tool_call` fields; the host does not translate metadata or implement a DSL.

Native and file-backed per-agent policies can be mixed. Shared YAML should bind
`agent_startup`, `input`, `pre_model_call`, `post_model_call`, `pre_tool_call`,
`post_tool_call`, `output` and `agent_shutdown`. Agent files need only the four
model/tool points. Missing bindings return native denial, not an implicit permit.
Shutdown denial is retained in actual records but cannot retroactively block an
otherwise allowed result.

| ACS behavior | Adapter outcome / limitation |
|---|---|
| Missing/invalid manifest | Native `ValueError` before run; builder `RuntimeError` stays visible |
| Missing/broken Rego bundle | Construction can succeed; denial at its first bound evaluation |
| Shared bundle first bound at startup | Tested denial before any model/tool call |
| Healthy startup, broken tool bundle | Tested denial after one model request, before the tool body |
| Broken later-agent bundle | Tested denial after an earlier agent's model request; no rollback |
| Transforms and approvals | Existing SDK validation/write-back/resolver; shared mandatory deny wins |
| YAML `approval` configuration | Not wired as host authority; provide the explicit `resolver` |
| Cedar | Released engine expects `snapshot.envelope.agent.id`; canonical contexts fail closed |
| Full snapshot budget | Default 1,048,576 serialized bytes, distinct from request-content bounds |
| Runtime deadlines | Rego default 5,000 ms; ambient `ACS_OPA_TIMEOUT_MS`/`ACS_REGO_V0` affect behavior |
| Native work after timeout | Not forcibly killed; two pools allow 32 abandoned workers each plus concurrency |
| Native fatal errors | Allocation/parser failures may abort the process, not produce an audit denial |
| Network and credentials | Trusted inheritance/annotators can access network and named host environment secrets |
| Audit names | Actual SDK summaries keep `name=None`; no fabricated provider labels |

Boundary probes allowed an exactly 1,048,576-byte native snapshot and denied a
1,048,580-byte one with `runtime_error:resource_limit_exceeded`. A valid request
at the host's 1-MiB input-text budget was blocked before any model call because
its complete context contains copies of the input; host refusal did not invent
an input SDK record. Tests also bracket the actual native threshold.

Keep manifests and referenced policy directories immutable throughout a run.
No hot reload, artifact snapshot or eager Rego readiness guarantee is supplied.
Use the native `ActivatedPolicy` API separately for deployment preflight, with
its deadline caveats, or a trusted prebuilt `AcsInterceptor` through
`interceptors=` for custom limits/cache reuse.

ACS 0.4.0a4 provides a Linux x86_64 wheel requiring glibc >=2.34; other platforms
need a source build/toolchain. See [contributing](../CONTRIBUTING.md) and
[security](../SECURITY.md#acs-configuration-is-trusted-code). ACS is optional:
the native-only source path imports no ACS module.

## Not covered or deliberately rejected

- Direct calls to `crew.kickoff()`, LLMs or tools outside the wrapper are not
  governed by this adapter.
- Async/hierarchical crews, asynchronous tasks, delegation, streaming,
  checkpoint resume, planning, memory/knowledge pipelines, task guardrails,
  human-input tasks and effectful lifecycle callbacks are outside this alpha
  profile and rejected at admission where configured.
- Task output files are rejected because CrewAI writes them before final output
  governance. Tool-argument transformation requires tool caching to be disabled,
  so a cached result cannot be reused under stale pre-transform arguments.
- Set each agent's `respect_context_window=False`. Automatic context summarization,
  configured agent/crew `function_calling_llm`, and task `output_json`,
  `output_pydantic` or `response_model` are rejected because their model
  conversion/repair paths can bypass public hooks. Agent and crew skills are
  also rejected before loading. Existing executors are checked too: clearing an
  agent's settings does not necessarily clear settings retained by its executor.
- The adapter reserves the last model iteration: it blocks at
  `iterations >= max_iter - 1`, before CrewAI can issue an unhooked forced-final
  request. `max_iter=1` allows no model request; choose at least 2. Exhaustion is
  an explicit block, not a promise that the model will produce a final answer.
- With the legacy `CrewAgentExecutor`, changing initially empty native tool
  arguments is rejected: that released executor otherwise executes a different
  argument dictionary. Nonempty argument transformations remain supported.
- CrewAI can bypass its textual post-model hook for structured native tool-call
  responses. **Do not use a post-model policy as your sole tool authorization
  control.** Govern tool execution at `pre_tool_call`.
- Arbitrary trusted Python hooks, tools and plugins can perform side effects or
  call providers directly. This package is not an in-process sandbox.
- A post hook acts after the model/tool has run. It cannot undo the operation or
  promise that CrewAI internal logs, telemetry or caches never saw the original.
  Structured-output transformations that could leave stale alternate output
  views are rejected.
- No durable external audit sink, distributed quota store, approval service,
  retry/idempotency layer or remote approval authentication is provided.

## Conformance claims

CrewAI's native hook scenarios have [complete SDK-backed replacements](crewai/examples.md):
model-response sanitization, destructive-tool denial, pre-tool approval and
agent-specific controls. Native hooks can swallow ordinary callback errors or
let the agent continue after a tool denial; this adapter fails closed and makes
denial sticky for the whole crew run. This is intentional, not drop-in equality
with every native hook behavior.

The SDK's reference-host conformance vectors, when run, establish **SDK reference
behavior**, not this adapter's completeness. The adapter's tests separately
exercise the real CrewAI runner, real SDK emitter, denial/transform paths,
per-agent isolation and cleanup. Model boundaries may be stubbed in deterministic
tests; the example e2e tier uses actual Ollama responses.

This alpha does **not claim full agent-hooks host conformance**. In particular,
it fixes enforce/run-all composition, has no evaluate-only configuration or
arbitrary identity provider, is single-run rather than multi-turn, does not
support streaming, and turns a caught tool denial into whole-run denial. These
are declared profile differences, not reasons to rewrite vector assertions.

The initial local reference run passed all **47** packaged SDK vectors without
changing expectations. Actual example receipts are in
[`example-runs`](example-runs/); human decisions there are explicitly piped test
input, not claimed human participation.

## Upstream contract evidence

- [CrewAI execution hooks](https://docs.crewai.com/en/learn/execution-hooks)
- Published CrewAI wheel: `crewai/hooks/dispatch.py`,
  `crewai/hooks/tool_hooks.py`, `crewai/utilities/agent_utils.py`,
  `crewai/crews/utils.py`, and `crewai/crew.py`.
- [agent-hooks specification and SDK](https://github.com/responsibleai/agent-hooks)

The implementation and tests must use published dependencies, never a sibling
CrewAI fork or its added governor/abort-provenance internals.
