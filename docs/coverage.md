# Coverage and compatibility

Initial dependency contract: Python 3.11-3.13, CrewAI **1.15.23**, agent-hooks-sdk
**0.1.0a5**. Runtime execution evidence records the actual interpreter separately;
declaring a Python range does not mean every interpreter/platform was tested.
The SDK includes a compiled native core; moving it outside CrewAI does **not**
remove that dependency for applications installing this adapter.

## Supported profile

Call `with_agent_hooks(...).run(...)` on a synchronous, sequential crew.
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

CrewAI's native hook scenarios have [complete SDK-backed replacements](../examples/crewai):
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
