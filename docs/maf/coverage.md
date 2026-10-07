# Supported profile and enforcement

This first profile accepts one native `Agent`, an explicit native
function-invocation chat client, and at most 64 local `FunctionTool` objects.
The native Agent, client, tool functions, schemas, instructions, and options are
fingerprinted and revalidated before admission effects and native callbacks.
Each tool name stays bound to its ordered tool object, function and schema;
swapping names between otherwise unchanged tools also invalidates the profile.
Caller mutation invalidates the scope; an admitted attempt is single-use even
when its prompt or revalidation fails.

| Point | Native enforcement and supported replacement |
| --- | --- |
| `agent_startup` | Host-owned SDK startup; deny before native work; target unchanged |
| `input` | Native middleware; plain user-content replacement only |
| `pre_model_call` | Native projected message list; text-content-only replacement with message count, roles, and tool identities preserved |
| `post_model_call` | Admit plain text or tool-call-only null content; validate raw tool-call arguments before native parsing; text replacement preserves tool calls and finish reason |
| `pre_tool_call` | Native whole-run denial; strictly validate policy-replaced arguments and reject dropped or coerced fields before the actual local tool executes |
| `post_tool_call` | Native result replacement with strict JSON; no rollback of a completed tool effect |
| `output` | Native plain output-content replacement; non-plain/history targets may remain unchanged but cannot be replaced |
| `agent_shutdown` | Host-owned audit-only SDK shutdown on success or failure; not a new execution permission |

**Host-owned Agent instructions are not policy-transform targets.** In the pinned
MAF mapper they remain separate in model options, not in the pre-model projected
message list. Tests prove changed request content, not changed system
instructions. Instructions remain unchanged and fingerprinted. There is no
second mapper to extend this native surface.

After tool use the native `output` target can contain multi-message history.
That unchanged native target is allowed. Any attempted replacement of such a
non-plain target fails closed rather than discarding the change or running with
the original content. Plain output transforms are supported separately.

All governance denials abort the whole run, including tool denials which the
native framework might otherwise handle as a recoverable tool failure. Native
callback failures use fixed, safe model-facing errors; the application boundary
retains original SDK causes for operators. Unrelated SDK/transport exceptions
are not reclassified as governance denials. Actual SDK records, not synthesized
events, provide one caller-loop builder sequence and one session. Audit retention
is bounded with exact eviction/drop counts.

Real OpenAI tool-denial runs also attempt a native conversation-settlement model
callback. Sticky governance failure blocks that callback before another model
request. Its builder context consumes a sequence number without creating a
record: the verified denial audit is `0, 1, 2, 3, 4, 6`, ending in shutdown.
This is not a dropped record (`records_dropped` remains zero), and the adapter
does not fabricate the missing context as a successful model or tool event.
Native settlement warnings can appear on stderr; they are not a successful
recovery or permission to resume the single-use adapter.

### Native output and source-argument admission

Native model content must be plain text. Null content is admitted only for
tool-call-only output; an empty response without tool calls fails closed.
Raw model-generated tool arguments are strictly validated at post-model
admission, before native Pydantic parsing can discard unknown fields. The same
argument-validation helper checks policy-replaced arguments after the actual
pre-tool SDK emission, using a model dump round-trip to reject dropped or
coerced fields.

Invalid original arguments therefore abort before any pre-tool emission or
tool effect. Invalid transformed arguments abort after an actual pre-tool
SDK record but still before the tool body. Both paths preserve whole-run
blocking, shutdown semantics, and the actual SDK audit. Admission failures
do not invent pre-tool records or a post-model `DENY` verdict when the SDK
did not emit one. Host argument validation and upstream policy denial are
distinct failure paths; neither grants permission to execute the tool.

## Explicit exclusions

No streaming, rich/multimodal or structured model outputs; persistence,
stored prompts, resume/conversation/session selectors; MCP, hosted/remote tools,
agent-as-tool or delegation; custom native run/client pipelines; extra
middleware; context providers, compaction/tokenizers, custom schemas/result
parsers, approval modes, tool budgets or arbitrary native additional properties.
Requests contain exactly one nonempty plain `prompt` (at most 65,536 characters);
they cannot forward messages, tools, callbacks, identities, or sessions.
Local tool bodies are trusted application code, not sandboxed by this adapter.

## Lifecycle and resource limits

- Async only: `arun` uses the caller's loop. `run` refuses before admission,
  so the caller can correct to `arun`.
- Native wall deadline: 120 seconds; bounded cancellation/drain: 2 seconds.
  Non-cooperative work stays owned and incomplete cleanup is surfaced.
- At most 10 actual model calls and one tool call per model output.
- Serial tool invocation; at most 64 pending correlations and 256 governed
  native events. These are fixed shared runtime limits, not caller-controlled
  request metadata.
- Default policy timeout 5 seconds and emission timeout 30 seconds. These do
  not replace caller-owned HTTP/tool timeouts or stop daemon-side generation.
- Default audit retention 1,000 real records; `max_records` explicitly bounds
  retention and `records_dropped` reports evictions.

The adapter closes its own native task scope and policy worker, not caller-owned
clients, transports, services, or event loops. Native Agent/client objects remain
usable on the same loop with a fresh adapter. The examples close transports in
`finally` because they created and own them. Cancellation and native errors
cannot grant later permission or erase already completed tool effects.
