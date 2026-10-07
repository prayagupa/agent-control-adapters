# Closed native profile and enforcement

## Accepted runtime

An existing exact `agents.Agent`, an explicit public `Model` object, string or
`None` instructions, plain-text output, and at most 16 ordinary local
`FunctionTool` instances with unique trusted names. Agent names/labels use the
shared bounded ASCII label rules. The supported profile has one actor.
No default/string model resolution occurs; a supplied Model and its transport
remain trusted application code, not a sandbox. An explicitly supplied model
may use an application-configured provider; examples use local Ollama only.

Reject before policy I/O: Agent.prompt, dynamic instructions, hosted/MCP/tools,
handoffs/delegation, hooks, native guardrails/approval, structured output,
tool-final-output callbacks, custom instance extensions, unsafe model settings,
and unsupported tool schemas. Model retries, arbitrary extra request body/
query/headers/args, stored configuration and parallel tool calls are rejected.
Accepted scalar settings are finite temperature/top-p/penalties, bounded
max_tokens (1..4096), tool choice auto/none/required, include_usage, and an
optional SDK model-attempt timeout in (0,120], with None retaining the native
default. This is separate from the caller-owned client's HTTP phase timeouts.
No caller RunConfig is forwarded.

Native `get_response` also rejects non-None `previous_response_id`,
`conversation_id` or `prompt` before delegating. Streaming always blocks.
Caller request input is exactly `{"prompt": nonempty string}` of length
1..65536; no history, identity, callback, policy or configuration fields.

## Complete pinned FunctionTool inventory

Every dataclass field and instance extension is inspected. Unknown future
fields reject, rather than inheriting a new remote/delegating execution mode.

| Fields | Accepted handling |
| --- | --- |
| name, description | Bounded trusted name/plain text; preserved |
| params_json_schema | Detached strict object schema; copied into the run tool |
| on_invoke_tool | Async callable; run-owned wrapper delegates the original |
| strict_json_schema, is_enabled | Exactly true; dynamic enable functions reject |
| tool_input_guardrails, tool_output_guardrails | None/empty ordinary list; independent clone lists |
| needs_approval, defer_loading | Exactly false |
| timeout_seconds, timeout_error_function | None; no model-visible native timeout fallback |
| timeout_behavior | Known default/error or raise setting, inactive with no native timeout |
| custom_data_extractor, allowed_callers, output_json_schema | None; even explicit direct/programmatic callers reject |
| _output_type_adapter, _failure_error_function | None |
| _use_default_failure_error_function | Boolean; decorated tools require false (`failure_error_function=None`) |
| _is_agent_tool, _is_codex_tool | Exactly false |
| _agent_tool_default_identity, _agent_instance | None |
| _tool_namespace, _tool_namespace_description | None |
| _mcp_title, _tool_origin, _mcp_tool_binding | None |
| _emit_tool_origin | Boolean; preserved, no remote origin is admitted |

Decorated tools must be async with required primitive str/int/float/bool
parameters whose actual annotations agree with the schema. Sync decorators
are rejected because their native implementation uses a default executor,
which cannot be claimed stopped by cancelling the Runner. Raw async
FunctionTool bodies are trusted application code.

The closed schema dialect supports typed primitives, strict objects, arrays,
enum/const and simple numeric/length/item bounds. All object properties are
required and additionalProperties=false. References, defaults, unions,
patterns, formats and other unsupported keywords reject. Original raw JSON
must validate **before** SDK argument parsing; changed arguments also validate.
Duplicate keys, coercion, field dropping, nonfinite numbers, bool/int
equivalence, nonportable integers and oversized JSON reject.

### Accepted private inspection risk

The upstream SDK does not expose a public profile predicate for delegation,
MCP and programmatic modes. This adapter deliberately inspects every pinned
private FunctionTool field. It also recognizes the SDK-generated Agent
`_function_approval_identity` cache only with an exact owner weak-reference
witness and a 32-character hex token. This is not portable approval authority:
native approval/resume remains disabled, and a new run-owned Agent is created.
Ordinary `Agent[T]` typing metadata is accepted; arbitrary extensions reject.
These version-specific checks are a documented compatibility risk. Upgrades
require a fresh field inventory and actual native seam/denial tests.

## Executable transform matrix

| Point | Actual target / write-back |
| --- | --- |
| agent_startup | Real SDK tools_registered notification; allow/deny only; changes block |
| input | Canonical content/user object; changed content becomes actual Runner input |
| pre_model_call | Message list; system_instructions is a separate first system entry with immutable source marker. Native message/item/part metadata is immutable. Text-only changes reconstruct typed real input and the separate native system parameter |
| post_model_call | Canonical content/tool_calls/finish_reason; content-only change reconstructs a real ModelResponse consumed by Runner |
| pre_tool_call | Strict argument object; replacement serialized into the original callback's actual JSON and independent ToolContext |
| post_tool_call | Actual strict JSON callback result; accepted replacement is returned to actual Runner |
| output | Canonical content object; replacement becomes RunResult.output |
| agent_shutdown | Actual emit_unchecked audit; verdict is recorded, never a veto/transform of completed output |

Model routing, names, IDs, usage, response_id, request_id, raw_usage, output item
IDs/status/phase, calls and supported provider metadata survive copies.
The SDK's chat-completions provider_data (model/response_id, plus a supplied
finish_reason) is readonly and preserved in replay. Unknown extensions reject.
The hook's model.id is the actual bound Model class's qualified name, not a
claim to know an abstract Model's remote provider identifier. Pre/post hook
request_id is the host correlation. If native finish metadata is absent,
finish_reason is the explicitly derived text/tool-call category (stop/tool_calls),
not a recovered transport finish reason.

Empty/rich/refusal/reasoning/multimodal outputs reject. A single supported local
function-call-only response is proven and accepted without text. Post-tool
non-string JSON values use the SDK's normal string rendering; this is not a
claim that the next model sees JSON encoding. The pinned Agent Hooks SDK
rejects null-valued transforms; these block rather than silently retaining the
original. Actual null callback values remain JSON values.

## Identity, errors, limits and cleanup

Strong actual-object binding selects one private SDK builder/thread. Shared
policies precede actor-specific policies. No request/tool target selects
agent/session/policy identity. Real InterceptionRecord objects, decisions and
points are retained; no post record is invented after pre denial.

Governance/codec/budget failures become fixed native-boundary AdapterBlocked
raised from None. Tool wrappers may be wrapped by SDK UserError, with the
fixed AdapterBlocked as direct cause. The private original policy/validation
error is reattached only at the outer application boundary. Unrelated SDK/
model/tool exceptions propagate; they are not mislabeled policy denials.
Sticky state and both caller/run-owned profiles are checked before effects
and successful return. Security logs contain fixed categories/trace IDs, not
policy reasons, prompts or tool values.

- Runner: max_turns=10, tracing_disabled=true, sensitive tracing=false,
  max_function_tool_concurrency=1.
- Shared bounds: actors/nodes 16, edges 64, models 10, one tool per model
  output, pending correlations 64, native events 256.
- Actual single-agent sequential execution is narrower than those shared caps.
- Records: max_records 1..100000, exact eviction count, detached reads.
- Strict JSON: depth 32, nodes 10000, UTF-8 bytes 1048576,
  integers within +/- (2^53-1).
- Policy lane: one active job plus eight waiters, finite emit timeout,
  one owned SDK thread; no default-executor policy bridge.
- Native wall deadline: 120 seconds, separate from policy emit_timeout and
  transport timeout. Owned native/tool tasks cancel and drain for at most
  two seconds. Cancellation-resistant work stays strongly retained and
  reports cleanup_error; it is contained, **not claimed stopped**.

Only owned tasks/threads are cleaned up. The caller's loops, clients, tools
and native models stay open and reusable. Native transport libraries can
perform their own executor work; the no-executor claim is specifically for
the policy bridge. Configure application transport/tool timeouts and retries
explicitly. Arbitrary trusted tool code is not preemptible or sandboxed.
A post-tool denial cannot undo an already completed effect; there is no
automatic rollback, retry, success-shaped fallback or later admission.
