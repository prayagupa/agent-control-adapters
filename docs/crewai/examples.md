# CrewAI controls using Python interceptors or ACS YAML/Rego

See the [adapter guide](README.md) for installation, supported execution,
and lifecycle requirements.

These are complete, standalone replacements for CrewAI's official
[execution-hook](https://docs.crewai.com/v1.15.23/en/learn/execution-hooks),
[LLM-hook](https://docs.crewai.com/v1.15.23/en/learn/llm-hooks) and
[tool-hook](https://docs.crewai.com/v1.15.23/en/learn/tool-hooks) scenarios.
They keep native `Agent`, `Task`, tools and `Crew` setup, but use SDK policies
instead of registering native `@on` functions. The scenarios are credited to
CrewAI; these files supply the complete application and SDK integration.
The broad Crew/Flow walkthrough is not required to learn the adapter.

Each file includes its local model configuration using CrewAI's built-in
`OpenAICompletion`, policies and execution call. There are no hidden
agent/crew factories, custom model wrappers or imports from other examples.

The examples print just their real answer or blocked status, with untrusted text
escaped for terminal safety. Model digest/version probes and detailed JSON
receipts live in [the test recorder](../../tests/record_example.py), not in the
examples.

| Complete example | What it demonstrates |
|---|---|
| [agent_hooks/llm_hooks.py](../../examples/crewai/agent_hooks/llm_hooks.py) | Replace the native response-sanitization hook with an SDK transform |
| [agent_hooks/tool_hooks.py](../../examples/crewai/agent_hooks/tool_hooks.py) | Replace the native destructive-tool hook with an SDK denial |
| [agent_hooks/human_approval.py](../../examples/crewai/agent_hooks/human_approval.py) | Replace the native pre-tool approval gate with an SDK approval resolver |
| [agent_hooks/multi_agent_llm_hooks.py](../../examples/crewai/agent_hooks/multi_agent_llm_hooks.py) | Enforce separate LLM response limits using trusted per-agent policy bindings |
| [acs/tool_hooks.py](../../examples/crewai/acs/tool_hooks.py) | The same destructive-tool control, configured with native ACS YAML/Rego |
| [acs/multi_agent_llm_hooks.py](../../examples/crewai/acs/multi_agent_llm_hooks.py) | Shared YAML and separate researcher/editor LLM response limits |

Native Python policies live in `examples/crewai/agent_hooks/`. ACS applications,
YAML manifests and the Rego bundle live together in `examples/crewai/acs/`.
Both use an `agent_control` runner returned by `with_agent_controls`.

## Minimal customer change

Keep your existing native workflow. Replace the native hook function with a
small `intercept(context) -> Verdict` policy, import `Backend`, `RunRequest`
and `with_agent_controls` from `agent_hooks_adapters`, and change the execution
call. Given an existing `crew`, input dictionary and policy, the migration is:

```diff
- result = crew.kickoff(inputs=inputs)
+ agent_control = with_agent_controls(
+     backend=Backend.CREWAI, runtime=crew, interceptors=[policy]
+ )
+ result = agent_control.run(RunRequest(inputs=inputs))
- print(result.raw)
+ print(result.output)
```

This is a migration diff, not another incomplete example; the six linked files
contain all runnable code. Do not register the old native hook and its replacement
policy together. The wrapper returns a governed runner; it does not modify the
crew so that a later direct `crew.kickoff()` becomes governed. `with_agent_hooks`
and `create_adapter` remain exact compatibility aliases.

The supported alpha profile still requires synchronous sequential execution,
`respect_context_window=False`, and enough iterations for the task (`max_iter=3`
in these examples). Unsupported native features are rejected, not silently
disabled; see [coverage](../coverage.md).

## Run the complete examples

Run from the **repository root**, after `uv sync --locked --group dev`.
For the two ACS examples use `uv sync --locked --group dev --group acs` instead.
Use an already installed/running Ollama with `qwen2.5:latest`:

```bash
curl --fail http://localhost:11434/api/tags
uv run --no-sync python -m examples.crewai.agent_hooks.llm_hooks
uv run --no-sync python -m examples.crewai.agent_hooks.tool_hooks
uv run --no-sync python -m examples.crewai.agent_hooks.human_approval
uv run --no-sync python -m examples.crewai.agent_hooks.multi_agent_llm_hooks
uv run --no-sync python -m examples.crewai.acs.tool_hooks
uv run --no-sync python -m examples.crewai.acs.multi_agent_llm_hooks
```

You can also run any file directly, for example:

```bash
uv run --no-sync python examples/crewai/agent_hooks/llm_hooks.py
```

A copied file can run outside this repository using the same provisioned Python
environment. It still requires the installed adapter, CrewAI and its native
OpenAI-provider dependencies, and the already running local model; "standalone" does not
mean dependency-free.
For an ACS example, copy the **entire `acs/` folder**, including its scripts,
manifests and `policy/` bundle. Each script locates its sibling manifests with
`Path(__file__).parent`, not the working directory. Install the optional
`[crewai,acs]` distribution extras when
using locally built packages outside the workspace; ACS is not needed for the
four Python-policy files.

No downloads or remote fallback occur. The endpoint is deliberately fixed to
`http://localhost:11434/v1`; these examples no longer read `LOCAL_MODEL_BASE_URL`.
`LOCAL_MODEL` selects another installed model, with an omitted tag becoming
`:latest`. Native model requests use a 60-second timeout, a 256-token output limit,
temperature zero, and no SDK retries. Telemetry/tracing is disabled.
Agents explicitly disable automatic context summarization and use `max_iter=3`.
The adapter reserves the last iteration to prevent an unhooked forced-final
model call; these short tasks ordinarily finish after one governed request.

The governance portion is `with_agent_controls(backend=Backend.CREWAI, runtime=crew, ...)`
followed by `agent_control.run(RunRequest(...))`. Successful runs expose `result.output`
and actual SDK audit records. Denials raise the common `AdapterBlocked` error.
The multi-agent example assigns policies to the stable application labels
`researcher` and `editor`, with an explicit trusted mapping to the actual agents.
Native SDK context IDs remain unchanged.

## ACS YAML/Rego controls

The full Python applications are in [acs/tool_hooks.py](../../examples/crewai/acs/tool_hooks.py) and
[acs/multi_agent_llm_hooks.py](../../examples/crewai/acs/multi_agent_llm_hooks.py). They retain the same native model,
agents, tasks and execution call as their Python-policy counterparts. Each ACS
script imports the published `AcsInterceptor` from `agent_control_spec` inside
`main()`, so importing an example module does not require the optional runtime.
The shared policy source switch is:

```diff
- interceptors=[python_policy],
+ interceptors=[AcsInterceptor(str(controls / "tool_hooks.yaml"))],
```

For an agent, replace the policy object while keeping its trusted binding:

```diff
- researcher_policy = AgentTextLimit(max_characters=1600)
+ researcher_policy = AcsInterceptor(str(controls / "researcher.yaml"))
```

The assignment remains `AgentPolicy("researcher", (researcher_policy,))`.
The multi-agent file explicitly constructs separate shared, researcher, and
editor interceptors. Constructor paths come from trusted sibling manifests,
not model or request content. Native construction parses manifests before the
adapter validates the Crew; it is outside the adapter's emission deadlines.
Constructor errors propagate, and Rego bundles still load lazily.

The shorter `policy_file=` convenience API remains available: replace
`interceptors=[policy]` with `policy_file=controls / "tool_hooks.yaml"`, or use
`AgentPolicy("researcher", policy_file=controls / "researcher.yaml")`.
That path validates runtime and agent bindings before loading manifests and
checks the supported ACS version. Explicit native construction uses the
application's installed ACS runtime; these examples use the locked 0.4.0a4
environment described above.

ACS 0.4.0a4 does not ship a Python typing marker. The two imports suppress only
Pyright's `reportMissingTypeStubs` diagnostic; constructor argument and
interceptor compatibility checks remain enabled.

Exactly one source is required per assignment. You can mix a native shared
policy with agent YAML, or shared YAML with native agent policies. There is no
second CrewAI hook mapper, worker, or policy evaluator: the published ACS
interceptor runs through the existing SDK contract.

Every external policy file is included here; none contains a scripted `test`
verdict or a canned model answer:

```text
acs/
  tool_hooks.py             Complete destructive-tool example
  multi_agent_llm_hooks.py   Complete researcher/editor LLM-hook example
  tool_hooks.yaml           Shared destructive-tool policy; binds all eight points
  shared.yaml               Shared no-tools policy for the two text-only agents
  researcher.yaml           Agent-only 1,600-character response limit
  editor.yaml               Agent-only 800-character response limit
  policy/
    controls.rego           Actual Rego code used by the manifests
```

Complete [tool manifest](../../examples/crewai/acs/tool_hooks.yaml):

```yaml
agent_control_specification_version: "0.5.0-alpha.1"
policies:
  destructive_tools:
    type: rego
    bundle: "./policy"
    query: data.controls.tool_guard
intervention_points:
  agent_startup: &check
    policy_target: "$snap.target"
    policy: {id: destructive_tools}
  input: *check
  pre_model_call: *check
  post_model_call: *check
  pre_tool_call: *check
  post_tool_call: *check
  output: *check
  agent_shutdown: *check
```

Complete [shared multi-agent manifest](../../examples/crewai/acs/shared.yaml):

```yaml
agent_control_specification_version: "0.5.0-alpha.1"
policies:
  no_tools:
    type: rego
    bundle: "./policy"
    query: data.controls.no_tools
intervention_points:
  agent_startup: &check
    policy_target: "$snap.target"
    policy: {id: no_tools}
  input: *check
  pre_model_call: *check
  post_model_call: *check
  pre_tool_call: *check
  post_tool_call: *check
  output: *check
  agent_shutdown: *check
```

Complete [researcher manifest](../../examples/crewai/acs/researcher.yaml):

```yaml
agent_control_specification_version: "0.5.0-alpha.1"
policies:
  researcher_text_limit:
    type: rego
    bundle: "./policy"
    query: data.controls.researcher_limit
intervention_points:
  pre_model_call: &check
    policy_target: "$snap.target"
    policy: {id: researcher_text_limit}
  post_model_call: *check
  pre_tool_call: *check
  post_tool_call: *check
```

Complete [editor manifest](../../examples/crewai/acs/editor.yaml):

```yaml
agent_control_specification_version: "0.5.0-alpha.1"
policies:
  editor_text_limit:
    type: rego
    bundle: "./policy"
    query: data.controls.editor_limit
intervention_points:
  pre_model_call: &check
    policy_target: "$snap.target"
    policy: {id: editor_text_limit}
  post_model_call: *check
  pre_tool_call: *check
  post_tool_call: *check
```

Complete [Rego bundle](../../examples/crewai/acs/policy/controls.rego):

```rego
package controls

import rego.v1

default tool_guard := {"decision": "allow"}

tool_guard := {"decision": "deny", "reason": "example:destructive_tool"} if {
    input.snapshot.interception_point == "pre_tool_call"
    not is_string(input.snapshot.tool_call.name)
}

tool_guard := {"decision": "deny", "reason": "example:destructive_tool"} if {
    input.snapshot.interception_point == "pre_tool_call"
    input.snapshot.tool_call.name in {"delete_file", "drop_table", "system_shutdown"}
}

default no_tools := {"decision": "allow"}

no_tools := {"decision": "deny", "reason": "example:no_tools"} if {
    input.snapshot.interception_point == "pre_tool_call"
}

within_limit(limit) if {
    input.snapshot.interception_point == "post_model_call"
    is_string(input.snapshot.target.content)
    count(input.snapshot.target.content) <= limit
}

exceeds_limit(limit) if {
    input.snapshot.interception_point == "post_model_call"
    not is_string(input.snapshot.target.content)
}

exceeds_limit(limit) if {
    input.snapshot.interception_point == "post_model_call"
    count(input.snapshot.target.content) > limit
}

default researcher_limit := {"decision": "allow"}

researcher_limit := {"decision": "allow", "reason": "example:researcher_text_limit"} if {
    within_limit(1600)
}

researcher_limit := {"decision": "deny", "reason": "example:text_limit"} if {
    exceeds_limit(1600)
}

default editor_limit := {"decision": "allow"}

editor_limit := {"decision": "allow", "reason": "example:editor_text_limit"} if {
    within_limit(800)
}

editor_limit := {"decision": "deny", "reason": "example:text_limit"} if {
    exceeds_limit(800)
}
```

The model-generated answers are not replaced by these controls. The multi-agent
file checks actual per-policy reasons on disjoint SDK sessions; summary names
remain `None`, not invented source labels. Tool authorization reads the
host-owned `snapshot.tool_call.name`, never a forged `name` argument.
Use canonical `snapshot.target.content` for response text rather than assuming
ACS's policy-input `target` has the same shape.

Shared YAML binds all eight points. Agent-only YAML needs four model/tool
points. Unbound points deny, with `agent_shutdown` retained as audit-only.
Keep the entire bundle immutable for a run. Manifest parsing is eager; Rego
loading/compilation is lazy, so a broken later policy may fail after prior
effects. These examples use the tested `agent-control-spec==0.4.0a4` Rego
profile. Its Cedar engine expects legacy metadata and fails closed here.
YAML `approval` resolver settings are not wired into host authority; use the
explicit SDK resolver. See [coverage](../coverage.md#acs-policy-source)
and [security](../../SECURITY.md#acs-configuration-is-trusted-code) for native
network, credential, deadline, pool, payload and platform limitations.

There is no custom `BaseLLM` subclass or model-client wrapper. `OpenAICompletion`
is the same built-in provider that CrewAI's `LLM` factory selects for this
OpenAI-compatible endpoint. Calling it directly keeps its provider-specific
options fully typed with pinned CrewAI 1.15.23.

These are one-shot CLI demonstrations. Native HTTP clients and message handling
belong to CrewAI and the OpenAI SDK; the examples do not override their proxy,
redirect or cleanup behavior, or implement a separate HTTP-request budget.
They are not a hardened transport layer for an untrusted model server.

## What to expect

- **LLM hooks:** the local model writes a sentence containing the fictional
  identifier `000-00-0000`; a `post_model_call` transform redacts SSN-shaped
  identifiers before the response reaches the agent. The file checks that real
  sanitization happened. This regex demonstrates the hook, not complete PII/DLP
  protection.
- **Tool hooks:** the real model requests `delete_file`; a `pre_tool_call` policy
  blocks it. The demonstration tool never reads or deletes a file, and raises if
  its body is unexpectedly reached. The example requires the actual tool-policy
  denial rather than treating any unrelated failure as success.
- **Human approval:** the SDK emits a liftable deny before `send_email`. The
  resolver displays the model-generated message with control characters escaped.
  Type `approve` within 15 seconds to allow the preview tool. Any other input,
  EOF, or timeout rejects it before execution. SDK
  callback timeout is 20 seconds; total emission timeout is 30 seconds.
  Despite its native-example name, the demonstration `send_email` tool only
  prints an approved preview: no email is sent.
- **Multiple agents:** the researcher produces notes; the editor consumes them.
  The researcher has a 1,600-character response limit and the editor has an
  800-character limit, represented by **different interceptor instances**. The
  example checks that each policy saw only its bound agent ID.
- **ACS tool controls:** the same real tool request is denied by the shipped
  YAML/Rego bundle before its demonstration-only body executes.
- **ACS multiple agents:** the same generated research/editor workflow has a
  shared no-tools manifest and separate YAML response limits; actual SDK
  verdict reasons show the per-agent policies stayed on disjoint sessions.

```mermaid
flowchart LR
    Shared["Shared no-tools policy"] --> Researcher["Researcher"]
    Shared --> Editor["Editor"]
    RP["Researcher interceptor: 1600 characters"] --> Researcher
    EP["Editor interceptor: 800 characters"] --> Editor
    Researcher -->|real generated notes| Editor
    Editor --> Output["Governed final output"]
```

```mermaid
sequenceDiagram
    participant Model as Local Qwen
    participant SDK as agent-hooks
    participant User as Operator
    participant App as Application
    Model-->>SDK: Generated send_email tool request
    SDK->>User: Escalation with tool arguments and context identity
    User-->>SDK: Approve or reject the displayed draft
    alt explicit approval
        SDK-->>App: Allow preview tool after remaining controls
    else rejection, EOF or timeout
        SDK-->>App: Block run before tool execution
    end
```

This console is a learning example, **not a production approval service**. It
does not authenticate an operator, persist a review ticket, or make the generated
text trustworthy. Approval binds to the displayed tool request's SDK context identity. A
production resolver should authenticate and authorize its approver.

## Deliberate differences from native hooks

Native `@on` hooks can fail open on ordinary callback exceptions. A blocked native
tool call can also let the agent continue. The adapter instead fails closed and
makes a denial sticky for the entire crew run. SDK approval cannot bypass another
policy. Agent selection uses actual host-owned objects, not a model-controlled
name or native role-string filter.

These examples govern crew/model/tool boundaries, not arbitrary Flow steps or
direct LLM/tool calls outside the returned runner.

## Reproduce the evidence

```bash
LOCAL_MODEL_REQUIRED=1 uv run --no-sync pytest -m e2e
```

The e2e suite runs all **16 cases** through the actual entry-point code as
modules and individual files copied outside the repository, including approval,
rejection and EOF, plus the copied YAML/Rego assets for ACS.
It requires real redaction, a real model-requested denied tool, and one approved
preview or no preview after rejection. Native structured tool-call responses can
bypass the textual post-model hook; tool authorization uses `pre_tool_call`,
not a post-model filter.
The test recorder observes the original adapter execution and real HTTP
requests; it does not replace model responses or governance decisions. Copied
files run in Python's isolated mode without adding the repository to
`sys.path`, so they cannot rely on its working directory or `PYTHONPATH`.

The suite pipes explicit decisions into the approval example and labels those
receipts `piped-test-input`. That tests the seam with a real model; it is **not
evidence that a person approved the run**. The fixture endpoint must match the
examples' fixed endpoint.

The recorder emits `RESULT_JSON` receipts containing actual output (or `null`
when blocked), SDK decisions, model name, installed artifact digest, Ollama
runtime version, observed request count, temperature and token limit. It checks
that the model artifact and runtime did not change during the run. Temperature
zero does not guarantee repeatable text. Assertions concern governance,
native model settings, request counts, and isolation, not exact wording.

To retain freshly generated receipts:

```bash
LOCAL_MODEL_REQUIRED=1 EXAMPLE_CAPTURE_DIR=docs/example-runs \
  uv run --no-sync pytest -m e2e
```

Module receipts use `llm-redacted`, `tool-blocked`, `human-approved`,
`human-rejected`, `human-eof`, `multi-agent-llm-hooks`, `acs-tool-blocked` and
`acs-multi-agent-llm-hooks`; copied-file receipts add a
`-standalone` suffix. Existing `allowed`/`blocked` receipts are historical evidence
from the previous generic demos, not evidence for these replacements.

The tool-blocking example verifies denial before the tool body executes, not
non-generation or absence from the model server's internal state. All examples
use harmless synthetic tasks and have no external tool effects.
