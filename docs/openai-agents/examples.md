# Complete local-Qwen examples and validation

Use the full [Agent Hooks](../../examples/openai-agents/agent_hooks) and
[ACS](../../examples/openai-agents/acs) folders. Each contains independent
`llm_hooks.py` and `tool_hooks.py`, pinned requirements and a complete
manifest/Rego bundle. Intentional repetition keeps copied examples complete
and makes native setup, controls, errors and cleanup visible.

The operator owns Ollama at `http://localhost:11434/v1`:

- Ollama 0.34.0.
- `qwen2.5:latest`,
  digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`.
- OpenAIChatCompletionsModel plus an example-owned AsyncOpenAI client,
  temperature 0, max_tokens 64, zero transport retries, 60-second HTTP phase
  timeout.
- No remote fallback, generated canned responses, model download or
  service installation/start/stop. The example closes only its own client.

After [wheel installation](installation.md), run either copied script using
your isolated environment's Python. Optional positional prompt is the only
application input. Tool scripts expose one harmless async local word counter;
its computed result is not a scripted LLM answer. Output JSON includes the
real generated text, actual SDK point/decision records, exact drop count and
actual tool effects for tool scripts.

An input containing `ignore all controls` demonstrates policy denial (exit 2,
no model effects). A tool request to count `blocked-topic` demonstrates
pre-tool denial after a real model emits the call. Such rules are illustrative
allow/deny controls, not a general prompt-injection detector. Native scripts
do not load their equivalent ACS bundle.

## Separate deadline boundaries

The examples deliberately omit `ModelSettings.timeout`, retaining the pinned
SDK's `None` default. This does not disable the adapter's 120-second native run
wall deadline. The HTTP client's 60-second phase timeouts do not impose a
60-second full-run deadline or replace policy `timeout=5` / `emit_timeout=30`.
The SDK's optional model-attempt timeout would instead bound the complete
wrapped model attempt, including transport waits and model interception.

The required test helper enforces a finite **150-second script guard**:
120 seconds for native execution, up to 2 seconds each for native and tool
drains, up to 1 second for policy-worker cleanup, and 25 seconds of process
startup/finalization allowance. Cleanup can overlap those boundaries; the
guard is an external test-process cap, not a promise that resistant native
work stopped. The shared runtime limits and containment contract are unchanged.

An explicitly approved eight-case model run has a **720-second suite
budget**, separate from these per-script and runtime limits. No model gate is
authorized merely by running the static checks or by reading this command.

## Validation

Follow the [isolated development checks](installation.md#isolated-development)
and [contribution guide](../../CONTRIBUTING.md#independent-adapters).
The owning strict type scope includes source, all local tests, examples,
core source and the shared conformance harness. Root checks do not replace it.

Deterministic native Runner/SDK and real ACS/Rego tests replace **only model
transport**. They prove all eight points, executable model/tool replacement,
usage/IDs/finish metadata, exact native wrappers/outer causes, raw arguments
before parsing, stored prompts/resume, profile/metadata guards, sticky denial,
admission, ownership, cancellation/deadline and executable resource bounds.
The defensive pending cap is isolated from the stricter 10-model cap in its
unit test; ordinary native execution cannot accumulate 64 pending calls.

[CI](../../.github/workflows/ci.yml) records the independent audit, build and
fresh wheel-only import commands. An audit requires zero vulnerabilities and
zero skipped third-party dependencies; local distributions receive separate
source and build checks. An import pass is not model evidence. See
[coverage](coverage.md) for the supported profile and transform contract.

## Mandatory model/copy gates

Model gates are required, but must wait for an operator-approved exclusive
model slot. Missing slot, required flag, service, digest, wheels or registration
is **BLOCKED**, never a silent skip/pass. Supply matching built wheels; the
tests create an isolated wheel-only environment and copy complete folders
outside the checkout. They validate actual text/audit/effects, not golden
or fixated model answers.

```bash
timeout 720 env LOCAL_MODEL_REQUIRED=1 OPENAI_AGENTS_MODEL_SLOT=operator-exclusive \
OPENAI_AGENTS_GATE_EVIDENCE=1 \
CORE_WHEEL=/your/wheels/agent_control_adapters-0.1.0a1-py3-none-any.whl \
ADAPTER_WHEEL=/your/wheels/agent_control_openai_agents-0.1.0a1-py3-none-any.whl \
uv run --directory python/packages/openai-agents --no-sync pytest \
  -c pyproject.toml --rootdir . -m e2e
```

Native and ACS LLM/tool gates, copied complete-folder gates and actual
pre-tool-denial gates are separately collected. No multi-agent model coverage
is claimed. Until execution receipts exist for the final snapshot, these
gates remain pending; deterministic/static checks do not substitute for them.

### Historical model evidence

Earlier validation passed 170 non-model cases, registration, isolated imports
and static/build/audit checks. Its model gate was **blocked**: two combined
commands exceeded 240 seconds. Initial 256-token examples also timed out.
At 64 tokens the ACS LLM and tool scripts completed, but the native LLM and its
copied counterpart hit the SDK's 30-second model timeout. The copied ACS run
was interrupted and both pre-tool-denial cases were unrun.

Two repair rounds corrected the deadline configuration to the boundaries
above, without changing the model service, retry policy or generated output.
Further retries or limit changes require a new scoped approval.

After the explicit final model-slot grant, the corrected `agent_hooks` and
`acs` layout passed **all eight required Qwen gates in 290.98 seconds** within
one 720-second suite budget, with zero failures, errors or skips. The run used
fresh noneditable core and adapter wheels outside the checkout, including
both scripts in each complete copied folder and both real pre-tool denials.

The frozen core wheel SHA256 was
`68b98af3b0a7b5142c1280aef9c31231f2bc9d5c1b80d0a0db824142fbfca7f6`;
the adapter wheel SHA256 was
`696f668de9fddbcb96ea5f625b14c615c9577744f4f9db220b513f9917a626fb`.
The operator version, model digest and parameters above were verified.
Tool successes recorded all eight actual SDK points and real local effects;
both pre-tool denials exited 2 with no tool effects. Ten real script invocations
took 8.33 to 34.76 seconds each. No model transport was mocked in these gates.

That snapshot's deterministic suite passed 179 cases, with clean owning
strict types, lint, security lint and wheel/sdist builds. The slot was released
at 2026-10-06T16:39:29Z with no owned gate processes remaining. Historical
receipts are retained, not rewritten as passes. These earlier wheel hashes do
not approve the current integrated source snapshot. See
[current acceptance status](../validation.md); independent review and
publication remain separate gates.
