# Validation and release status

**Alpha source only; not production-approved.** Package metadata, readable
documentation, and deterministic tests do not replace model acceptance,
independent reviews, or a clean dependency audit.

## Required checks

[CI](../.github/workflows/ci.yml) runs owning-project Ruff lint/format, strict
Pyright, deterministic/native-framework tests, package builds, and dependency
audits. Independent adapters also exercise isolated, noneditable wheel imports.
Use the locked commands in [CONTRIBUTING](../CONTRIBUTING.md); do not combine
incompatible framework environments.

The Python-directory relocation baseline passed **1,473 deterministic/native
tests** across six environments, with no test skips, plus six strict type scopes,
per-project lint, static source scanning, seven wheel/sdist builds, and isolated
imports. That baseline predates the public-documentation/metadata cleanup and is
not a receipt for a later commit. Each source push must validate its exact staged
snapshot and report its exact-SHA CI separately.

Model cases are marked `e2e` and intentionally excluded from the ordinary
deterministic CI run. Deselection is **missing model evidence**, not a pass.
Structural checks validate paths, manifests, licensing, and documentation links;
they do not prove policy enforcement against a live model.

## Open release blockers

| Gate | Status |
|---|---|
| CrewAI dependency audit | Four unresolved Chroma advisories: two critical, two high |
| Final real-model acceptance | Incomplete; failed and unexecuted cases remain |
| Independent architecture/code/security approval | No verified current-snapshot approval recorded |
| Private vulnerability reporting | A working private channel must be configured before public release |

The [security policy](../SECURITY.md#known-dependency-release-blocker) lists the
advisories. No patched compatible dependency was verified. The audit remains
unsuppressed; an explicitly accepted private alpha-source push is not a waiver
for production deployment or public release.

## Real-model evidence and limits

The most recent refresh did **not** finish:

- Four root cases passed, then a CrewAI human-approved example timed out; eleven
  remaining root cases were not executed.
- Remaining MAF, Google ADK, and OpenAI Agents run grants were withheld.
- The final sixteen LangChain/LangGraph model cases remain unexecuted.
- A copied LangChain tool-allow diagnostic returned HTTP 200 with empty assistant
  content, no tool call/arguments, `stop`, and 23 completion tokens. Actual tool
  effects were zero. A successful HTTP response is not a successful tool example.

Previous bounded repair budgets were exhausted. Do not automatically retry,
change provider settings/assertions, warm up a model, or operate the model service
without a new scoped run grant. Documentation/layout/metadata changes do not
authorize those actions.

Recorded local-model provenance:

- Model: `qwen2.5:latest`
- Digest: `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`
- Ollama: `0.34.0`
- Endpoint: `http://localhost:11434/v1`

[Historical receipts](example-runs/README.md) belong to earlier snapshots. Do not
reuse them as current acceptance or rewrite them into canned examples.

## Scope of assurance

Tests cover fail-closed results, actual-object identity bindings, strict JSON,
native admission, atomic budgets, model/tool correlation, bounded worker/record
retention, repeated timeout/cancellation, and explicit cleanup failure. Previously
repaired areas include MAF tool name/function/schema permutation, host-owned trace
correlation, and opt-in graph diagnostics privacy.

These checks do not make arbitrary Python callbacks preemptible, undo completed
tool effects, secure an unauthenticated demo approval service, or govern calls
made outside the adapter. See [coverage](coverage.md) and [security](../SECURITY.md).
