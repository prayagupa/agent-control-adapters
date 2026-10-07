# Security scope

These alpha adapters provide opt-in policy checks at each documented framework
boundary. They are not a sandbox, operating-system permission system, transactional
rollback mechanism, or guarantee against arbitrary trusted Python code.

Treat model output, tool results, prompts and retrieved content as untrusted.
Bind identity and policy selection from application-owned objects. Keep tool
permissions narrow and apply pre-tool authorization before an irreversible
effect. Use application-level timeouts and idempotency for external services.

Synchronous Python policy callbacks cannot be forcibly stopped safely. The
adapter bounds admitted work and reports unsuccessful cleanup; use process
isolation for untrusted or nonterminating callbacks. Read
[coverage](docs/coverage.md) and [architecture](docs/architecture.md).

Do not put secrets or sensitive data in policy reason strings, terminal approval
prompts or example receipts. The console approval example is not authenticated
and must not be used as a production authorization mechanism.

## ACS configuration is trusted code

The optional `policy_file` source uses the published native
`agent-control-spec==0.4.0a4` interceptor. Policy paths, manifests, Rego modules,
data, inheritance and annotators must be owned by the application, never chosen
by a model, tool result or tenant request. Keep all referenced files immutable
for the run. Native manifest parsing is eager, but Rego loading/compilation is
lazy; a later failure does not roll back earlier agent/model/tool effects.

Trusted manifests can reference HTTPS inheritance, configure blocking HTTP
annotators, and select host environment credentials. This adapter does not
filter egress, isolate credentials, or sandbox the dependency. Constructor
inheritance is outside `timeout`/`emit_timeout`; native defaults include 30-second
per-fetch deadlines, a 1-MiB response limit, five redirects and depth 16. These
are dependency limits, not a total construction deadline.

Native Rego defaults to a 5,000-ms deadline; `ACS_OPA_TIMEOUT_MS` and
`ACS_REGO_V0` can change runtime behavior. A native deadline cannot kill
nonpreemptible work. There are two native pools allowing up to 32 abandoned
workers each, plus concurrent work, not a service-wide quota. Allocation failure
or parser stack overflow can abort the process instead of returning a denial.
Use terminating trusted policies, service-level admission and process isolation
where required. Host worker cleanup does not prove all abandoned native work
stopped. `timeout` cannot preempt synchronous ACS; `emit_timeout` bounds caller
waiting and preserves observable host cleanup failure.

ACS limits the full serialized snapshot to 1,048,576 bytes by default, not only
request text. A valid near-limit request can still be refused by the full-context
boundary before any model call. Configure advanced native limits through a
trusted prebuilt interceptor if needed; do not weaken the host's JSON validation.

The adapter does not wire the manifest's `approval` configuration into host
authority. Supply an authenticated `resolver`; YAML `on_timeout: allow` does not
override SDK fail-closed approval or another policy's mandatory deny.
The pinned Cedar runtime expects a legacy envelope absent from canonical SDK
contexts and fails closed; no metadata shim or allow fallback is added.

Actual audit summaries retain the SDK's positional decisions and reasons, with
`name=None`. This host does not register provenance names or promise unique SDK
names. Missing point bindings deny; `agent_shutdown` is audit-only and cannot
undo completed effects. See [coverage](docs/coverage.md#acs-policy-source).

## Known dependency release blocker

The October 5, 2026 installed-environment recheck reports four advisories affecting
`chromadb==1.1.1`, a mandatory transitive dependency of the pinned CrewAI release:

| Advisory | Reported issue |
|---|---|
| [GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c) | Pre-authentication server code injection |
| [GHSA-36p7-vc44-83pf](https://github.com/advisories/GHSA-36p7-vc44-83pf) | Authenticated server code injection |
| [GHSA-xph7-9rjv-w5fr](https://github.com/advisories/GHSA-xph7-9rjv-w5fr) | Tenant/resource authorization scope missing |
| [GHSA-2wm9-hf6c-p5cr](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr) | Cross-tenant collection access |

The audit and current GitHub advisories report no fixed version. PyPI's latest
Chroma release, `1.5.9`, is still in all four affected ranges. CrewAI `1.15.23`
requires `chromadb~=1.1.0`; forcing the newer release would violate that
constraint without fixing the advisories. No compatible dependency-only repair
was verified, so the versions and unsuppressed audit remain unchanged.

These are **upstream dependency findings**,
not demonstrated exploit paths through this adapter. The examples do not start
a Chroma server and the supported profile rejects memory/knowledge integration.
Those limits are not a vulnerability waiver: the clean dependency-audit release
gate remains **blocked**. CI does not suppress these advisories or use
`continue-on-error`. Recheck when CrewAI/Chroma publish a compatible remediation.

The unpublished local adapter cannot be looked up on PyPI by `pip-audit`; its
source receives separate static checks and review.

No private vulnerability-reporting channel is configured. Before making the
repository public, maintainers must configure one rather than asking reporters
to disclose vulnerabilities in public issues.
