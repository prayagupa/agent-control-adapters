# Security scope

This alpha adapter provides opt-in policy checks at documented CrewAI hook
boundaries. It is not a sandbox, operating-system permission system, transactional
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

## Known dependency release blocker

The October 2, 2026 installed-environment audit reports four advisories affecting
`chromadb==1.1.1`, a mandatory transitive dependency of the pinned CrewAI release:

| Advisory | Reported issue |
|---|---|
| [GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c) | Pre-authentication server code injection |
| [GHSA-36p7-vc44-83pf](https://github.com/advisories/GHSA-36p7-vc44-83pf) | Authenticated server code injection |
| [GHSA-xph7-9rjv-w5fr](https://github.com/advisories/GHSA-xph7-9rjv-w5fr) | Tenant/resource authorization scope missing |
| [GHSA-2wm9-hf6c-p5cr](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr) | Cross-tenant collection access |

The audit reports no fixed version. These are **upstream dependency findings**,
not demonstrated exploit paths through this adapter. The examples do not start
a Chroma server and the supported profile rejects memory/knowledge integration.
Those limits are not a vulnerability waiver: the clean dependency-audit release
gate remains **blocked**. CI does not suppress these advisories or use
`continue-on-error`. Recheck when CrewAI/Chroma publish a compatible remediation.

The unpublished local adapter cannot be looked up on PyPI by `pip-audit`; its
source receives separate static checks and review.

No public vulnerability-reporting endpoint is configured for this local,
unpublished workspace. Before publication, maintainers must configure a private
reporting channel rather than asking reporters to expose vulnerabilities publicly.
