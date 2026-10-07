# Contributing

Requires Python **3.11-3.13** and [uv](https://docs.astral.sh/uv/).
Python packages live in `python/packages`; guides in `docs/<framework>`; runnable
applications in `examples/<framework>`. Keep new languages in their own folders.

## Core and CrewAI

Run from the repository root:

```bash
uv sync --frozen --group dev --group acs
uv run --no-sync ruff check python/packages/core python/packages/crewai examples/crewai tests
uv run --no-sync ruff format --check python/packages/core python/packages/crewai examples/crewai tests
uv run --no-sync pyright
uv run --no-sync pytest -m 'not e2e'
uv run --no-sync bandit -r python/packages/core/src python/packages/crewai/src examples/crewai -q
uv run --no-sync pip-audit --skip-editable --progress-spinner off
uv build --no-sources python/packages/core
uv build --no-sources python/packages/crewai
```

The audit deliberately remains failing on the
[known Chroma advisories](SECURITY.md#known-dependency-release-blocker).
`--skip-editable` excludes only local source distributions, not their third-party
dependencies. Do not suppress advisories to make the gate appear green.

## Independent adapters

The root workspace owns only core and CrewAI. Each additional adapter owns its
lockfile and virtual environment. Do not install incompatible framework closures
together or lint all projects using one project's configuration.

From an independent adapter directory under `python/packages/<framework>`:

```bash
uv sync --locked --all-groups --all-extras
uv run --no-sync ruff check src tests ../../../examples/<framework>
uv run --no-sync ruff format --check src tests ../../../examples/<framework>
uv run --no-sync pyright
uv run --no-sync pytest -c pyproject.toml --rootdir . -m 'not e2e'
uv run --no-sync bandit -r src ../../../examples/<framework> -q
uv build --no-sources .
```

Replace `<framework>` with the selected package directory. LangChain also needs
the sibling LangGraph adapter; MAF uses native ACS middleware. See the
[available adapter guides](README.md#get-started) and
[CI](.github/workflows/ci.yml) for exact independent audit/build/import commands.
When relocating an existing environment, add `--reinstall` to the locked sync
to refresh absolute entry-point paths.

## Test expectations

Keep deterministic tests fast and local. Cover native framework boundaries,
fail-closed denial, invalid transforms, actual-object bindings, atomic accounting,
timeout/cancellation, bounded retained state, and cleanup postconditions.
Mocks are allowed at the model boundary only for deterministic tests.

Model tests are marked `e2e`. They require an operator-managed loopback Ollama
service and a real local Kimi/Qwen model; see [validation](docs/validation.md)
before running them. Use the owning project with `pytest -m e2e`, record the exact
snapshot/model/settings, and report unexecuted or skipped cases as missing
evidence. Examples must never present canned model output as a real run.

Current model repair budgets were exhausted. Do not restart the model service,
warm it up, alter provider settings, or retry missing acceptance without a new
scoped run grant.

## Change checklist

- Preserve existing user changes and each package's dependency isolation.
- Keep public contracts typed and source-compatible; validate trusted/untrusted
  boundaries and expose failures rather than returning defaults.
- Keep package READMEs brief, use portable documentation links, and include the
  original MIT notice in independently built wheels and sdists.
- Update the directly related guide, coverage, and runnable assets.
- Run applicable tests, lint, strict types, security scans, and isolated builds.
  Documentation-only edits do not require unrelated model execution.
- Never log secrets or publish raw prompt/tool payloads by default.

Use Conventional Commits with DCO sign-off (`git commit -s`). Publishing source
does not authorize changing repository visibility or uploading to PyPI. Review
the [security policy](SECURITY.md) before requesting production approval.

## Manual releases

The [release workflow](.github/workflows/release.yml) is adapted from
[Agent Hooks](https://github.com/prayagupa/agent-hooks/blob/main/.github/workflows/release.yml).
Only manual dispatch starts it; pushes, tags and GitHub releases do not.
`dry_run=true` is the default: build all seven wheels/sdists, verify their names
and shared version, and save checksummed preview artifacts. It does not publish
or create attestations, tags, or GitHub releases.

Before a non-dry run, clear every [release blocker](docs/validation.md), configure
required reviewers and tag restrictions on the `release` environment, and add
PyPI trusted publishers for all seven `agent-control-*` projects using this
repository, `release.yml`, and that environment. No stored PyPI token is needed.

Select an existing `v<version>` tag and explicitly set `dry_run=false`. Every
package version must match the tag, and the latest push CI run for its exact SHA
must be successful, including the unsuppressed dependency audit. The protected
job verifies downloaded checksums, attests provenance and publishes to PyPI.
Uploads are not atomic; retries skip distributions already published.
