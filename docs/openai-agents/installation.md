# Installation

Requires Python 3.11-3.13. This unpublished alpha uses locally built wheels.

## Isolated development

From the repository root:

```bash
uv sync --project python/packages/openai-agents --locked --group dev --extra acs \
  --index-url https://pypi.org/simple
uv run --directory python/packages/openai-agents --no-sync pyright
uv run --directory python/packages/openai-agents --no-sync pytest \
  -c pyproject.toml --rootdir . -m 'not e2e'
```

The package has its own lock and `.venv`, and uses editable local core
`agent-hooks-adapters==0.1.0a1` from `../core`. The optional `acs` extra pins
`agent-control-spec==0.4.0a4`. Neither selects the core `crewai` extra.
JSON Schema 4.26.0 validates tool arguments; it is not a governance evaluator.

The ordinary suite includes public factory and registration checks against
the matching core. Missing registration is a failure, not a reason to exclude
tests. Internal constructor tests do not replace the public factory gate.

## Wheel installation and copied examples

Build both packages with `uv build python/packages/core` and
`uv build python/packages/openai-agents`. Supply both wheel filenames explicitly
rather than assuming either alpha is available on PyPI:

```bash
uv venv --python 3.13 /your/isolated/example-env
uv pip install --python /your/isolated/example-env/bin/python \
  --index-url https://pypi.org/simple \
  /your/wheels/agent_hooks_adapters-0.1.0a1-py3-none-any.whl \
  '/your/wheels/agent_hooks_openai_agents-0.1.0a1-py3-none-any.whl[acs]' \
  openai==3.24.0
```

Copy the **entire** [Agent Hooks folder](../../examples/openai-agents/agent_hooks) or
[ACS folder](../../examples/openai-agents/acs) outside the checkout, including
both scripts, `requirements.txt`, `controls.yaml` and `policy/guard.rego`.
Each script contains its full native setup, policy selection, error handling
and cleanup; there is no checkout-local helper import.

Install the copied folder's requirements into the same isolated environment
after installing the locally built matching wheels. For published releases,
the requirements alone suffice. Run the copied scripts with that environment's
Python, from any working directory. The ACS manifest resolves its bundle
relative to the copied manifest, not the process working directory.

Native scripts use native interceptors. Their manifest/Rego files are the
equivalent **opt-in ACS** configuration, not an evaluator secretly loaded by
native scripts. The ACS scripts load their own real manifest.

## Opt out / rollback

Controls are explicitly selected per adapter instance. To opt out, do not
construct a governed adapter; use the application-owned native Agent/Runner
directly. This changes the enforcement guarantee and must be an intentional
application decision. Removing the separately installed adapter does not
change the root core/CrewAI environment or uninstall a caller-owned client.
There is no automatic fail-open fallback on missing packages, policies or
model services.
