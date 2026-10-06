# Contributing

Each `packages/<framework>` directory builds its own wheel and declares only its
own framework plus agent-hooks dependencies. Do not add empty adapter packages
or an umbrella install that pulls in every framework.

For the complete native/ACS change gate, provision the locked optional group:

```bash
uv sync --locked --group dev --group acs
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync pytest -m 'not e2e'
LOCAL_MODEL_REQUIRED=1 uv run --no-sync pytest -m e2e
uv build --package agent-hooks-crewai
uv build --package agent-hooks-adapters
```

Default CI runs deterministic checks without downloading models. Local-model
tests are a separate required evidence tier; a skip is not a pass.

Native-only contributors can use `uv sync --locked --group dev` and
`uv run --no-sync pytest -m 'not e2e and not acs'`. That selector deliberately
does not validate ACS. Selected `acs` tests must fail when the dependency is
missing, not skip or substitute an evaluator. CI installs both groups and audits
the whole dependency set.

`agent-control-spec==0.4.0a4` publishes a Linux x86_64 `cp311-abi3` wheel requiring
glibc >=2.34 and an sdist. Other platforms need a Rust toolchain and source-build
dependencies/network access; do not silently install a toolchain or impose that
build on native-only contributors. This aarch64 host successfully built the
published sdist with rustc/cargo 1.97.1 and glibc 2.43; that is not evidence for
every platform. Bound lock, sync and build operations to 600 seconds, and report
a timeout as missing evidence rather than a pass.

Run these checks before proposing a change. Before widening a dependency range, run the
real-runner integration suite against every supported version. Publish packages
independently; future release tags should identify the adapter, for example
`crewai/v0.1.0a1`. No automatic publication workflow is enabled here.

Keep the root README a framework-neutral product overview. Put framework-specific
installation, usage, coverage, and example guides in `docs/<framework>/`.
Package READMEs should link to those guides; complete runnable applications stay
in `examples/<framework>/`.

Keep public Python APIs typed, run strict pyright, and prefer small explicit
boundary mappings over private framework patches. Unit tests may stub model
I/O. Integration tests must use the real framework runner and SDK emitter.
Model-behavior examples must use a real local model and retain honest receipts.
Copy the entire `examples/crewai/acs/` folder, including its scripts and policy
assets, when testing outside the checkout. Native Python-policy examples are
separate standalone files in `examples/crewai/agent_hooks/`.
Never demonstrate governance using canned model
responses or ACS's scripted `test` policy type.

Every new adapter must document its enforcement coverage, transforms, failure
behavior, concurrency and lifecycle ownership. Observation hooks are not
automatically enforcement hooks. Do not weaken upstream conformance assertions
to make an adapter pass.

Never commit credentials, model weights, virtual environments or private
prompts. Obtain explicit authorization before publishing packages, creating
remote repositories, or changing upstream framework repositories.
