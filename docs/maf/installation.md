# Installation and native setup

MAF is independently installed and resolved; do not use the root CrewAI
environment. The supported pins are:

| Component | Version |
| --- | --- |
| `agent-control-adapters` / `agent-control-maf` | `0.1.0a1` |
| `agent-framework-core` | `1.20.0` |
| `agent-control-spec` | `0.4.0a4` |
| Example OpenAI connector | `agent-framework-openai==1.15.0` |
| Example OpenAI SDK / transport | `openai==3.24.0` / `httpx2==2.13.1` |

The native middleware factory is experimental. Other framework versions are not
covered by this alpha. Python 3.11 through 3.13 is supported by the package.
ACS provides a Linux x86_64 wheel with glibc >=2.34; other platforms may require
a Rust source build. A successful local source build is not a portability claim.

From the checkout, provision the owning environment if needed:

```bash
uv sync --directory python/packages/maf --locked
uv run --directory python/packages/maf --no-sync pytest -c pyproject.toml --rootdir . -m 'not e2e'
uv run --directory python/packages/maf --no-sync pyright
```

For an installation outside the checkout, build the core and MAF wheels and
install those wheels together with the pinned example requirements. The
pre-release packages need not already exist on an index:

```bash
uv build --package agent-control-adapters --out-dir /absolute/path/to/wheels
uv build --directory python/packages/maf --out-dir /absolute/path/to/wheels
uv venv /absolute/path/to/demo/.venv
uv pip install --index-url https://pypi.org/simple \
  --python /absolute/path/to/demo/.venv/bin/python \
  /absolute/path/to/wheels/agent_control_adapters-0.1.0a1-py3-none-any.whl \
  /absolute/path/to/wheels/agent_control_maf-0.1.0a1-py3-none-any.whl \
  -r examples/maf/acs/requirements.txt
```

See the [complete scripts](../../examples/maf/acs/) for visible native
`AsyncOpenAI`, `OpenAIChatClient`, `Agent`, tool, policy, and execution setup.
The OpenAI 3 SDK requires HTTPX2; do not pass an `httpx.AsyncClient` from an
incompatible environment.

The explicit native chat client must be constructed with
`function_invocation_configuration={"allow_concurrent_invocation": False}`.
Native detailed tool errors remain disabled (the default). Extra tools,
approval binding overrides, and disabled function invocation are not supported.
Set an explicit model; the adapter does not select or provision a remote model.

`policy_file` and `interceptors` are exclusive sources. Prebuilt exact pinned
native `AcsInterceptor` instances are also accepted, including portable
`AgentPolicy` assignments to actual native Agent objects. Ordinary Python
interceptors, foreign/duplicate bindings, and unsafe native profiles are
rejected before manifest loading or native effects.
