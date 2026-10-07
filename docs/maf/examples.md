# Real local-Qwen examples

Copy the **entire** [ACS folder](../../examples/maf/acs/), including
`requirements.txt`, both YAML manifests, and the `policy/` Rego bundle.
The scripts resolve manifests relative to their own files, not the working
directory. There is no MAF Python-policy (`agent_hooks/`) example.

These are complete native applications: each visibly creates its own
`AsyncOpenAI` client, native serial `OpenAIChatClient`, native `Agent`, ACS
`policy_file`, public `with_agent_controls` adapter, and awaited `RunRequest`.
The tool example also visibly defines its harmless native local tool. No shared
helper hides native setup.

After obtaining approval and an exclusive model slot, use operator-managed Ollama at
`http://localhost:11434/v1`. Do not start, stop, or download models as part of the
tests. The verified artifact is `qwen2.5:latest`, digest
`845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`;
the observed runtime is Ollama `0.34.0`. Both examples use temperature 0,
seed 7 and a 128-token maximum. This reduces variation, not guarantees exact
generated text; model tags can move.

```bash
LOCAL_MODEL=qwen2.5:latest uv run --directory python/packages/maf --no-sync \
  python ../../../examples/maf/acs/llm_hooks.py
LOCAL_MODEL=qwen2.5:latest uv run --directory python/packages/maf --no-sync \
  python ../../../examples/maf/acs/tool_hooks.py
```

- The text example asks Qwen for a short safety-policy explanation and applies
  a real ACS/Rego post-model text-length policy.
- The tool example lets Qwen generate a `delete_file` call. ACS denies the
  actual native `pre_tool_call`, the whole run aborts, and no tool body executes.
  Even if accidentally called, the demonstration tool only records an input;
  it never accesses the filesystem.

Each script prints its actual result and real SDK point/decision/sequence/session
records as JSON lines, followed by the exact audit drop count. Generated wording
is intentionally not reproduced as a canned expected response.
The request counter observes real HTTP transport requests, not projected audit
events. In the pinned connector this run uses Ollama's `/v1/responses` endpoint;
the facade's name is not evidence that it uses `/v1/chat/completions`. Both
examples require one real model request, and native denial settlement must not
make a second request.

## Required evidence gate

```bash
LOCAL_MODEL_REQUIRED=1 LOCAL_MODEL=qwen2.5:latest \
  uv run --directory python/packages/maf --no-sync \
  pytest -c pyproject.toml --rootdir . tests/test_e2e.py -m e2e
```

The three required cases execute the text example, the tool example, and both
examples copied outside the checkout into a fresh wheel-only environment with
the complete policy assets and pinned connectors. No repository working
directory or `PYTHONPATH` fallback is used by the copied scripts.

Required mode fails on unavailable models, skipped e2e cases, or no passing e2e
case. Exploratory offline skips are missing evidence, never a release pass.
HTTP errors, redirects, malformed discovery responses and read/protocol errors
fail in either mode. The probe uses HTTPX2 to match the pinned OpenAI 3 SDK.

Set `MAF_RECEIPTS_DIR` to retain JSON receipts: exact command/cwd/exit, elapsed
time, real stdout/stderr, source and lock hashes, model digest, runtime, package
versions, and model parameters. `MAF_STANDALONE_DIR` optionally selects a new
empty directory outside the checkout for copied scripts, wheel builds and the
isolated environment. The installation receipt includes actual build/install
outputs and wheel hashes. These local artifacts are not committed.
