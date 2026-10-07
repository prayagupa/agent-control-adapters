"""Actual local-Qwen receipts and isolated built-wheel copy gates for both graph adapters."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal, cast

import httpx2

BackendName = Literal["langchain", "langgraph"]
SourceName = Literal["agent_hooks", "acs"]
Scenario = Literal["llm", "tool-allow", "tool-deny"]
ROOT = Path(__file__).resolve().parents[4]
DIGEST = "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"


def model_receipt() -> dict[str, object]:
    """Require explicit authority and verify the immutable operator-managed model."""
    if os.getenv("LOCAL_MODEL_REQUIRED") != "1":
        raise RuntimeError("Required model gate needs LOCAL_MODEL_REQUIRED=1; not a skip")
    if os.getenv("ADAPTER_GRAPH_MODEL_SLOT") != "approved":
        raise RuntimeError("Parent must explicitly approve the exclusive model slot")
    with httpx2.Client(trust_env=False, follow_redirects=False, timeout=5) as client:
        tags = client.get("http://localhost:11434/api/tags")
        runtime = client.get("http://localhost:11434/api/version")
        tags.raise_for_status()
        runtime.raise_for_status()
        if tags.is_redirect or runtime.is_redirect:
            raise ValueError("Remote model redirects are unsupported")
        tag_payload: object = tags.json()
        runtime_payload: object = runtime.json()
    if not isinstance(tag_payload, dict) or not isinstance(runtime_payload, dict):
        raise ValueError("Malformed local model metadata")
    models: object = cast("dict[str, object]", tag_payload).get("models")
    if not isinstance(models, list):
        raise ValueError("Local model list is required")
    matches = [
        cast("dict[str, object]", item)
        for item in cast("list[object]", models)
        if isinstance(item, dict)
        and cast("dict[str, object]", item).get("name") == "qwen2.5:latest"
    ]
    if len(matches) != 1 or matches[0].get("digest") != DIGEST:
        raise ValueError("Approved immutable Qwen artifact is required")
    if cast("dict[str, object]", runtime_payload).get("version") != "0.34.0":
        raise ValueError("Approved Ollama 0.34.0 is required")
    return {
        "model": "qwen2.5:latest",
        "digest": DIGEST,
        "ollama": runtime_payload,
        "endpoint": "http://localhost:11434/v1",
        "temperature": 0,
        "timeout_seconds": 60,
        "retries": 0,
    }


def clean_environment() -> dict[str, str]:
    """Prevent editable imports, Python download and inherited external tracing."""
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"):
        environment.pop(name, None)
    environment["UV_NO_PYTHON_DOWNLOADS"] = "1"
    return environment


def command(arguments: list[str], cwd: Path, *, timeout: int) -> subprocess.CompletedProcess[str]:
    """Run a bounded explicit executable with no shell or unbounded retries."""
    result = subprocess.run(
        arguments,
        cwd=cwd,
        env=clean_environment(),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, (
        f"Command failed ({result.returncode}): {arguments}\n{result.stdout}\n{result.stderr}"
    )
    return result


def copied_environment(backend: BackendName, source: SourceName, directory: Path) -> Path:
    """Copy the complete folder and install only built products, never editables."""
    model_receipt()
    wheel_setting = os.getenv("ADAPTER_WHEELHOUSE")
    if not wheel_setting:
        raise RuntimeError("Required copy gate needs the actual built ADAPTER_WHEELHOUSE")
    wheelhouse = Path(wheel_setting).resolve(strict=True)
    distributions = ["agent_hooks_adapters", "agent_hooks_langgraph"]
    if backend == "langchain":
        distributions.append("agent_hooks_langchain")
    products: list[str] = []
    for name in distributions:
        wheels = list(wheelhouse.glob(f"{name}-0.1.0a1-*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"One actual built {name} wheel is required")
        products.append(str(wheels[0]))
    copied = directory / source
    shutil.copytree(ROOT / "examples" / backend / source, copied)
    command(["uv", "venv", "--python", sys.executable, str(copied / ".venv")], copied, timeout=60)
    python = copied / ".venv" / "bin" / "python"
    command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--index-url",
            "https://pypi.org/simple",
            "--find-links",
            str(wheelhouse),
            "-r",
            str(copied / "requirements.txt"),
            *products,
        ],
        copied,
        timeout=180,
    )
    package = f"agent_hooks_{backend}"
    proof = command(
        [
            str(python),
            "-I",
            "-c",
            "import importlib,json,pathlib;"
            f"m=importlib.import_module('{package}');"
            "print(json.dumps({'origin':str(pathlib.Path(m.__file__).resolve())}))",
        ],
        copied,
        timeout=15,
    )
    imported: object = json.loads(proof.stdout)
    assert isinstance(imported, dict)
    origin = cast("dict[str, object]", imported).get("origin")
    assert isinstance(origin, str) and str(copied / ".venv") in origin
    assert str(ROOT) not in origin
    return copied


def run_example(
    backend: BackendName,
    source: SourceName,
    scenario: Scenario,
    receipts: Path,
    *,
    copied: Path | None = None,
) -> None:
    """Run real model I/O and assert actual point/verdict/effect receipts."""
    runtime = model_receipt()
    folder = copied or ROOT / "examples" / backend / source
    python = copied / ".venv/bin/python" if copied else Path(sys.executable)
    script = folder / ("llm_hooks.py" if scenario == "llm" else "tool_hooks.py")
    arguments = [str(python), "-I", str(script), "--synthetic-diagnostics", "local-toy-v1"]
    if scenario != "llm":
        arguments.extend(["--mode", "deny" if scenario == "tool-deny" else "allow"])
    started = time.monotonic()
    result = subprocess.run(
        arguments,
        cwd=folder,
        env=clean_environment(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    elapsed = time.monotonic() - started
    requests: list[dict[str, object]] = []
    for line in result.stderr.splitlines():
        if not line.startswith('{"request_index":'):
            continue
        observed: object = json.loads(line)
        assert isinstance(observed, dict)
        requests.append(cast("dict[str, object]", observed))
    environment_result = command(
        [
            str(python),
            "-I",
            "-c",
            "import importlib,importlib.metadata as m,json,pathlib;"
            "versions={d.metadata['Name']:d.version for d in m.distributions()};"
            "names=['agent_hooks_adapters','agent_hooks_langgraph']"
            + ("+['agent_hooks_langchain'];" if backend == "langchain" else ";")
            + "print(json.dumps({'versions':versions,'origins':"
            "{name:str(pathlib.Path(importlib.import_module(name).__file__).resolve()) "
            "for name in names},'direct_urls':{name:m.distribution(name.replace('_','-'))"
            ".read_text('direct_url.json') for name in names}}))",
        ],
        folder,
        timeout=15,
    )
    installed: object = json.loads(environment_result.stdout)
    assert isinstance(installed, dict)
    wheel_setting = os.getenv("ADAPTER_WHEELHOUSE")
    if not wheel_setting:
        raise RuntimeError("Required real gates need the actual built wheelhouse")
    wheelhouse = Path(wheel_setting).resolve(strict=True)
    files = [
        *(ROOT / "python/packages/core/src").rglob("*.py"),
        *(ROOT / "python/packages/langgraph/src").rglob("*.py"),
        *(ROOT / "python" / "packages" / backend / "src").rglob("*.py"),
        *(ROOT / "examples" / backend).rglob("*.py"),
        *(ROOT / "examples" / backend).rglob("*.yaml"),
        *(ROOT / "examples" / backend).rglob("*.rego"),
        ROOT / "uv.lock",
        ROOT / "python" / "packages" / backend / "uv.lock",
        folder / "requirements.txt",
    ]
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / f"{backend}-{source}-{scenario}-{'copy' if copied else 'native'}.json").write_text(
        json.dumps(
            {
                "command": arguments,
                "exit": result.returncode,
                "elapsed_seconds": elapsed,
                "runtime": runtime,
                "backend": backend,
                "source": source,
                "scenario": scenario,
                "layout": "copy" if copied else "native",
                "request_count": len(requests),
                "observed_requests": requests,
                "installed": installed,
                "wheel_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(wheelhouse.glob("*.whl"))
                },
                "stdout": result.stdout,
                "stderr": result.stderr,
                "source_and_lock_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, result.stdout
    raw: object = json.loads(lines[0])
    assert isinstance(raw, dict)
    payload = cast("dict[str, object]", raw)
    assert payload.get("synthetic_payload_omitted") is False
    assert len(requests) == (2 if scenario == "tool-allow" else 1)
    for index, request in enumerate(requests, start=1):
        assert request.get("request_index") == index
        body = request.get("model_request")
        assert isinstance(body, dict)
        actual = cast("dict[str, object]", body)
        assert actual.get("model") == "qwen2.5:latest"
        assert actual.get("temperature") == 0
        assert actual.get("max_completion_tokens") == (128 if scenario == "llm" else 256)
        assert actual.get("stream") is False
        if scenario != "llm":
            selected: object = "required"
            if backend == "langchain":
                selected = {
                    "type": "function",
                    "function": {"name": "blocked_echo" if scenario == "tool-deny" else "echo"},
                }
            assert actual.get("tool_choice") == (selected if index == 1 else "none")
    expected_status = "blocked" if scenario == "tool-deny" else "allowed"
    assert payload.get("status") == expected_status
    assert payload.get("records_dropped") == 0
    if scenario != "llm":
        assert payload.get("tool_effects") == (0 if scenario == "tool-deny" else 1)
    if expected_status == "allowed":
        output = payload.get("output")
        assert isinstance(output, str) and output.strip()
    raw_records = payload.get("records")
    assert isinstance(raw_records, list) and raw_records
    records: list[dict[str, object]] = []
    for record in cast("list[object]", raw_records):
        assert isinstance(record, dict)
        records.append(cast("dict[str, object]", record))
    points = {record["point"] for record in records}
    required = {"agent_startup", "input", "pre_model_call", "post_model_call", "agent_shutdown"}
    if scenario != "llm":
        required.add("pre_tool_call")
    if expected_status == "allowed":
        required.add("output")
        if scenario != "llm":
            required.add("post_tool_call")
        assert all(record["decision"] == "allow" for record in records)
    else:
        assert payload.get("native_error") == "AdapterBlocked"
        cause = payload.get("sdk_cause")
        assert isinstance(cause, dict)
        assert cast("dict[str, object]", cause).get("type") == "InterceptionBlocked"
        assert payload.get("tool_effect_values") == []
        assert "post_tool_call" not in points and "output" not in points
        assert any(
            record["point"] == "pre_tool_call" and record["decision"] == "deny"
            for record in records
        )
    assert points == required
    assert payload.get("cleanup_error") is False
