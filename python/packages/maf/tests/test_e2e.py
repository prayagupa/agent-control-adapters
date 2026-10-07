"""Required real-Qwen examples, including a wheel-only copy outside the checkout."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import httpx2
import pytest

_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLES = _ROOT / "examples" / "maf" / "acs"
_DIGEST = "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"


def _environment(model: str) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"):
        env.pop(name, None)
    env.update(LOCAL_MODEL=model, UV_NO_PYTHON_DOWNLOADS="1")
    return env


def _command(
    command: list[str],
    *,
    cwd: Path,
    model: str,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=_environment(model),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, (
        f"Command failed ({result.returncode}): {command}\n{result.stdout}\n{result.stderr}"
    )
    return result


def _model_receipt(base: str, model: str) -> dict[str, object]:
    if base != "http://localhost:11434/v1":
        raise ValueError("These examples intentionally use http://localhost:11434/v1")
    with httpx2.Client(
        trust_env=False,
        follow_redirects=False,
        timeout=httpx2.Timeout(5, connect=2),
    ) as client:
        tags = client.get("http://localhost:11434/api/tags")
        runtime = client.get("http://localhost:11434/api/version")
        for response in (tags, runtime):
            assert not response.is_redirect
            response.raise_for_status()
        payload: object = tags.json()
        runtime_payload: object = runtime.json()
    assert isinstance(payload, dict)
    models = cast("dict[str, object]", payload).get("models")
    assert isinstance(models, list)
    digest: str | None = None
    for entry in cast("list[object]", models):
        assert isinstance(entry, dict)
        item = cast("dict[str, object]", entry)
        if item.get("name") == model:
            value = item.get("digest")
            assert isinstance(value, str)
            digest = value
    assert digest == _DIGEST, "The approved immutable Qwen artifact is required"
    return {
        "model": model,
        "digest": digest,
        "ollama": runtime_payload,
        "parameters": {"temperature": 0, "seed": 7, "max_tokens": 128},
    }


def _example(
    script: Path,
    *,
    python: Path,
    cwd: Path,
    settings: tuple[str, str],
    receipts: Path,
    label: str,
) -> list[dict[str, object]]:
    base, model = settings
    model_receipt = _model_receipt(base, model)
    metadata = _command(
        [
            str(python),
            "-c",
            (
                "import json;from importlib.metadata import version;"
                "print(json.dumps({name:version(name) for name in "
                "('agent-framework-core','agent-framework-openai','agent-control-spec',"
                "'openai','httpx2','agent-control-adapters','agent-control-maf')}))"
            ),
        ],
        cwd=cwd,
        model=model,
    )
    versions: object = json.loads(metadata.stdout)
    assert versions == {
        "agent-framework-core": "1.20.0",
        "agent-framework-openai": "1.15.0",
        "agent-control-spec": "0.4.0a4",
        "openai": "3.24.0",
        "httpx2": "2.13.1",
        "agent-control-adapters": "0.1.0a1",
        "agent-control-maf": "0.1.0a1",
    }
    model_receipt["versions"] = versions
    command = [str(python), str(script)]
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=cwd,
        env=_environment(model),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    snapshot = [
        *_ROOT.joinpath("python/packages/core/src").rglob("*.py"),
        *_ROOT.joinpath("python/packages/maf/src").rglob("*.py"),
        *_EXAMPLES.rglob("*.py"),
        *_EXAMPLES.rglob("*.yaml"),
        *_EXAMPLES.rglob("*.rego"),
        _EXAMPLES / "requirements.txt",
        _ROOT / "uv.lock",
        _ROOT / "python/packages/maf/uv.lock",
    ]
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / f"{label}.json").write_text(
        json.dumps(
            {
                "command": command,
                "cwd": str(cwd),
                "exit": result.returncode,
                "elapsed_seconds": time.monotonic() - started,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "runtime": model_receipt,
                "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                "source_sha256": {
                    str(path.relative_to(_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in snapshot
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        value: object = json.loads(line)
        assert isinstance(value, dict)
        output.append(cast("dict[str, object]", value))
    audit = [value for value in output if value.get("audit") is True]
    assert audit
    # Native denial settlement reserves context 5 but cannot emit or execute it.
    expected_sequence = (
        [0, 1, 2, 3, 4, 6] if output[0].get("status") == "blocked" else list(range(len(audit)))
    )
    assert [value["sequence"] for value in audit] == expected_sequence
    assert len({value["session_id"] for value in audit}) == 1
    assert output[-1] == {"records_dropped": 0}
    return output


@pytest.fixture
def receipts_directory(tmp_path: Path) -> Path:
    """Persist receipts when requested; otherwise retain them in pytest artifacts."""
    configured = os.getenv("MAF_RECEIPTS_DIR")
    return Path(configured) if configured else tmp_path / "receipts"


def _assert_llm(output: list[dict[str, object]]) -> None:
    assert output[0]["status"] == "allowed"
    assert output[0]["model_requests"] == 1
    assert isinstance(output[0]["output"], str) and output[0]["output"].strip()
    assert {value["point"] for value in output if value.get("audit")} == {
        "agent_startup",
        "input",
        "pre_model_call",
        "post_model_call",
        "output",
        "agent_shutdown",
    }
    assert all(value["decision"] == "allow" for value in output if value.get("audit"))


def _assert_tool(output: list[dict[str, object]]) -> None:
    assert output[0]["status"] == "blocked"
    assert output[0]["tool_effects"] == 0
    assert output[0]["model_requests"] == 1
    audit = [value for value in output if value.get("audit")]
    assert any(value["point"] == "pre_tool_call" and value["decision"] == "deny" for value in audit)
    assert not any(value["point"] in {"post_tool_call", "output"} for value in audit)
    assert {value["point"] for value in audit} >= {
        "agent_startup",
        "input",
        "pre_model_call",
        "post_model_call",
        "agent_shutdown",
    }


@pytest.mark.e2e
def test_local_llm_example(
    local_model_settings: tuple[str, str],
    receipts_directory: Path,
    tmp_path: Path,
) -> None:
    output = _example(
        _EXAMPLES / "llm_hooks.py",
        python=Path(sys.executable),
        cwd=tmp_path,
        settings=local_model_settings,
        receipts=receipts_directory,
        label="llm",
    )
    _assert_llm(output)


@pytest.mark.e2e
def test_local_tool_example(
    local_model_settings: tuple[str, str],
    receipts_directory: Path,
    tmp_path: Path,
) -> None:
    output = _example(
        _EXAMPLES / "tool_hooks.py",
        python=Path(sys.executable),
        cwd=tmp_path,
        settings=local_model_settings,
        receipts=receipts_directory,
        label="tool",
    )
    _assert_tool(output)


@pytest.mark.e2e
def test_copied_examples_outside_checkout(
    local_model_settings: tuple[str, str],
    receipts_directory: Path,
    tmp_path: Path,
) -> None:
    _, model = local_model_settings
    configured = os.getenv("MAF_STANDALONE_DIR")
    isolated = Path(configured) if configured else tmp_path / "standalone"
    assert not isolated.resolve().is_relative_to(_ROOT.resolve())
    isolated.mkdir(parents=True, exist_ok=True)
    copied = isolated / "acs"
    shutil.copytree(_EXAMPLES, copied, ignore=shutil.ignore_patterns("__pycache__"))
    dist = isolated / "dist"
    build_commands = [
        ["uv", "build", "--package", "agent-control-adapters", "--out-dir", str(dist)],
        ["uv", "build", "--directory", str(_ROOT / "python/packages/maf"), "--out-dir", str(dist)],
    ]
    build_receipts: list[dict[str, object]] = []
    for command in build_commands:
        result = _command(command, cwd=_ROOT, model=model)
        build_receipts.append(
            {
                "command": command,
                "exit": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    wheels = sorted(dist.glob("*.whl"))
    assert len(wheels) == 2
    venv = isolated / "venv"
    _command(
        ["uv", "venv", "--python", sys.executable, str(venv)],
        cwd=isolated,
        model=model,
    )
    python = venv / "bin/python"
    install_command = [
        "uv",
        "pip",
        "install",
        "--index-url",
        "https://pypi.org/simple",
        "--python",
        str(python),
        *(str(wheel) for wheel in wheels),
        "-r",
        str(copied / "requirements.txt"),
    ]
    install = _command(install_command, cwd=isolated, model=model)
    receipts_directory.mkdir(parents=True, exist_ok=True)
    (receipts_directory / "standalone-install.json").write_text(
        json.dumps(
            {
                "builds": build_receipts,
                "install": {
                    "command": install_command,
                    "exit": install.returncode,
                    "stdout": install.stdout,
                    "stderr": install.stderr,
                },
                "wheel_sha256": {
                    wheel.name: hashlib.sha256(wheel.read_bytes()).hexdigest() for wheel in wheels
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    probe = _command(
        [
            str(python),
            "-c",
            (
                "import agent_control_adapters,agent_control_maf; "
                "print(agent_control_adapters.__file__);print(agent_control_maf.__file__)"
            ),
        ],
        cwd=isolated,
        model=model,
    )
    assert str(venv) in probe.stdout and str(_ROOT) not in probe.stdout
    for name, assertion in (("llm_hooks.py", _assert_llm), ("tool_hooks.py", _assert_tool)):
        assertion(
            _example(
                copied / name,
                python=python,
                cwd=isolated,
                settings=local_model_settings,
                receipts=receipts_directory,
                label=f"standalone-{Path(name).stem}",
            )
        )
