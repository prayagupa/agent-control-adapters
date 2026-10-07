"""Mandatory local-Qwen gates; selection without slot/wheels/service fails, never skips."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest
from agent_control_adapters.json import JSON, object_value

_EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "openai-agents"
_DIGEST = "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"


def _operator_model() -> None:
    if os.environ.get("LOCAL_MODEL_REQUIRED") != "1":
        raise RuntimeError("BLOCKED: LOCAL_MODEL_REQUIRED=1 is mandatory for the model gate")
    if not os.environ.get("OPENAI_AGENTS_MODEL_SLOT"):
        raise RuntimeError(
            "BLOCKED: parent-assigned exclusive OpenAI adapter model slot is required"
        )
    for route in ("version", "tags"):
        with urllib.request.urlopen(f"http://localhost:11434/api/{route}", timeout=5) as response:
            raw: object = json.loads(response.read(1_048_576))
        data = object_value(raw)
        if route == "version":
            assert data["version"] == "0.34.0", "BLOCKED: wrong operator-managed Ollama version"
        else:
            models = data["models"]
            assert isinstance(models, list), "BLOCKED: missing operator model inventory"
            assert any(
                object_value(model).get("name") == "qwen2.5:latest"
                and object_value(model).get("digest") == _DIGEST
                for model in models
            ), "BLOCKED: required operator-managed Qwen digest is absent"


@pytest.fixture(scope="module")
def wheel_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Install only matching built wheels into an isolated environment outside the checkout."""
    _operator_model()
    root = tmp_path_factory.mktemp("openai-wheel-model-gates")
    core = os.environ.get("CORE_WHEEL")
    adapter = os.environ.get("ADAPTER_WHEEL")
    if core is None or adapter is None:
        raise RuntimeError("BLOCKED: CORE_WHEEL and ADAPTER_WHEEL are mandatory")
    for path in (Path(core), Path(adapter)):
        if not path.is_file() or path.suffix != ".whl":
            raise RuntimeError("BLOCKED: a required built wheel does not exist")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["UV_CACHE_DIR"] = str(root / "uv-cache")
    env = root / "env"
    subprocess.run(["uv", "venv", str(env)], cwd=root, env=environment, check=True, timeout=30)
    python = env / "bin" / "python"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--index-url",
            "https://pypi.org/simple",
            core,
            adapter + "[acs]",
            "openai==3.24.0",
        ],
        cwd=root,
        env=environment,
        check=True,
        timeout=180,
    )
    subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import agent_control_adapters, agent_control_openai_agents; "
            "from agent_control_adapters import Backend; "
            "assert Backend('openai-agents').value == 'openai-agents'; "
            "assert 'site-packages' in agent_control_adapters.__file__; "
            "assert 'site-packages' in agent_control_openai_agents.__file__",
        ],
        cwd=root,
        env=environment,
        check=True,
        timeout=30,
    )
    return python


def _run_script(python: Path, path: Path, prompt: str | None = None) -> tuple[int, dict[str, JSON]]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    arguments = [str(python), "-I", str(path)]
    if prompt is not None:
        arguments.append(prompt)
    started = time.monotonic()
    process = subprocess.run(
        arguments,
        cwd=python.parents[2],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=150,
    )
    if os.environ.get("OPENAI_AGENTS_GATE_EVIDENCE") == "1":
        print(
            json.dumps(
                {
                    "event": "openai_agents_native_model_gate",
                    "command": arguments,
                    "cwd": str(python.parents[2]),
                    "test_case": os.environ.get("PYTEST_CURRENT_TEST"),
                    "elapsed_seconds": time.monotonic() - started,
                    "exit_code": process.returncode,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                },
                sort_keys=True,
            )
        )
    assert process.stdout.strip(), f"Example returned no JSON: {process.stderr}"
    raw: object = json.loads(process.stdout.strip().splitlines()[-1])
    return process.returncode, object_value(raw)


@pytest.mark.parametrize("expire", [False, True])
def test_script_guard_has_bounded_native_run_and_cleanup_margin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, expire: bool
) -> None:
    """Verify the actual subprocess guard without invoking an example or model."""
    python = tmp_path / "env" / "bin" / "python"
    script = tmp_path / "budget-only.py"
    calls: list[int] = []
    monkeypatch.delenv("OPENAI_AGENTS_GATE_EVIDENCE", raising=False)

    def run(
        arguments: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        assert arguments == [str(python), "-I", str(script), "budget-only"]
        assert cwd == tmp_path and "PYTHONPATH" not in env
        assert env["PYTHONNOUSERSITE"] == "1"
        assert not check and capture_output and text
        assert timeout == 150
        calls.append(timeout)
        if expire:
            raise subprocess.TimeoutExpired(arguments, timeout)
        return subprocess.CompletedProcess(
            arguments, 0, stdout=json.dumps({"status": "harness-only"}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", run)
    if expire:
        with pytest.raises(subprocess.TimeoutExpired):
            _run_script(python, script, "budget-only")
    else:
        code, report = _run_script(python, script, "budget-only")
        assert code == 0 and report == {"status": "harness-only"}
    assert calls == [150]


def _assert_actual_success(report: dict[str, JSON], *, tool: bool) -> None:
    assert report["status"] == "allowed"
    output = report["output"]
    assert isinstance(output, str) and output.strip(), "No actual model text"
    records = report["records"]
    assert isinstance(records, list) and records
    points: set[str] = set()
    for record in records:
        point = object_value(record)["point"]
        assert isinstance(point, str), "Actual SDK point must be a string"
        points.add(point)
    assert {
        "agent_startup",
        "input",
        "pre_model_call",
        "post_model_call",
        "output",
        "agent_shutdown",
    } <= points
    assert all(object_value(record)["decision"] != "deny" for record in records)
    assert report["records_dropped"] == 0
    if tool:
        assert {"pre_tool_call", "post_tool_call"} <= points
        assert isinstance(report["tool_effects"], list) and report["tool_effects"]


@pytest.mark.e2e
@pytest.mark.parametrize("source", [pytest.param("agent_hooks", id="native"), "acs"])
@pytest.mark.parametrize("script", ["llm_hooks.py", "tool_hooks.py"])
def test_local_qwen_native_and_acs_examples(source: str, script: str, wheel_python: Path) -> None:
    code, report = _run_script(wheel_python, _EXAMPLES / source / script)
    assert code == 0, report
    _assert_actual_success(report, tool=script == "tool_hooks.py")


@pytest.mark.e2e
@pytest.mark.parametrize("source", [pytest.param("agent_hooks", id="native"), "acs"])
def test_complete_copied_folders_wheel_only_outside_checkout(
    source: str, wheel_python: Path, tmp_path: Path
) -> None:
    copied = tmp_path / source
    shutil.copytree(_EXAMPLES / source, copied)
    assert (copied / "requirements.txt").is_file()
    assert (copied / "controls.yaml").is_file()
    assert (copied / "policy" / "guard.rego").is_file()
    for script in ("llm_hooks.py", "tool_hooks.py"):
        code, report = _run_script(wheel_python, copied / script)
        assert code == 0, report
        _assert_actual_success(report, tool=script == "tool_hooks.py")


@pytest.mark.e2e
@pytest.mark.parametrize("source", [pytest.param("agent_hooks", id="native"), "acs"])
def test_local_qwen_pre_tool_denial_zero_tool_effects(source: str, wheel_python: Path) -> None:
    code, report = _run_script(
        wheel_python,
        _EXAMPLES / source / "tool_hooks.py",
        "Use word_count on 'blocked-topic' and explain the result.",
    )
    assert code == 2 and report["status"] == "blocked"
    assert report["tool_effects"] == []
    records = report["records"]
    assert isinstance(records, list)
    assert any(
        object_value(record) == {"point": "pre_tool_call", "decision": "deny"} for record in records
    )
