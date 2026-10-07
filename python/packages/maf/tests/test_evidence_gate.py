"""Deterministic subprocess checks of the required local-model evidence hooks."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]


def _gate(
    tmp_path: Path,
    body: str,
    *,
    required: str,
    overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    shutil.copyfile(Path(__file__).with_name("conftest.py"), tmp_path / "conftest.py")
    shutil.copyfile(
        _ROOT / "tests/agent_adapter_conformance.py",
        tmp_path / "agent_adapter_conformance.py",
    )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\naddopts = -p no:agent_hooks_ctk\nmarkers = e2e: evidence hook canary\n",
        encoding="utf-8",
    )
    (tmp_path / "test_canary.py").write_text("import pytest\n" + body, encoding="utf-8")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["LOCAL_MODEL_REQUIRED"] = required
    if overrides:
        env.update(overrides)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-c", "pytest.ini", "--rootdir", "."],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    "body,required,exit_code",
    [
        ("def test_unit():\n    assert True\n", "1", 1),
        ("@pytest.mark.e2e\ndef test_protocol():\n    assert True\n", "1", 0),
        ("@pytest.mark.e2e\ndef test_protocol():\n    pytest.skip('unavailable')\n", "1", 1),
        (
            "@pytest.mark.e2e\n@pytest.mark.xfail(reason='unproven')\n"
            "def test_protocol():\n    assert False\n",
            "1",
            1,
        ),
        ("@pytest.mark.e2e\ndef test_protocol():\n    pytest.skip('offline')\n", "0", 0),
        ("def test_unit():\n    assert True\n", "invalid", 4),
    ],
)
def test_required_mode_exit_contract(
    body: str,
    required: str,
    exit_code: int,
    tmp_path: Path,
) -> None:
    result = _gate(tmp_path, body, required=required)
    assert result.returncode == exit_code, result.stdout + result.stderr


@pytest.mark.parametrize(
    "base",
    [
        "https://example.invalid/v1",
        "http://user:password@localhost:11434/v1",
        "http://localhost:11434/v1?redirect=remote",
        "http://localhost:11434/v1#fragment",
        "http://localhost:11434/not-v1",
        "http://localhost:0/v1",
        "http://localhost:bad/v1",
    ],
)
def test_model_fixture_rejects_unsafe_endpoints_before_network(
    base: str,
    tmp_path: Path,
) -> None:
    result = _gate(
        tmp_path,
        "@pytest.mark.e2e\ndef test_probe(local_model_settings):\n    assert False\n",
        required="1",
        overrides={"LOCAL_MODEL_BASE_URL": base},
    )
    assert result.returncode == 1
    assert "ValueError" in result.stdout
    assert "ConnectError" not in result.stdout


@pytest.mark.parametrize("model", ["", "../qwen", "qwen\n", "qwen:latest:other"])
def test_model_fixture_rejects_malformed_names_before_network(
    model: str,
    tmp_path: Path,
) -> None:
    result = _gate(
        tmp_path,
        "@pytest.mark.e2e\ndef test_probe(local_model_settings):\n    assert False\n",
        required="1",
        overrides={
            "LOCAL_MODEL": model,
            "LOCAL_MODEL_BASE_URL": "http://localhost:11434/v1",
        },
    )
    assert result.returncode == 1
    assert "LOCAL_MODEL must name an Ollama model" in result.stdout
