"""Run the actual example entry points, not fake model responses or callback-only tests."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]


def _object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


@pytest.mark.e2e
@pytest.mark.parametrize("standalone", [False, True], ids=["module", "standalone"])
@pytest.mark.parametrize(
    ("example", "operator_input", "status", "receipt_name"),
    [
        ("llm_hooks", "", "allowed", "llm-redacted"),
        ("tool_hooks", "", "blocked", "tool-blocked"),
        ("human_approval", "approve\n", "allowed", "human-approved"),
        ("human_approval", "reject\n", "blocked", "human-rejected"),
        ("human_approval", "", "blocked", "human-eof"),
        ("multi_agent", "", "allowed", "multi-agent"),
    ],
)
def test_real_local_example(
    local_model_client: OpenAI,
    local_model_name: str,
    standalone: bool,
    example: str,
    operator_input: str,
    status: str,
    receipt_name: str,
    tmp_path: Path,
) -> None:
    """Assert governance outcomes without expecting particular generated words."""
    assert not local_model_client.is_closed()
    assert str(local_model_client.base_url).rstrip("/") == "http://localhost:11434/v1", (
        "These examples use the fixed local Ollama endpoint"
    )
    env = {
        **os.environ,
        "LOCAL_MODEL": local_model_name,
        "OTEL_SDK_DISABLED": "true",
        "CREWAI_TRACING_ENABLED": "false",
        "CREWAI_TELEMETRY_ENABLED": "false",
        "PYTHONUNBUFFERED": "1",
    }
    command = [sys.executable, "-I", str(ROOT / "tests" / "record_example.py"), example]
    if standalone:
        copied_example = tmp_path / f"{example}.py"
        shutil.copyfile(ROOT / "examples" / "crewai" / f"{example}.py", copied_example)
        command.extend(["--script", str(copied_example)])
        working_directory = tmp_path
    else:
        working_directory = ROOT
    completed = subprocess.run(
        command,
        cwd=working_directory,
        env=env,
        input=operator_input,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    transcript = completed.stdout + "\nSTDERR:\n" + completed.stderr
    (tmp_path / f"{receipt_name}.txt").write_text(transcript)
    assert completed.returncode == 0, transcript
    receipts = [
        line.removeprefix("RESULT_JSON: ")
        for line in completed.stdout.splitlines()
        if line.startswith("RESULT_JSON: ")
    ]
    assert len(receipts) == 1, transcript
    result = _object(json.loads(receipts[0]))
    assert result["status"] == status
    metadata = _object(result["model"])
    assert metadata["model"] == local_model_name
    assert isinstance(metadata["digest"], str) and len(metadata["digest"]) == 64
    assert isinstance(metadata["requests"], int) and 1 <= metadata["requests"] <= 6
    assert metadata["temperature"] == 0
    assert metadata["max_tokens"] == 256
    raw_decisions = result["decisions"]
    assert isinstance(raw_decisions, list) and raw_decisions
    decisions = [_object(item) for item in cast(list[object], raw_decisions)]
    if example in {"llm_hooks", "multi_agent"} or status == "allowed":
        assert any(item["point"] == "post_model_call" for item in decisions)
    if status == "allowed":
        assert isinstance(result["output"], str) and result["output"]
        assert not any(item["decision"] == "deny" for item in decisions)
    else:
        assert result["output"] is None
        assert any(item["decision"] == "deny" for item in decisions)
    details = _object(result["details"])
    if example == "llm_hooks":
        output = result["output"]
        assert isinstance(output, str) and "[REDACTED]" in output
        assert re.search(r"\b\d{3}-\d{2}-\d{4}\b", output) is None
        assert any(
            item["point"] == "post_model_call" and item["decision"] == "transform"
            for item in decisions
        )
    if example == "tool_hooks":
        assert any(
            item["point"] == "pre_tool_call" and item["reason"] == "example:destructive_tool"
            for item in decisions
        )
        assert not any(item["point"] == "post_tool_call" for item in decisions)
        assert "Blocked: delete_file did not execute." in completed.stdout
    if example == "human_approval":
        assert details["input_mode"] == "piped-test-input"
        resolution = "approval" if status == "allowed" else "rejection"
        assert any(
            item["point"] == "pre_tool_call" and item["resolved_by"] == resolution
            for item in decisions
        )
        previews = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("Approved preview (not sent): ")
        ]
        assert len(previews) == (1 if status == "allowed" else 0)
        assert any(item["point"] == "post_tool_call" for item in decisions) is (status == "allowed")
    expected_requests = (
        2
        if example == "multi_agent" or (example == "human_approval" and status == "allowed")
        else 1
    )
    assert metadata["requests"] == expected_requests
    if example == "multi_agent":
        assert metadata["requests"] >= 2
        assert "Per-agent policies stayed isolated." in completed.stdout
        assert (
            len({item["session_id"] for item in decisions if item["point"] == "post_model_call"})
            == 2
        )
        assert len(cast(list[object], details["task_outputs"])) == 2
    capture_dir = os.getenv("EXAMPLE_CAPTURE_DIR")
    if capture_dir:
        destination = Path(capture_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        suffix = "-standalone" if standalone else ""
        (destination / f"{receipt_name}{suffix}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
