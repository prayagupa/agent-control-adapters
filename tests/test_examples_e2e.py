"""Run the actual example entry points, not fake model responses or callback-only tests."""

from __future__ import annotations

import json
import os
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
@pytest.mark.parametrize(
    ("example", "operator_input", "status", "receipt_name"),
    [
        ("allowed", "", "allowed", "allowed"),
        ("blocked", "", "blocked", "blocked"),
        ("human_approval", "approve\n", "allowed", "human-approved"),
        ("human_approval", "reject\n", "blocked", "human-rejected"),
        ("human_approval", "", "blocked", "human-eof"),
        ("multi_agent", "", "allowed", "multi-agent"),
    ],
)
def test_real_local_example(
    local_model_client: OpenAI,
    local_model_name: str,
    example: str,
    operator_input: str,
    status: str,
    receipt_name: str,
    tmp_path: Path,
) -> None:
    """Assert governance outcomes without expecting particular generated words."""
    assert not local_model_client.is_closed()
    env = {
        **os.environ,
        "LOCAL_MODEL": local_model_name,
        "OTEL_SDK_DISABLED": "true",
        "CREWAI_TRACING_ENABLED": "false",
        "CREWAI_TELEMETRY_ENABLED": "false",
        "PYTHONUNBUFFERED": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-m", f"examples.crewai.{example}"],
        cwd=ROOT,
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
    assert isinstance(metadata["requests"], int) and metadata["requests"] >= 1
    raw_decisions = result["decisions"]
    assert isinstance(raw_decisions, list) and raw_decisions
    decisions = [_object(item) for item in cast(list[object], raw_decisions)]
    assert any(item["point"] == "post_model_call" for item in decisions)
    if status == "allowed":
        assert isinstance(result["output"], str) and result["output"]
        assert not any(item["decision"] == "deny" for item in decisions)
    else:
        assert result["output"] is None
        assert any(item["decision"] == "deny" for item in decisions)
    details = _object(result["details"])
    if example == "human_approval":
        assert details["input_mode"] == "piped-test-input"
        expected = "approve" if status == "allowed" else "reject"
        assert details["operator_decisions"] == [expected]
        resolution = "approval" if status == "allowed" else "rejection"
        assert any(item["resolved_by"] == resolution for item in decisions)
    if example == "multi_agent":
        assert metadata["requests"] >= 2
        assert details["researcher_policy_agents"] == [details["researcher_agent_id"]]
        assert details["editor_policy_agents"] == [details["editor_agent_id"]]
        assert details["researcher_agent_id"] != details["editor_agent_id"]
        assert len(cast(list[object], details["task_outputs"])) == 2
    capture_dir = os.getenv("EXAMPLE_CAPTURE_DIR")
    if capture_dir:
        destination = Path(capture_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{receipt_name}.json").write_text(json.dumps(result, indent=2) + "\n")
