"""Real model gates, deliberately not executed before a parent-owned model slot."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from agent_hooks_adapters.json import object_value

pytestmark = pytest.mark.e2e
_EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "google-adk"
_SOURCE_FOLDERS: dict[str, str] = {"native": "agent_hooks", "acs": "acs"}


@pytest.mark.parametrize(
    "source,script,status",
    [
        ("native", "llm_hooks.py", "allowed"),
        ("native", "tool_hooks.py", "blocked"),
        ("acs", "llm_hooks.py", "allowed"),
        ("acs", "tool_hooks.py", "blocked"),
    ],
)
def test_self_contained_real_qwen_example_from_copied_folder(
    source: str,
    script: str,
    status: str,
    tmp_path: Path,
) -> None:
    if os.environ.get("ADK_MODEL_SLOT") != "owned":
        pytest.fail("BLOCKED: parent must grant the exclusive model slot (ADK_MODEL_SLOT=owned)")
    folder = _SOURCE_FOLDERS[source]
    copied = tmp_path / folder
    shutil.copytree(_EXAMPLES / folder, copied, ignore=shutil.ignore_patterns("__pycache__"))
    completed = subprocess.run(
        [sys.executable, str(copied / script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=115,
        check=False,
    )
    print(completed.stdout)
    print(completed.stderr, file=sys.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [
        object_value(json.loads(line))
        for line in completed.stdout.splitlines()
        if line.startswith("{")
    ]
    summary = next(row for row in rows if "status" in row)
    assert summary["status"] == status
    assert summary["model"] == "qwen2.5:latest"
    requests = summary["model_requests"]
    assert type(requests) is int and requests >= 1
    if status == "allowed":
        assert isinstance(summary["output"], str) and summary["output"]
        for point in ("pre_model_call", "post_model_call"):
            assert any(
                row.get("point") == point and row.get("decision") == "transform" for row in rows
            )
    else:
        assert summary["tool_effects"] == 0
        assert any(
            row.get("point") == "pre_tool_call" and row.get("decision") == "deny" for row in rows
        )
        assert not any(row.get("point") == "post_tool_call" for row in rows)
    assert any(row.get("point") == "agent_shutdown" for row in rows)
