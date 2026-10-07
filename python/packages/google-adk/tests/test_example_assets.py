"""Real example YAML/Rego assets, tested without pretending model execution occurred."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest
from adk_adapter_test_helpers import controlled, native_case
from agent_hooks import Decision
from agent_hooks_adapters import AdapterBlocked, RunRequest
from agent_hooks_google_adk._profile import profile

_EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "google-adk"


def test_complete_source_only_example_folder_layout() -> None:
    assert {
        path.name for path in _EXAMPLES.iterdir() if path.is_dir() and path.name != "__pycache__"
    } == {"agent_hooks", "acs"}
    expected: dict[str, set[str]] = {
        "agent_hooks": {"llm_hooks.py", "tool_hooks.py", "requirements.txt"},
        "acs": {
            "llm_hooks.py",
            "tool_hooks.py",
            "requirements.txt",
            "llm_hooks.yaml",
            "tool_hooks.yaml",
            "policy/models.rego",
            "policy/tools.rego",
        },
    }
    for folder, assets in expected.items():
        directory = _EXAMPLES / folder
        assert {
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        } == assets


def test_actual_example_model_manifest_trim_transforms() -> None:
    async def scenario() -> None:
        case = native_case()
        response = case.model.responses[0]
        assert response.content is not None and response.content.parts is not None
        response.content.parts[0].text = "  actual native transport text  "
        adapter = controlled(case, policy_file=_EXAMPLES / "acs" / "llm_hooks.yaml")
        result = await adapter.arun(RunRequest({"prompt": "  actual user text  "}))
        assert case.model.seen[0].contents[0].parts is not None
        assert case.model.seen[0].contents[0].parts[0].text == "actual user text"
        assert result.output == "actual native transport text"
        assert sum(record.verdict.decision is Decision.TRANSFORM for record in result.records) == 2

    asyncio.run(scenario())


def test_actual_example_tool_manifest_denies_before_effect() -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        function = profile(case.runner).tools[0].function
        assert inspect.isfunction(function)
        function.__name__ = "delete_file"
        response = case.model.responses[0]
        assert response.content is not None and response.content.parts is not None
        call = response.content.parts[0].function_call
        assert call is not None
        call.name = "delete_file"
        adapter = controlled(case, policy_file=_EXAMPLES / "acs" / "tool_hooks.yaml")
        with pytest.raises(AdapterBlocked):
            await adapter.arun(RunRequest({"prompt": "safe unit input"}))
        assert case.effects == []
        assert any(
            record.interception_point.value == "pre_tool_call"
            and record.verdict.decision is Decision.DENY
            and record.verdict.reason == "example:destructive_tool"
            for record in adapter.records
        )

    asyncio.run(scenario())
