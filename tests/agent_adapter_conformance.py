"""Framework-neutral vectors used by independently configured backend test suites."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from pathlib import Path

from agent_control_adapters import Adapter, RunRequest, RunResult
from agent_hooks import Decision, InterceptionRecord

INVALID_PROMPTS: tuple[RunRequest | None, ...] = (
    None,
    RunRequest(),
    RunRequest({}),
    RunRequest({"prompt": ""}),
    RunRequest({"prompt": " "}),
    RunRequest({"prompt": False}),
    RunRequest({"prompt": "x" * 65_537}),
    RunRequest({"prompt": "safe", "session_id": "spoof"}),
    RunRequest({"prompt": "safe", "agent_id": "spoof"}),
    RunRequest({"prompt": "safe", "messages": []}),
    RunRequest({"prompt": "safe", "tools": []}),
    RunRequest({"prompt": "safe", "callbacks": []}),
)


def assert_governed_success(result: RunResult, *, points: Iterable[str]) -> None:
    """Require actual SDK records for every advertised point, never fabricated audit."""
    assert isinstance(result.output, str)
    assert result.records
    assert result.records_dropped >= 0
    assert all(isinstance(record, InterceptionRecord) for record in result.records)
    actual = {record.interception_point.value for record in result.records}
    assert set(points) <= actual
    assert not any(
        record.verdict.decision is Decision.DENY
        and record.interception_point.value != "agent_shutdown"
        for record in result.records
    )


def assert_denied_before_effect(adapter: Adapter, *, point: str, effects: int) -> None:
    """Bind zero-effect denial evidence to the actual SDK point/verdict."""
    assert effects == 0
    assert any(
        record.interception_point.value == point and record.verdict.decision is Decision.DENY
        for record in adapter.records
    )


def assert_graph_example_transport_budget(path: Path) -> None:
    """Verify real graph example constructors retain their finite transport settings."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"OpenAI", "AsyncOpenAI", "ChatOpenAI", "LocalToolModel"}
    ]
    assert len(calls) == 3
    for call in calls:
        keywords = {item.arg: item.value for item in call.keywords}
        assert ast.literal_eval(keywords["timeout"]) == 60
        assert ast.literal_eval(keywords["max_retries"]) == 0
        if "temperature" in keywords:
            assert ast.literal_eval(keywords["temperature"]) == 0
            assert ast.literal_eval(keywords["max_completion_tokens"]) == (
                128 if path.name == "llm_hooks.py" else 256
            )
    if path.name == "tool_hooks.py":
        assertions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "mode != 'allow' or len(effects) != 1"
        ]
        assert len(assertions) == 1


def make_rego_manifest(
    root: Path,
    rule: str = "",
    *,
    points: tuple[str, ...] = (
        "agent_startup",
        "input",
        "pre_model_call",
        "post_model_call",
        "pre_tool_call",
        "post_tool_call",
        "output",
        "agent_shutdown",
    ),
) -> Path:
    """Create real ACS/Rego test assets, never an alternate evaluator or scripted policy."""
    directory = root / "policy"
    directory.mkdir(parents=True)
    (directory / "guard.rego").write_text(
        'package controls\nimport rego.v1\ndefault verdict := {"decision": "allow"}\n' + rule,
        encoding="utf-8",
    )
    manifest = root / "controls.yaml"
    manifest.write_text(
        json.dumps(
            {
                "agent_control_specification_version": "0.5.0-alpha.1",
                "policies": {
                    "guard": {
                        "type": "rego",
                        "bundle": "./policy",
                        "query": "data.controls.verdict",
                    }
                },
                "intervention_points": {
                    point: {"policy_target": "$snap.target", "policy": {"id": "guard"}}
                    for point in points
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest
