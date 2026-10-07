"""Actual ACS/Rego evaluation through the real ADK runner and native SDK emitter."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from adk_adapter_test_helpers import POINTS, controlled, native_case
from agent_adapter_conformance import assert_governed_success, make_rego_manifest
from agent_control_adapters import AdapterBlocked, RunRequest
from agent_hooks import Decision


def test_actual_acs_allow_all_eight_points(tmp_path: Path) -> None:
    path = make_rego_manifest(tmp_path)

    async def scenario() -> None:
        case = native_case(use_tool=True)
        result = await controlled(case, policy_file=path).arun(RunRequest({"prompt": "safe"}))
        assert_governed_success(result, points=POINTS)
        assert case.effects == ["native"]
        assert all(record.verdict.decision is Decision.ALLOW for record in result.records)

    asyncio.run(scenario())


@pytest.mark.parametrize("point", POINTS)
def test_actual_acs_deny_each_point_with_real_audit(point: str, tmp_path: Path) -> None:
    path = make_rego_manifest(
        tmp_path,
        f'''
verdict := {{"decision": "deny", "reason": "actual-rego-deny"}} if {{
    input.snapshot.interception_point == "{point}"
}}
''',
    )

    async def scenario() -> None:
        case = native_case(use_tool=True)
        adapter = controlled(case, policy_file=path)
        if point == "agent_shutdown":
            assert (await adapter.arun(RunRequest({"prompt": "safe"}))).output == "native answer"
        else:
            with pytest.raises(AdapterBlocked):
                await adapter.arun(RunRequest({"prompt": "safe"}))
        assert any(
            record.interception_point.value == point
            and record.verdict.decision is Decision.DENY
            and record.verdict.reason == "actual-rego-deny"
            for record in adapter.records
        )
        assert case.effects == (
            ["native"]
            if point
            in (
                "post_tool_call",
                "output",
                "agent_shutdown",
            )
            else []
        )
        assert adapter.cleanup_error is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "point,path,value",
    [
        ("input", "$target.content", "changed input"),
        ("pre_model_call", "$target[1].content", "changed actual request"),
        ("post_model_call", "$target.content", "changed actual model"),
        ("pre_tool_call", "$target.value", "changed actual arguments"),
        ("post_tool_call", "$target.result", "changed actual tool result"),
        ("output", "$target.content", "changed final output"),
    ],
)
def test_actual_acs_transform_writeback(
    point: str,
    path: str,
    value: str,
    tmp_path: Path,
) -> None:
    manifest = make_rego_manifest(
        tmp_path,
        f'''
verdict := {{
    "decision": "transform",
    "transform": {{"path": "{path}", "value": "{value}"}}
}} if {{ input.snapshot.interception_point == "{point}" }}
''',
    )

    async def scenario() -> None:
        case = native_case(use_tool=point in ("pre_tool_call", "post_tool_call"))
        adapter = controlled(case, policy_file=manifest)
        result = await adapter.arun(RunRequest({"prompt": "original"}))
        if point in ("input", "pre_model_call"):
            assert case.model.seen[0].contents[0].parts is not None
            assert case.model.seen[0].contents[0].parts[0].text == value
        elif point == "pre_tool_call":
            assert case.effects == [value]
        elif point == "post_tool_call":
            responses = [
                part.function_response
                for content in case.model.seen[1].contents
                for part in content.parts or []
                if part.function_response is not None
            ]
            assert responses[0].response == {"result": value}
        else:
            assert result.output == value
        assert any(record.verdict.decision is Decision.TRANSFORM for record in result.records)
        assert adapter.cleanup_error is None

    asyncio.run(scenario())


def test_actual_broken_rego_engine_is_fail_closed_not_fake_policy(tmp_path: Path) -> None:
    path = make_rego_manifest(tmp_path, "not valid rego")

    async def scenario() -> None:
        case = native_case(use_tool=True)
        adapter = controlled(case, policy_file=path)
        with pytest.raises(AdapterBlocked) as caught:
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert caught.value.__cause__ is not None
        assert case.model.seen == [] and case.effects == []
        assert adapter.records[0].verdict.decision is Decision.DENY
        assert adapter.records[0].interception_point.value == "agent_startup"
        assert adapter.cleanup_error is None

    asyncio.run(scenario())
