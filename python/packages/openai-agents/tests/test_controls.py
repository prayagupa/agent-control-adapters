"""All actual points, ACS/Rego, transforms, audit and trusted-identity conformance."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from agent_adapter_conformance import (
    assert_denied_before_effect,
    assert_governed_success,
    make_rego_manifest,
)
from agent_control_adapters import AdapterBlocked, AgentPolicy, RunRequest
from agent_control_adapters.json import object_value
from agent_hooks import ALLOW, AgentContext, Decision, Transform, Verdict
from agent_hooks.exceptions import InterceptionBlocked
from openai_adapter_test_helpers import Policy, controlled, native_case, scope_of

POINTS = (
    "agent_startup",
    "input",
    "pre_model_call",
    "post_model_call",
    "pre_tool_call",
    "post_tool_call",
    "output",
    "agent_shutdown",
)


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("source", ["native", "acs"])
def test_all_deny_points_whole_run_or_shutdown_audit(
    point: str, source: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        if source == "acs":
            path = make_rego_manifest(
                tmp_path,
                f'''
verdict := {{"decision": "deny", "reason": "private policy reason"}} if {{
    input.snapshot.interception_point == "{point}"
}}''',
            )
            adapter = controlled(case, path=path)
        else:
            adapter = controlled(
                case,
                policies=[
                    Policy(
                        lambda context: (
                            Verdict(Decision.DENY, reason="private policy reason")
                            if context["interception_point"] == point
                            else ALLOW
                        )
                    )
                ],
            )
        if point == "agent_shutdown":
            result = await adapter.arun(RunRequest({"prompt": "call ping"}))
            assert result.output == "native response"
            assert_governed_success(result, points=POINTS)
        else:
            with pytest.raises(AdapterBlocked) as blocked:
                await adapter.arun(RunRequest({"prompt": "call ping"}))
            assert isinstance(blocked.value.__cause__, InterceptionBlocked)
            assert str(blocked.value) == "OpenAI Agents execution blocked by governance."
        assert any(
            record.interception_point.value == point and record.verdict.decision is Decision.DENY
            for record in adapter.records
        )
        if point in (
            "agent_startup",
            "input",
            "pre_model_call",
            "post_model_call",
            "pre_tool_call",
        ):
            assert_denied_before_effect(adapter, point=point, effects=len(case.effects))
        else:
            assert case.effects == [{"value": "native"}]
        if point in ("agent_startup", "input", "pre_model_call"):
            assert case.model.seen == []
        if point in ("post_model_call", "pre_tool_call", "post_tool_call"):
            assert len(case.model.seen) == 1
        assert adapter.cleanup_error is None
        assert not case.model.closed

    asyncio.run(scenario())
    assert "private policy reason" not in caplog.text


def test_real_acs_all_eight_points_and_owned_sequence(tmp_path: Path) -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        adapter = controlled(case, path=make_rego_manifest(tmp_path))
        result = await adapter.arun(RunRequest({"prompt": "call ping"}))
        assert_governed_success(result, points=POINTS)
        assert [record.sequence for record in result.records] == list(range(10))
        assert len({record.session_id for record in result.records}) == 1
        assert result.records_dropped == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "point,path,replacement",
    [
        ("input", "$target.content", "changed input"),
        ("pre_model_call", "$target[0].content", "changed system"),
        ("pre_model_call", "$target[1].content", "changed request"),
        ("post_model_call", "$target.content", "changed model"),
        ("pre_tool_call", "$target.value", "changed arguments"),
        ("post_tool_call", "$target", "changed result"),
        ("output", "$target.content", "changed output"),
    ],
)
def test_real_acs_transforms_consumed_by_native_runner(
    point: str, path: str, replacement: str, tmp_path: Path
) -> None:
    manifest = make_rego_manifest(
        tmp_path,
        f'''
verdict := {{"decision": "transform", "transform": {{
    "path": {json.dumps(path)}, "value": {json.dumps(replacement)}
}}}} if {{
    input.snapshot.interception_point == "{point}"
    {"input.snapshot.target.content != null" if point == "post_model_call" else "true"}
}}''',
    )

    async def scenario() -> None:
        case = native_case(use_tool=True)
        result = await controlled(case, path=manifest).arun(RunRequest({"prompt": "original"}))
        if point == "input":
            assert object_value(case.model.seen[0][1][0])["content"] == replacement
        elif point == "pre_model_call" and path == "$target[0].content":
            assert case.model.seen[0][0] == replacement
        elif point == "pre_model_call":
            assert object_value(case.model.seen[0][1][0])["content"] == replacement
        elif point == "pre_tool_call":
            assert case.effects == [{"value": replacement}]
            assert case.contexts[0].tool_arguments == json.dumps({"value": replacement})
        elif point == "post_tool_call":
            tool_item = next(
                object_value(item)
                for item in case.model.seen[1][1]
                if object_value(item).get("type") == "function_call_output"
            )
            assert tool_item["output"] == replacement
        else:
            assert result.output == replacement

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "point,path,value",
    [
        ("agent_startup", "$target", {"tools_registered": []}),
        ("input", "$target.role", "system"),
        ("input", "$target", {"content": "safe", "role": "user", "agent_id": "spoof"}),
        ("pre_model_call", "$target[0].source", "spoof"),
        ("pre_model_call", "$target[1].native.role", "system"),
        ("post_model_call", "$target.tool_calls[0].args.value", "spoof"),
        ("post_model_call", "$target.finish_reason", "spoof"),
        ("pre_tool_call", "$target", {"value": "safe", "session_id": "spoof"}),
        ("pre_tool_call", "$target.value", True),
        ("output", "$target", {"content": "safe", "trace_id": "spoof"}),
    ],
)
def test_invalid_writeback_and_metadata_fail_closed(point: str, path: str, value: object) -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        policy = Policy(
            lambda context: (
                Verdict(Decision.TRANSFORM, transform=Transform(path, value))
                if context["interception_point"] == point
                else ALLOW
            )
        )
        adapter = controlled(case, policies=[policy])
        with pytest.raises(AdapterBlocked) as blocked:
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert blocked.value.__cause__ is not None
        if point != "output":
            assert case.effects == []
        assert adapter.cleanup_error is None

    asyncio.run(scenario())


def test_engine_error_is_sticky_actual_sdk_record_no_success() -> None:
    def engine_error(context: AgentContext) -> Verdict:
        if context["interception_point"] == "pre_model_call":
            raise RuntimeError("private engine detail")
        return ALLOW

    async def scenario() -> None:
        case = native_case()
        adapter = controlled(case, policies=[Policy(engine_error)])
        with pytest.raises(AdapterBlocked):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert case.model.seen == []
        assert adapter.records
        assert any(record.verdict.decision is Decision.DENY for record in adapter.records)
        scope = scope_of(adapter)
        for point in POINTS[:-1]:
            with pytest.raises(AdapterBlocked):
                await scope.emit(lambda builder, point=point: builder.input(content=point))
        assert adapter.cleanup_error is None

    asyncio.run(scenario())


def test_shared_then_actor_policy_order_actual_binding() -> None:
    order: list[str] = []

    def shared(context: AgentContext) -> Verdict:
        order.append("shared:" + context["interception_point"])
        return ALLOW

    def actor(context: AgentContext) -> Verdict:
        order.append("actor:" + context["interception_point"])
        return ALLOW

    async def scenario() -> None:
        case = native_case()
        adapter = controlled(
            case,
            policies=[Policy(shared)],
            agents={"trusted": case.agent},
            agent_policies=[AgentPolicy("trusted", interceptors=(Policy(actor),))],
        )
        scope = scope_of(adapter)
        assert scope.session.controls.actors[0].native is case.agent
        assert scope.session.controls.actors[0].label == "trusted"
        result = await adapter.arun(RunRequest({"prompt": "safe"}))
        assert result.output == "native response"
        for index in range(0, len(order), 2):
            assert order[index].removeprefix("shared:") == order[index + 1].removeprefix("actor:")
        assert result.records[0].session_id == scope.session.controls.actors[0].session_id

    asyncio.run(scenario())


def test_bounded_detached_real_records_exact_drop_count() -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        adapter = controlled(case, policies=[Policy()], max_records=3)
        result = await adapter.arun(RunRequest({"prompt": "safe"}))
        assert len(result.records) == 3 and result.records_dropped == 7
        assert [record.sequence for record in result.records] == [7, 8, 9]
        assert adapter.records == result.records
        assert adapter.records is not result.records

    asyncio.run(scenario())
