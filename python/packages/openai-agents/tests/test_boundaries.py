"""Raw argument rejection before native parsing and lossless executable output guards."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from agent_control_adapters import AdapterBlocked, RunRequest
from agent_control_adapters.json import JSON
from agent_hooks import ALLOW, Decision, Transform, Verdict
from agents.items import ModelResponse
from agents.usage import Usage
from openai.types.responses import ResponseOutputMessage, ResponseOutputRefusal
from openai_adapter_test_helpers import (
    Policy,
    controlled,
    native_case,
    scope_of,
    text_response,
    tool_response,
)


@pytest.mark.parametrize(
    "raw",
    [
        '{"value":"safe","session_id":"spoof"}',
        '{"value":true}',
        '{"value":1}',
        '{"value":"first","value":"second"}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":9007199254740992}',
        '{"value":null}',
        "{}",
        "[]",
        "null",
        "",
        '{"value":"unterminated}',
        '{"value":' + "[" * 34 + "0" + "]" * 34 + "}",
        '{"value":"' + "x" * 1_048_577 + '"}',
    ],
)
def test_raw_original_args_reject_before_sdk_parser_and_effect(raw: str) -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True, responses=[tool_response(raw)])
        adapter = controlled(case, policies=[Policy()])
        with pytest.raises(AdapterBlocked) as blocked:
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert blocked.value.__cause__ is not None
        assert len(case.model.seen) == 1 and case.effects == []
        assert not any(
            record.interception_point.value == "pre_tool_call" for record in adapter.records
        )
        assert adapter.cleanup_error is None

    asyncio.run(scenario())


def test_changed_valid_args_cannot_rescue_invalid_original() -> None:
    async def scenario() -> None:
        case = native_case(
            use_tool=True,
            responses=[tool_response('{"value":"safe","unexpected":"native would drop this"}')],
        )
        policy = Policy(
            lambda context: (
                Verdict(Decision.TRANSFORM, transform=Transform("$target", {"value": "rescued"}))
                if context["interception_point"] == "pre_tool_call"
                else ALLOW
            )
        )
        with pytest.raises(AdapterBlocked):
            await controlled(case, policies=[policy]).arun(RunRequest({"prompt": "safe"}))
        assert case.effects == []
        assert not any(
            context["interception_point"] == "pre_tool_call" for context in policy.contexts
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "kind", ["empty-list", "empty-text", "refusal", "rich-text", "two-messages"]
)
def test_empty_and_rich_native_output_reject_except_proven_call_only(kind: str) -> None:
    response = text_response("")
    if kind == "empty-list":
        response = ModelResponse(output=[], usage=Usage(), response_id=None)
    elif kind == "refusal":
        response = ModelResponse(
            output=[
                ResponseOutputMessage(
                    id="refusal",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputRefusal(type="refusal", refusal="unsupported")],
                )
            ],
            usage=Usage(),
            response_id=None,
        )
    elif kind == "rich-text":
        message = text_response().output[0]
        assert isinstance(message, ResponseOutputMessage)
        response = replace(
            text_response(), output=[message.model_copy(update={"status": "incomplete"})]
        )
    elif kind == "two-messages":
        response = replace(text_response(), output=text_response().output + text_response().output)

    async def scenario() -> None:
        case = native_case(responses=[response])
        with pytest.raises(AdapterBlocked):
            await controlled(case, policies=[Policy()]).arun(RunRequest({"prompt": "safe"}))
        assert case.effects == []

    asyncio.run(scenario())


def test_one_tool_call_per_model_output_blocks_all_sibling_effects() -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True, responses=[tool_response(count=2)])
        adapter = controlled(case, policies=[Policy()])
        with pytest.raises(AdapterBlocked):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert case.effects == [] and scope_of(adapter).tool_tasks == set()

    asyncio.run(scenario())


@pytest.mark.parametrize("field,value", [("namespace", "remote"), ("async_", True)])
def test_programmatic_remote_native_call_flags_reject(field: str, value: object) -> None:
    response = tool_response()
    response.output[0] = response.output[0].model_copy(update={field: value})

    async def scenario() -> None:
        case = native_case(use_tool=True, responses=[response])
        with pytest.raises(AdapterBlocked):
            await controlled(case, policies=[Policy()]).arun(RunRequest({"prompt": "safe"}))
        assert case.effects == []

    asyncio.run(scenario())


@pytest.mark.parametrize("result", [True, 1, 1.0, {"answer": ["changed"]}])
def test_post_tool_strict_json_replacement_actual_native_consumption(result: JSON) -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        adapter = controlled(
            case,
            policies=[
                Policy(
                    lambda context: (
                        Verdict(Decision.TRANSFORM, transform=Transform("$target", result))
                        if context["interception_point"] == "post_tool_call"
                        else ALLOW
                    )
                )
            ],
        )
        output = await adapter.arun(RunRequest({"prompt": "safe"}))
        assert output.output == "native response"
        tool_inputs = [
            item for item in case.model.seen[1][1] if item.get("type") == "function_call_output"
        ]
        assert len(tool_inputs) == 1
        expected = str(result)
        assert tool_inputs[0].get("output") == expected

    asyncio.run(scenario())


def test_sdk_null_transform_explicitly_blocks_no_silent_original_fallback() -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        adapter = controlled(
            case,
            policies=[
                Policy(
                    lambda context: (
                        Verdict(Decision.TRANSFORM, transform=Transform("$target", None))
                        if context["interception_point"] == "post_tool_call"
                        else ALLOW
                    )
                )
            ],
        )
        with pytest.raises(AdapterBlocked):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert case.effects == [{"value": "native"}]
        assert len(case.model.seen) == 1
        assert any(record.verdict.decision is Decision.DENY for record in adapter.records)

    asyncio.run(scenario())


def test_unknown_native_model_error_propagates_not_governance() -> None:
    async def scenario() -> None:
        case = native_case()
        error = LookupError("unrelated model transport")
        case.model.error = error
        adapter = controlled(case, policies=[Policy()])
        with pytest.raises(LookupError) as caught:
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert caught.value is error
        assert not scope_of(adapter).session.policy_failed
        assert adapter.cleanup_error is None and not case.model.closed

    asyncio.run(scenario())


def test_unknown_native_tool_error_preserves_sdk_wrapper_and_original_cause() -> None:
    from agents.exceptions import UserError
    from agents.tool_context import ToolContext

    error = LookupError("unrelated native tool")

    async def failing(context: ToolContext[object], raw: str) -> str:
        raise error

    async def scenario() -> None:
        case = native_case(use_tool=True)
        case.tool.on_invoke_tool = failing
        adapter = controlled(case, policies=[Policy()])
        with pytest.raises(UserError) as caught:
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert caught.value.__cause__ is error
        assert not scope_of(adapter).session.policy_failed
        assert adapter.cleanup_error is None

    asyncio.run(scenario())
