"""Mandatory C4 real Runner model/function-call proof before enum/factory registration."""

from __future__ import annotations

import asyncio

import pytest
from agent_adapter_conformance import assert_governed_success
from agent_hooks import ALLOW, AgentContext, Decision, Transform, Verdict
from agent_hooks.exceptions import InterceptionBlocked
from agent_hooks_adapters import AdapterBlocked, RunRequest
from agent_hooks_adapters.json import normalize, object_value
from agents import RunConfig, Runner
from agents.exceptions import UserError
from agents.items import ModelResponse
from agents.run_config import ToolExecutionConfig
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage
from openai_adapter_test_helpers import Policy, controlled, native_case, scope_of


def test_c4_native_runner_model_tool_replacements_usage_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        original_first, original_last = case.model.responses

        def transform(context: AgentContext) -> Verdict:
            point = context["interception_point"]
            if point == "input":
                return Verdict(Decision.TRANSFORM, transform=Transform("$target.content", "input"))
            if point == "pre_model_call":
                messages = normalize(context["target"])
                assert isinstance(messages, list)
                replacement = [object_value(item) for item in messages]
                for item in replacement:
                    if isinstance(item["content"], str):
                        item["content"] = "rewritten " + item["content"]
                return Verdict(Decision.TRANSFORM, transform=Transform("$target", replacement))
            if (
                point == "post_model_call"
                and object_value(context["target"])["content"] is not None
            ):
                return Verdict(
                    Decision.TRANSFORM, transform=Transform("$target.content", "model replacement")
                )
            if point == "pre_tool_call":
                return Verdict(
                    Decision.TRANSFORM, transform=Transform("$target.value", "argument replacement")
                )
            if point == "post_tool_call":
                return Verdict(
                    Decision.TRANSFORM, transform=Transform("$target", "tool replacement")
                )
            return ALLOW

        adapter = controlled(case, policies=[Policy(transform)])
        scope = scope_of(adapter)
        actual_responses: list[ModelResponse] = []
        original_after = scope.after_model

        async def capture(correlation: str, response: ModelResponse) -> ModelResponse:
            actual = await original_after(correlation, response)
            actual_responses.append(actual)
            return actual

        monkeypatch.setattr(scope, "after_model", capture)
        result = await adapter.arun(RunRequest({"prompt": "caller prompt"}))
        assert result.output == "model replacement"
        assert case.effects == [{"value": "argument replacement"}]
        assert case.model.seen[0][0] == "rewritten Trusted system text"
        assert case.model.seen[0][1] == [{"role": "user", "content": "rewritten input"}]
        tool_item = next(
            object_value(item)
            for item in case.model.seen[1][1]
            if object_value(item).get("type") == "function_call_output"
        )
        assert tool_item["output"] == "rewritten tool replacement"
        assert isinstance(actual_responses[0].output[0], ResponseFunctionToolCall)
        for original, actual in zip(case.model.responses, actual_responses, strict=True):
            assert actual is not original
            assert actual.usage == original.usage and actual.usage is not original.usage
            assert actual.response_id == original.response_id
            assert actual.request_id == original.request_id
            assert (
                actual.raw_usage == original.raw_usage
                and actual.raw_usage is not original.raw_usage
            )
            assert actual.output[0].id == original.output[0].id
            assert isinstance(actual.output[0], ResponseOutputMessage | ResponseFunctionToolCall)
            assert isinstance(original.output[0], ResponseOutputMessage | ResponseFunctionToolCall)
            assert actual.output[0].status == original.output[0].status
        assert original_first.output[0].model_dump() == actual_responses[0].output[0].model_dump()
        assert isinstance(original_last.output[0], ResponseOutputMessage)
        assert original_last.output[0].content[0].model_dump()["text"] == "native response"
        assert case.agent.model is case.model and case.agent.tools == [case.tool]
        assert case.agent.instructions == "Trusted system text"
        assert case.tool.params_json_schema["properties"] == {"value": {"type": "string"}}
        assert not case.model.closed
        assert_governed_success(
            result,
            points=(
                "agent_startup",
                "input",
                "pre_model_call",
                "post_model_call",
                "pre_tool_call",
                "post_tool_call",
                "output",
                "agent_shutdown",
            ),
        )
        assert adapter.cleanup_error is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "point,native_type",
    [
        ("pre_model_call", AdapterBlocked),
        ("post_model_call", AdapterBlocked),
        ("pre_tool_call", UserError),
        ("post_tool_call", UserError),
    ],
)
def test_c4_native_wrapper_denials_exact_sdk_causes(
    point: str, native_type: type[Exception], caplog: pytest.LogCaptureFixture
) -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)

        def deny(context: AgentContext) -> Verdict:
            return (
                Verdict(Decision.DENY, reason="sensitive original policy reason")
                if context["interception_point"] == point
                else ALLOW
            )

        adapter = controlled(case, policies=[Policy(deny)])
        scope = scope_of(adapter)
        scope.session.begin()
        agent = scope.clone()
        primary: BaseException | None = None
        try:
            with pytest.raises(native_type) as caught:
                await Runner.run(
                    agent,
                    "private prompt",
                    max_turns=10,
                    run_config=RunConfig(
                        tracing_disabled=True,
                        trace_include_sensitive_data=False,
                        tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1),
                    ),
                )
            primary = caught.value
            fixed = caught.value.__cause__ if isinstance(caught.value, UserError) else caught.value
            assert isinstance(fixed, AdapterBlocked)
            assert fixed.__cause__ is None and fixed.__suppress_context__
            assert str(fixed) == "OpenAI Agents execution blocked by governance."
            assert isinstance(scope.session.last_error, InterceptionBlocked)
            assert "sensitive original policy reason" not in str(caught.value)
            assert case.effects == ([{"value": "native"}] if point == "post_tool_call" else [])
            assert len(case.model.seen) == (0 if point == "pre_model_call" else 1)
        finally:
            await scope.drain_tools(primary)
            await scope.session.aclose(primary)

    asyncio.run(scenario())
    assert "sensitive original policy reason" not in caplog.text


def test_c4_outer_error_reattaches_private_original_sdk_reason() -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        adapter = controlled(
            case,
            policies=[
                Policy(
                    lambda context: (
                        Verdict(Decision.DENY, reason="private operator reason")
                        if context["interception_point"] == "pre_tool_call"
                        else ALLOW
                    )
                )
            ],
        )
        with pytest.raises(AdapterBlocked) as blocked:
            await adapter.arun(RunRequest({"prompt": "Call ping"}))
        assert isinstance(blocked.value.__cause__, InterceptionBlocked)
        assert case.effects == []
        assert str(blocked.value) == "OpenAI Agents execution blocked by governance."

    asyncio.run(scenario())
