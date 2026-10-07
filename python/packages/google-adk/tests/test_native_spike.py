"""First actual-runner model replacement and native wrapper-denial proof."""

from __future__ import annotations

import asyncio

import pytest
from adk_adapter_test_helpers import Control, controlled, native_case
from agent_adapter_conformance import assert_denied_before_effect
from agent_control_adapters import AdapterBlocked, RunRequest
from agent_control_adapters.json import object_value
from agent_control_google_adk._profile import native_field


def test_actual_runner_model_replacements_preserve_system_usage_and_ids() -> None:
    async def scenario() -> None:
        case = native_case()
        model_pre = Control(
            "pre_model_call",
            replacement=[
                {"role": "system", "content": "Replaced trusted system"},
                {"role": "user", "content": "Replaced actual user"},
            ],
        )
        model_post = Control(
            "post_model_call",
            replacement={
                "content": "Replaced actual model text",
                "tool_calls": [],
                "finish_reason": "stop",
            },
        )
        adapter = controlled(case, interceptors=[model_pre, model_post])
        result = await adapter.arun(RunRequest({"prompt": "original prompt"}))
        assert result.output == "Replaced actual model text"
        assert (
            native_field(case.model.seen[0].config, "system_instruction")
            == "Replaced trusted system"
        )
        assert case.model.seen[0].contents[0].parts is not None
        assert case.model.seen[0].contents[0].parts[0].text == "Replaced actual user"
        assert case.model.responses[0].usage_metadata is not None
        assert case.model.responses[0].usage_metadata.total_token_count == 5
        contexts = [object_value(context) for context in model_pre.seen]
        pre = next(
            context for context in contexts if context["interception_point"] == "pre_model_call"
        )
        post = next(
            context for context in contexts if context["interception_point"] == "post_model_call"
        )
        assert pre["model"] == post["model"]
        assert case.agent.instruction == "Trusted system text"
        assert case.runner.plugin_manager.plugins == []
        assert adapter.cleanup_error is None

    asyncio.run(scenario())


@pytest.mark.parametrize("point", ["pre_model_call", "post_model_call", "pre_tool_call"])
def test_actual_runner_wrapper_denial_aborts_before_tool_effect(point: str) -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        adapter = controlled(case, interceptors=[Control(point, deny=True)])
        with pytest.raises(AdapterBlocked) as caught:
            await adapter.arun(RunRequest({"prompt": "Call ping"}))
        assert caught.value.__cause__ is not None
        assert_denied_before_effect(adapter, point=point, effects=len(case.effects))
        native = native_field(adapter, "_native_error")
        assert isinstance(native, RuntimeError)
        assert isinstance(native.__cause__, AdapterBlocked)
        assert native.__cause__.__cause__ is None
        assert str(native.__cause__) == "Google ADK execution blocked by governance."
        assert "private policy test detail" not in str(native)
        assert adapter.cleanup_error is None

    asyncio.run(scenario())
