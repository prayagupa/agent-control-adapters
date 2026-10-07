"""Actual pinned decorator/parser consumption, not a raw-callback-only proxy."""

from __future__ import annotations

import asyncio

import pytest
from agent_control_adapters import AdapterBlocked, RunRequest
from agent_hooks import ALLOW, Decision, Transform, Verdict
from agents import function_tool
from openai_adapter_test_helpers import Policy, controlled, native_case, tool_response


def test_decorated_native_parser_consumes_changed_arguments() -> None:
    effects: list[str] = []

    @function_tool(failure_error_function=None)
    async def ping(value: str) -> str:
        """Echo an authorized argument."""
        effects.append(value)
        return value

    async def scenario() -> None:
        case = native_case(use_tool=True)
        case.agent.tools = [ping]
        policy = Policy(
            lambda context: (
                Verdict(
                    Decision.TRANSFORM, transform=Transform("$target.value", "actual replacement")
                )
                if context["interception_point"] == "pre_tool_call"
                else ALLOW
            )
        )
        result = await controlled(case, policies=[policy]).arun(RunRequest({"prompt": "safe"}))
        assert result.output == "native response" and effects == ["actual replacement"]

    asyncio.run(scenario())


def test_decorated_native_parser_cannot_coerce_original_integer() -> None:
    effects: list[int] = []

    @function_tool(failure_error_function=None)
    async def ping(value: int) -> str:
        """Echo an authorized integer."""
        effects.append(value)
        return str(value)

    async def scenario() -> None:
        case = native_case(use_tool=True, responses=[tool_response('{"value":1.0}')])
        case.agent.tools = [ping]
        with pytest.raises(AdapterBlocked):
            await controlled(case, policies=[Policy()]).arun(RunRequest({"prompt": "safe"}))
        assert effects == []

    asyncio.run(scenario())


def test_decorated_schema_cannot_contradict_actual_parser_types() -> None:
    @function_tool(failure_error_function=None)
    async def ping(value: int) -> str:
        """Echo an integer."""
        return str(value)

    case = native_case(use_tool=True)
    ping.params_json_schema["properties"] = {"value": {"type": "string"}}
    case.agent.tools = [ping]
    with pytest.raises(ValueError, match="contradict"):
        controlled(case, policies=[Policy()])
