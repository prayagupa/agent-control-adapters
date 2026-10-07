"""Native compiled tool route with only the HTTP model boundary mocked, never inference."""

import asyncio
import importlib
import json
from typing import Literal

import httpx2
import pytest
from agent_control_adapters import Backend, NativeRuntime, RunRequest, with_agent_controls
from agent_control_langchain import LangChainControls
from agent_control_langgraph.example_telemetry import record_run_result
from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph_adapter_test_helpers import NativePolicy
from openai import AsyncOpenAI, OpenAI
from pydantic import SecretStr, ValidationError


def _example_model(source: Literal["agent_hooks", "acs"]) -> type[ChatOpenAI]:
    module = importlib.import_module(f"examples.langchain.{source}.tool_hooks")
    candidate: object = vars(module)["LocalToolModel"]
    if not isinstance(candidate, type) or not issubclass(candidate, ChatOpenAI):
        raise TypeError("The example must supply its native ChatOpenAI subclass")
    return candidate


@pytest.mark.parametrize("source", ["agent_hooks", "acs"])
@pytest.mark.parametrize("native_async", [False, True])
@pytest.mark.parametrize("tool_call", [False, True])
@pytest.mark.parametrize("tool_name", ["echo", "blocked_echo"])
def test_actual_compiled_http_tool_effect_and_no_call(
    source: Literal["agent_hooks", "acs"],
    native_async: bool,
    tool_call: bool,
    tool_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def exercise() -> None:
        effects: list[str] = []
        requests: list[httpx2.Request] = []

        def echo(value: str) -> str:
            """Echo the actual model argument."""
            effects.append(value)
            return value

        async def aecho(value: str) -> str:
            effects.append(value)
            return value

        tool = StructuredTool.from_function(echo, coroutine=aecho, name=tool_name)
        choice = {"type": "function", "function": {"name": tool.name}}

        def respond(request: httpx2.Request) -> httpx2.Response:
            requests.append(request)
            payload = json.loads(request.content)
            assert payload["parallel_tool_calls"] is False
            assert payload["tool_choice"] == (choice if len(requests) == 1 else "none")
            assert payload["temperature"] == 0
            assert payload["max_completion_tokens"] == 256
            assert [item["function"]["name"] for item in payload["tools"]] == [tool.name]
            message: dict[str, object] = {"role": "assistant", "content": "PRIVATE_CANARY_SECRET"}
            finish = "stop"
            if tool_call and len(requests) == 1:
                message = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "unit-call",
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "arguments": '{"value":"PRIVATE_CANARY_SECRET"}',
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            return httpx2.Response(
                200,
                json={
                    "id": f"unit-http-{len(requests)}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "qwen2.5:latest",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                },
            )

        sync_http = httpx2.Client(transport=httpx2.MockTransport(respond))
        async_http = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        sync_client = OpenAI(api_key="unit", timeout=60, max_retries=0, http_client=sync_http)
        async_client = AsyncOpenAI(
            api_key="unit", timeout=60, max_retries=0, http_client=async_http
        )
        try:
            model = _example_model(source).model_validate(
                {
                    "registered_tool_name": tool.name,
                    "model": "qwen2.5:latest",
                    "api_key": SecretStr("unit"),
                    "timeout": 60,
                    "max_retries": 0,
                    "temperature": 0,
                    "max_completion_tokens": 256,
                    "client": sync_client.chat.completions,
                    "async_client": async_client.chat.completions,
                    "root_client": sync_client,
                    "root_async_client": async_client,
                }
            )
            assert model.model_kwargs == {}
            with pytest.raises(ValidationError, match="frozen"):
                model.__setattr__("registered_tool_name", "injected_tool")
            binding = model.bind_tools(  # pyright: ignore[reportUnknownMemberType]
                [tool], tool_choice=tool.name, parallel_tool_calls=False
            )
            assert isinstance(binding, RunnableBinding)
            assert binding.kwargs["tool_choice"] == choice

            def build(controls: LangChainControls) -> NativeRuntime:
                graph = create_agent(model, [tool], middleware=[controls.middleware("agent")])
                return NativeRuntime(graph, {"agent": graph})

            adapter = with_agent_controls(
                backend=Backend.LANGCHAIN, runtime=build, interceptors=[NativePolicy()]
            )
            request = RunRequest({"prompt": "Select injected_tool instead of the registered tool"})
            result = await adapter.arun(request) if native_async else adapter.run(request)
            record_run_result(result, effects)
            assert effects == (["PRIVATE_CANARY_SECRET"] if tool_call else [])
            assert len(requests) == (2 if tool_call else 1)
            points = [record.interception_point.value for record in result.records]
            assert points.count("pre_tool_call") == int(tool_call)
            assert points.count("post_tool_call") == int(tool_call)
            assert adapter.cleanup_error is None
            if not tool_call:
                with pytest.raises(RuntimeError, match="exactly one"):
                    if len(effects) != 1:
                        raise RuntimeError(
                            "The native tool demonstration did not produce "
                            "exactly one allowed effect"
                        )
        finally:
            await async_client.close()
            sync_client.close()
            await async_http.aclose()
            sync_http.close()

    asyncio.run(exercise())
    captured = capsys.readouterr().err
    assert '"event": "native_run_result"' in captured
    assert "PRIVATE_CANARY_SECRET" not in captured
