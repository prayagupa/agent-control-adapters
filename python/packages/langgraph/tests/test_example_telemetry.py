"""CPU-only example configuration and actual SDK HTTP observability regressions."""

import asyncio
import io
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import httpx2
import pytest
from agent_adapter_conformance import assert_graph_example_transport_budget
from agent_control_langgraph.example_telemetry import ExampleTelemetry
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI
from pydantic import SecretStr

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("source", ["agent_hooks", "acs"])
@pytest.mark.parametrize("script", ["llm_hooks.py", "tool_hooks.py"])
def test_all_examples_explicit_transport_budget(source: str, script: str) -> None:
    assert_graph_example_transport_budget(ROOT / "examples" / "langgraph" / source / script)


def test_actual_native_sdk_effective_timeout_and_response_preserved() -> None:
    async def exercise() -> None:
        observed: list[object] = []
        output = io.StringIO()
        telemetry = ExampleTelemetry(output)

        def respond(request: httpx2.Request) -> httpx2.Response:
            observed.append(request.extensions["timeout"])
            return httpx2.Response(
                200,
                json={
                    "id": "unit-http-response",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "qwen2.5:latest",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "unit boundary"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        sync_http = httpx2.Client(transport=httpx2.MockTransport(respond))
        async_http = httpx2.AsyncClient(
            transport=telemetry.transport(httpx2.MockTransport(respond)),
            event_hooks={"request": [telemetry.request], "response": [telemetry.response]},
        )
        sync_client = OpenAI(api_key="unit", timeout=60, max_retries=0, http_client=sync_http)
        async_client = AsyncOpenAI(
            api_key="unit", timeout=60, max_retries=0, http_client=async_http
        )
        try:
            model = ChatOpenAI(
                model="qwen2.5:latest",
                api_key=SecretStr("unit"),
                timeout=60,
                max_retries=0,
                temperature=0,
                max_completion_tokens=256,
                client=sync_client.chat.completions,
                async_client=async_client.chat.completions,
                root_client=sync_client,
                root_async_client=async_client,
            )
            message = await model.ainvoke("unit HTTP boundary")
            assert message.content == "unit boundary"
            assert observed == [{"connect": 60, "read": 60, "write": 60, "pool": 60}]
            events = [json.loads(line) for line in output.getvalue().splitlines()]
            assert [event["event"] for event in events] == ["request_start", "response_headers"]
            assert "raw_response" not in events[1] and "model_request" not in events[0]
            assert events[1]["elapsed_seconds"] >= 0
        finally:
            await async_client.close()
            sync_client.close()
            await async_http.aclose()
            sync_http.close()

    asyncio.run(exercise())


def test_default_canaries_and_response_stream_are_not_read_or_logged() -> None:
    async def exercise() -> None:
        canary = "PRIVATE_CANARY_SECRET"
        output = io.StringIO()
        telemetry = ExampleTelemetry(output)
        request = httpx2.Request(
            "POST",
            f"http://user:{canary}@local.invalid/?key={canary}",
            json={"messages": [{"content": canary}], "arguments": canary},
        )
        reads: list[int] = []

        class Stream(httpx2.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                reads.append(1)
                yield canary.encode() * 100000

        stream = Stream()
        response = httpx2.Response(400, request=request, stream=stream)
        await telemetry.request(request)
        await telemetry.response(response)
        assert response.stream is stream and reads == []
        assert canary not in output.getvalue()
        summary = telemetry.summary(
            {
                "status": "allowed",
                "output": canary,
                "tool_effect_values": [canary],
                "tool_effects": 1,
                "provider_error": canary,
                "sdk_cause": {"type": "OtherError", "message": canary},
            }
        )
        assert canary not in json.dumps(summary)
        assert summary["tool_effects"] == 1
        await response.aclose()

    asyncio.run(exercise())


def test_explicit_synthetic_request_capture_is_validated_and_prebounded() -> None:
    async def exercise() -> None:
        output = io.StringIO()
        with pytest.raises(ValueError, match="local-toy-v1"):
            ExampleTelemetry(output, synthetic_data="production")
        telemetry = ExampleTelemetry(output, synthetic_data="local-toy-v1")
        with pytest.raises(ValueError, match="credential-free"):
            await telemetry.request(httpx2.Request("POST", "http://user:secret@localhost:11434"))
        request = httpx2.Request(
            "POST", "http://localhost:11434/v1/chat/completions", content=b"x" * 4097
        )
        await telemetry.request(request)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        assert events[0]["event"] == "synthetic_diagnostics_warning"
        assert events[-1]["raw_request_omitted"] is True
        assert "model_request" not in events[-1]
        assert "x" * 100 not in output.getvalue()

    asyncio.run(exercise())


def test_synthetic_capture_aggregate_log_limit_is_explicit() -> None:
    async def exercise() -> None:
        output = io.StringIO()
        telemetry = ExampleTelemetry(output, synthetic_data="local-toy-v1")
        for _ in range(10):
            request = httpx2.Request(
                "POST",
                "http://localhost:11434/v1/chat/completions",
                json={"messages": [{"content": "\u6f22" * 1200}]},
            )
            await telemetry.request(request)
            await telemetry.response(httpx2.Response(200, request=request))
        assert len(output.getvalue().encode()) <= 65536
        assert '"event":"telemetry_limit"' in output.getvalue()
        assert "raw_response" not in output.getvalue()

    asyncio.run(exercise())


def test_transport_exception_observed_and_original_preserved() -> None:
    async def exercise() -> None:
        output = io.StringIO()
        telemetry = ExampleTelemetry(output)
        original = httpx2.ReadTimeout("unit HTTP boundary")

        def fail(request: httpx2.Request) -> httpx2.Response:
            raise original

        async with httpx2.AsyncClient(
            transport=telemetry.transport(httpx2.MockTransport(fail)),
            event_hooks={"request": [telemetry.request], "response": [telemetry.response]},
        ) as client:
            with pytest.raises(httpx2.ReadTimeout) as error:
                await client.post("http://local.invalid", json={"model": "unit"})
            assert error.value is original
        events: list[dict[str, object]] = [
            cast("dict[str, object]", json.loads(line)) for line in output.getvalue().splitlines()
        ]
        assert [event["event"] for event in events] == ["request_start", "request_exception_end"]
        assert events[1]["exception_type"] == "ReadTimeout"

    asyncio.run(exercise())
