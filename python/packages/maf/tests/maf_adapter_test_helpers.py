"""Deterministic model I/O, actual native MAF runner and actual ACS policy engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import httpx2
import pytest
from agent_framework import Agent, ChatOptions, ChatResponse, Content, FunctionTool, Message, tool
from agent_framework.openai import OpenAIChatClient
from agent_hooks_adapters import AsyncAdapter
from agent_hooks_maf.portable import create_backend
from openai import AsyncOpenAI


def _seen_messages() -> list[list[Message]]:
    return []


def _seen_options() -> list[dict[str, object]]:
    return []


@dataclass
class ModelBoundary:
    use_tool: bool
    repeat_tools: bool = False
    tool_calls_per_response: int = 1
    calls: int = 0
    seen: list[list[Message]] = field(default_factory=_seen_messages)
    seen_options: list[dict[str, object]] = field(default_factory=_seen_options)

    async def respond(
        self,
        *,
        messages: Sequence[Message],
        options: Mapping[str, object],
        stream: bool = False,
        **kwargs: object,
    ) -> ChatResponse[None]:
        if stream:
            raise ValueError("Test transport supports non-streaming calls only")
        self.calls += 1
        self.seen.append(list(messages))
        self.seen_options.append(dict(options))
        contents = (
            [
                Content.from_function_call(
                    f"call-{self.calls}-{index}", "ping", arguments={"value": "native"}
                )
                for index in range(self.tool_calls_per_response)
            ]
            if self.use_tool and (self.calls == 1 or self.repeat_tools)
            else [Content.from_text("native response")]
        )
        return ChatResponse(
            messages=[Message(role="assistant", contents=contents)],
            response_id=f"response-{self.calls}",
        )


@dataclass
class NativeCase:
    agent: Agent[ChatOptions[None]]
    client: OpenAIChatClient
    transport: AsyncOpenAI
    model: ModelBoundary
    effects: list[str]


def native_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_tool: bool = False,
    serial: bool = True,
    repeat_tools: bool = False,
    tool_calls_per_response: int = 1,
) -> NativeCase:
    effects: list[str] = []

    @tool
    async def ping(value: str) -> str:
        """Return a harmless input after recording the actual native tool effect."""
        effects.append(value)
        return value

    def model_listing(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/models", "Unexpected deterministic transport request"
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "deterministic-model-boundary", "object": "model"}],
            },
        )

    http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(model_listing),
        trust_env=False,
        follow_redirects=False,
    )
    transport = AsyncOpenAI(
        api_key="local-test",
        base_url="http://127.0.0.1:1/v1",
        http_client=http,
        max_retries=0,
        timeout=1,
    )
    client = OpenAIChatClient(
        model="deterministic-model-boundary",
        async_client=transport,
        function_invocation_configuration={"allow_concurrent_invocation": not serial},
    )
    model = ModelBoundary(use_tool, repeat_tools, tool_calls_per_response)
    monkeypatch.setattr(client, "_inner_get_response", model.respond)
    agent = Agent[ChatOptions[None]](
        client, instructions="Trusted system text", tools=[ping] if use_tool else []
    )
    return NativeCase(agent=agent, client=client, transport=transport, model=model, effects=effects)


def controlled(case: NativeCase, path: Path, *, max_records: int = 1000) -> AsyncAdapter:
    return create_backend(
        runtime=case.agent,
        interceptors=None,
        policy_file=path,
        agents=None,
        agent_policies=(),
        resolver=None,
        timeout=5,
        emit_timeout=30,
        max_records=max_records,
    )


def native_tool_pair(case: NativeCase) -> tuple[FunctionTool, FunctionTool]:
    """Attach a second native tool with the same schema but a different effect."""
    raw_tools: object = case.agent.default_options["tools"]
    assert isinstance(raw_tools, list)
    tools = cast("list[object]", raw_tools)
    assert len(tools) == 1
    first = tools[0]
    assert isinstance(first, FunctionTool)

    @tool
    async def other(value: str) -> str:
        """Record the second tool's distinct effect."""
        case.effects.append(f"other:{value}")
        return value

    assert isinstance(other, FunctionTool)
    tools.append(other)
    return first, other
