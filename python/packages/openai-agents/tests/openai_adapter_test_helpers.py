"""Deterministic model boundary only; native Runner, callbacks, SDK and ACS stay real."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from agent_hooks import ALLOW, AgentContext, Interceptor, Verdict
from agent_hooks_adapters import AgentPolicy
from agent_hooks_adapters.json import JSON, object_value
from agent_hooks_openai_agents import OpenAIAgentsAdapter
from agent_hooks_openai_agents._codec import raw_arguments
from agent_hooks_openai_agents.adapter import NativeScope
from agent_hooks_openai_agents.portable import create_backend
from agents import Agent, FunctionTool, ModelSettings
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import (
    ModelResponse,
    TResponseInputItem,
    TResponseOutputItem,
    TResponseStreamEvent,
)
from agents.models.interface import Model, ModelTracing
from agents.tool import Tool
from agents.tool_context import ToolContext
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types.responses.response_prompt_param import ResponsePromptParam


def text_response(text: str = "native response", number: int = 1) -> ModelResponse:
    message = ResponseOutputMessage(
        id=f"message-{number}",
        type="message",
        role="assistant",
        status="completed",
        phase="final_answer",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    ).model_copy(
        update={
            "provider_data": {
                "model": "unit-model",
                "response_id": f"provider-response-{number}",
                "finish_reason": "stop",
            }
        }
    )
    return ModelResponse(
        output=[message],
        usage=Usage(requests=1, input_tokens=7, output_tokens=3, total_tokens=10),
        response_id=f"response-{number}",
        request_id=f"request-{number}",
        raw_usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    )


def tool_response(
    raw: str = '{"value":"native"}', *, count: int = 1, number: int = 1
) -> ModelResponse:
    output: list[TResponseOutputItem] = [
        ResponseFunctionToolCall(
            id=f"function-item-{number}-{index}",
            call_id=f"call-{number}-{index}",
            name="ping",
            arguments=raw,
            type="function_call",
            status="completed",
        ).model_copy(
            update={
                "provider_data": {
                    "model": "unit-model",
                    "response_id": f"provider-response-{number}",
                    "finish_reason": "tool_calls",
                }
            }
        )
        for index in range(count)
    ]
    return ModelResponse(
        output=output,
        usage=Usage(requests=1, input_tokens=11, output_tokens=5, total_tokens=16),
        response_id=f"response-{number}",
        request_id=f"request-{number}",
        raw_usage={"input_tokens": 11, "output_tokens": 5, "total_tokens": 16},
    )


def _seen() -> list[tuple[str | None, list[TResponseInputItem]]]:
    return []


def _seen_tools() -> list[list[Tool]]:
    return []


@dataclass
class RecordingModel(Model):
    """Model transport fixture; all surrounding native execution is real."""

    responses: list[ModelResponse]
    seen: list[tuple[str | None, list[TResponseInputItem]]] = field(default_factory=_seen)
    seen_tools: list[list[Tool]] = field(default_factory=_seen_tools)
    entered: asyncio.Event | None = None
    release: asyncio.Event | None = None
    cancelled: asyncio.Event | None = None
    suppress_cancel: bool = False
    error: Exception | None = None
    closed: bool = False

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        assert not isinstance(input, str), "Expected actual typed pre-model reconstruction"
        assert output_schema is None and not handoffs and tracing is ModelTracing.DISABLED
        assert previous_response_id is None and conversation_id is None and prompt is None
        assert model_settings.parallel_tool_calls is False
        self.seen.append((system_instructions, list(input)))
        self.seen_tools.append(list(tools))
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    if self.cancelled is not None:
                        self.cancelled.set()
                    if not self.suppress_cancel:
                        raise
        if self.error is not None:
            raise self.error
        return self.responses[min(len(self.seen) - 1, len(self.responses) - 1)]

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        raise RuntimeError("The deterministic transport supports no streaming")

    async def close(self) -> None:
        self.closed = True


class Policy:
    """Real native interceptor with detached captured contexts for assertions."""

    def __init__(self, decide: Callable[[AgentContext], Verdict] | None = None) -> None:
        self.decide = decide
        self.contexts: list[dict[str, JSON]] = []

    def intercept(self, context: AgentContext) -> Verdict:
        self.contexts.append(object_value(context))
        return self.decide(context) if self.decide is not None else ALLOW


@dataclass
class NativeCase:
    agent: Agent[object]
    model: RecordingModel
    tool: FunctionTool
    effects: list[dict[str, JSON]]
    contexts: list[ToolContext[object]]


def native_case(
    *, use_tool: bool = False, responses: list[ModelResponse] | None = None
) -> NativeCase:
    effects: list[dict[str, JSON]] = []
    contexts: list[ToolContext[object]] = []

    async def ping(context: ToolContext[object], raw: str) -> str:
        arguments = raw_arguments(raw)
        effects.append(arguments)
        contexts.append(context)
        value = arguments["value"]
        assert isinstance(value, str)
        return value

    tool = FunctionTool(
        name="ping",
        description="Harmless local echo",
        on_invoke_tool=ping,
        params_json_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        _use_default_failure_error_function=False,
    )
    model = RecordingModel(
        responses
        if responses is not None
        else [tool_response(), text_response(number=2)]
        if use_tool
        else [text_response()]
    )
    agent = Agent[object](
        name="native-agent",
        model=model,
        instructions="Trusted system text",
        tools=[tool] if use_tool else [],
        model_settings=ModelSettings(temperature=0, max_tokens=128, parallel_tool_calls=False),
    )
    return NativeCase(agent, model, tool, effects, contexts)


def controlled(
    case: NativeCase,
    *,
    policies: Sequence[Interceptor] | None = None,
    path: Path | None = None,
    max_records: int = 1000,
    agents: Mapping[str, object] | None = None,
    agent_policies: Sequence[AgentPolicy] = (),
) -> OpenAIAgentsAdapter:
    adapter = create_backend(
        runtime=case.agent,
        interceptors=policies,
        policy_file=path,
        agents=agents,
        agent_policies=agent_policies,
        resolver=None,
        timeout=5,
        emit_timeout=30,
        max_records=max_records,
    )
    assert isinstance(adapter, OpenAIAgentsAdapter)
    return adapter


def scope_of(adapter: OpenAIAgentsAdapter) -> NativeScope:
    value: object = vars(adapter)["_scope"]
    assert isinstance(value, NativeScope)
    return value
