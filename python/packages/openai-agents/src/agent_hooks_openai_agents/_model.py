"""Public executable Model delegate; policy work remains on the shared private lane."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from agents import ModelSettings
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from agents.models.interface import Model, ModelTracing
from agents.tool import Tool
from openai.types.responses.response_prompt_param import ResponsePromptParam

if TYPE_CHECKING:
    from .adapter import NativeScope


class GovernedModel(Model):
    """Delegate real non-streaming model calls with executable pre/post replacement."""

    def __init__(self, inner: Model, scope: NativeScope) -> None:
        self.inner = inner
        self.scope = scope

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
        """Reject resume/stored prompts before invoking the caller's actual Model."""
        self.scope.checked(
            lambda: self.scope.validate_model_call(
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id,
                conversation_id,
                prompt,
            )
        )
        instructions, items, correlation = await self.scope.before_model(system_instructions, input)
        response = await self.inner.get_response(
            instructions,
            items,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
        return await self.scope.after_model(correlation, response)

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
        """Streaming has no admitted profile and never invokes the inner model."""
        self.scope.checked(lambda: self.scope.unsupported("Streaming is unsupported"))
        raise AssertionError("Unreachable streaming boundary")
