"""Actual ADK plugin callbacks, backed by the shared SDK policy lane."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar, cast
from uuid import uuid4

from agent_hooks_adapters import AdapterBlocked
from agent_hooks_adapters.codecs import (
    input_text,
    model_text,
    output_text,
    text_messages,
    text_value,
    unchanged,
    wire_equal,
)
from agent_hooks_adapters.json import JSON, normalize, object_value
from agent_hooks_adapters.runtime import PolicySession
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from ._codecs import plain, request_projection, response_projection
from ._profile import Profile, native_field, revalidate

BLOCK_MESSAGE = "Google ADK execution blocked by governance."
_T = TypeVar("_T")


class ControlsPlugin(BasePlugin):
    """One run-owned plugin; native callbacks never evaluate policy on the caller loop."""

    def __init__(
        self,
        native: Profile,
        session: PolicySession,
        *,
        user_id: str,
        session_id: str,
        abort_signal: asyncio.Event,
    ) -> None:
        super().__init__(name="agent_hooks_google_adk_controls")
        self.native = native
        self.session = session
        self.user_id = user_id
        self.session_id = session_id
        self._abort_signal = abort_signal
        self.output: str | None = None
        self._final_text: str | None = None
        self._started: list[int] = []
        self._shutdown: set[int] = set()
        self._pending_models: dict[int, tuple[str, str]] = {}
        self._native_events = 0
        self._root_clone: LlmAgent | None = None

    async def _guard(self, operation: Callable[[], Awaitable[_T]]) -> _T:
        try:
            if self._abort_signal.is_set():
                raise asyncio.CancelledError
            revalidate(self.native)
            return await operation()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._abort_signal.is_set():
                raise asyncio.CancelledError from None
            if not self.session.policy_failed:
                self.session.fail(error, policy=True)
            raise AdapterBlocked(BLOCK_MESSAGE) from None

    def _scope(self, context: InvocationContext) -> tuple[int, BaseAgent]:
        if (
            context.user_id != self.user_id
            or context.session.id != self.session_id
            or context.plugin_manager.plugins != [self]
            or not isinstance(context.agent, BaseAgent)
        ):
            raise ValueError("Unknown native invocation ownership")
        if context.agent is self._root_clone:
            return self.session.binding(self.native.root), self.native.root
        return self.session.binding(context.agent), context.agent

    def _register_native_clone(self, agent: BaseAgent) -> None:
        root = self.native.root
        if any(agent is actor for actor in self.native.actors) or agent is self._root_clone:
            return
        if (
            self._root_clone is not None
            or type(root) is not LlmAgent
            or type(agent) is not LlmAgent
            or agent.model is not root.model
            or agent.parent_agent is not root.parent_agent
            or agent.sub_agents
        ):
            raise ValueError("Foreign native actor or repeated root clone")
        original_tools = tuple(tool.native for tool in self.native.tools if tool.actor is root)
        raw_tools = native_field(agent, "tools")
        if not isinstance(raw_tools, list):
            raise ValueError("Native clone tools must retain their native list shape")
        clone_tools = cast("list[object]", raw_tools)
        if len(original_tools) != len(clone_tools) or any(
            original is not clone
            for original, clone in zip(original_tools, clone_tools, strict=True)
        ):
            raise ValueError("Native clone changed executable tool identities")
        excluded = {"model", "tools", "parent_agent", "sub_agents"}
        expected = root.model_dump(mode="json", exclude=excluded)
        expected.update(rerun_on_resume=True, wait_for_output=True)
        if not wire_equal(expected, agent.model_dump(mode="json", exclude=excluded)):
            raise ValueError("Native clone differs from the pinned build_node transformation")
        self._root_clone = agent

    def _actor(self, context: CallbackContext) -> tuple[int, BaseAgent]:
        invocation: object = getattr(context, "_invocation_context", None)
        if not isinstance(invocation, InvocationContext):
            raise ValueError("Unknown native callback context")
        return self._scope(invocation)

    async def before_run_callback(self, *, invocation_context: InvocationContext) -> None:
        async def before() -> None:
            self._scope(invocation_context)
            if self._started:
                return
            for actor in self.native.actors:
                binding = self.session.binding(actor)
                self._started.append(binding)
                tools = [tool.name for tool in self.native.tools if tool.actor is actor]
                original = {"tools_registered": tools}
                await self.session.aemit(
                    binding,
                    lambda builder, tools=tools: builder.agent_startup(tools_registered=tools),
                    lambda target, original=original: unchanged(target, original),
                )

        await self._guard(before)

    async def on_user_message_callback(
        self, *, invocation_context: InvocationContext, user_message: types.Content
    ) -> types.Content:
        async def receive() -> types.Content:
            await self.before_run_callback(invocation_context=invocation_context)
            binding, _ = self._scope(invocation_context)
            if user_message.role != "user":
                raise ValueError("Native prompt must have the host-owned user role")
            text = plain(user_message)
            replacement = await self.session.aemit(
                binding,
                lambda builder: builder.input(content=text),
                input_text,
            )
            return types.Content(role="user", parts=[types.Part(text=text_value(replacement))])

        return await self._guard(receive)

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        async def before() -> None:
            invocation: object = getattr(callback_context, "_invocation_context", None)
            if not isinstance(invocation, InvocationContext) or invocation.agent is not agent:
                raise ValueError("Foreign native agent callback")
            if invocation.user_id != self.user_id or invocation.session.id != self.session_id:
                raise ValueError("Foreign native agent invocation")
            self._register_native_clone(agent)
            self._actor(callback_context)

        await self._guard(before)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        async def before() -> None:
            binding, actor = self._actor(callback_context)
            if not isinstance(actor, LlmAgent) or binding in self._pending_models:
                raise ValueError("Unknown or overlapping native model request")
            projection = request_projection(llm_request)
            model_id = llm_request.model
            if not isinstance(model_id, str) or not model_id:
                raise ValueError("Native model request requires an explicit model identifier")
            for name, tool in llm_request.tools_dict.items():
                owned = self.native.tool(actor, name)
                if type(tool) is not FunctionTool or tool.func is not owned.function:
                    raise ValueError("Unguarded native model tool")
            correlation = str(uuid4())
            self.session.budget.before_model(binding, correlation)
            self._pending_models[binding] = (correlation, model_id)
            replacement = await self.session.aemit(
                binding,
                lambda builder: builder.pre_model_call(
                    model_id=model_id,
                    messages=projection.messages,
                    request_id=correlation,
                ),
                lambda target: text_messages(target, projection.messages),
            )
            projection.apply(llm_request, replacement)

        await self._guard(before)

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse:
        async def after() -> LlmResponse:
            binding, actor = self._actor(callback_context)
            pending = self._pending_models.get(binding)
            if pending is None:
                raise ValueError("Unexpected native model completion")
            correlation, model_id = pending
            projection = response_projection(llm_response)
            calls = projection.target["tool_calls"]
            if not isinstance(calls, list):
                raise ValueError("Unknown native function calls")
            identifiers: list[str] = []
            projected_calls: list[dict[str, JSON]] = []
            for value in calls:
                call = object_value(value)
                name, identifier = call["name"], call["id"]
                if not isinstance(name, str) or not isinstance(identifier, str):
                    raise ValueError("Invalid native tool routing")
                self.native.tool(actor, name).arguments(call["args"])
                identifiers.append(identifier)
                projected_calls.append(call)
            self.session.budget.after_model(binding, correlation, identifiers)
            del self._pending_models[binding]
            replacement = await self.session.aemit(
                binding,
                lambda builder: builder.post_model_call(
                    model_id=model_id,
                    content=projection.target["content"],
                    tool_calls=projected_calls,
                    finish_reason=str(projection.target["finish_reason"]),
                    request_id=correlation,
                ),
                lambda target: model_text(target, projection.target),
            )
            return projection.apply(replacement)

        return await self._guard(after)

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, object], tool_context: ToolContext
    ) -> None:
        async def before() -> None:
            binding, actor = self._actor(tool_context)
            owned = self.native.tool(actor, tool.name)
            if type(tool) is not FunctionTool or tool.func is not owned.function:
                raise ValueError("Foreign native executable tool")
            arguments = owned.arguments(tool_args)
            identifier = tool_context.function_call_id
            if not isinstance(identifier, str):
                raise ValueError("Missing native tool correlation")
            self.session.budget.before_tool(binding, identifier)
            replacement = await self.session.aemit(
                binding,
                lambda builder: builder.pre_tool_call(
                    call_id=identifier,
                    name=tool.name,
                    args=arguments,
                ),
                owned.arguments,
            )
            tool_args.clear()
            tool_args.update(owned.arguments(replacement))

        await self._guard(before)

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, object],
        tool_context: ToolContext,
        result: dict[str, object],
    ) -> dict[str, object]:
        async def after() -> dict[str, object]:
            binding, actor = self._actor(tool_context)
            owned = self.native.tool(actor, tool.name)
            if type(tool) is not FunctionTool or tool.func is not owned.function:
                raise ValueError("Foreign native completed tool")
            arguments = owned.arguments(tool_args)
            identifier = tool_context.function_call_id
            if not isinstance(identifier, str):
                raise ValueError("Missing native tool completion correlation")
            original = normalize(result)
            self.session.budget.after_tool(binding, identifier)
            replacement = normalize(
                await self.session.aemit(
                    binding,
                    lambda builder: builder.post_tool_call(
                        call_id=identifier,
                        name=tool.name,
                        args=arguments,
                        value=original,
                    ),
                )
            )
            normalized = replacement if isinstance(replacement, dict) else {"result": replacement}
            return {key: value for key, value in normalized.items()}

        return await self._guard(after)

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> None:
        async def receive() -> None:
            self._scope(invocation_context)
            self._native_events += 1
            if self._native_events > 256:
                raise ValueError("Native event limit exceeded")
            if event.is_final_response() and event.content is not None:
                self._final_text = plain(event.content)

        await self._guard(receive)

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        async def after() -> None:
            binding, _ = self._scope(invocation_context)
            if self._final_text is None:
                raise ValueError("Native run did not produce a final plain text response")
            replacement = await self.session.aemit(
                binding,
                lambda builder: builder.output(content=self._final_text),
                output_text,
            )
            self.output = text_value(replacement)

        await self._guard(after)
        await self.shutdown("completed")

    async def shutdown(self, reason: str) -> None:
        """Emit real audit-only shutdown once, including native failure/cancellation paths."""
        for binding in self._started:
            if binding not in self._shutdown:
                self._shutdown.add(binding)
                await self.session.aemit(
                    binding,
                    lambda builder: builder.agent_shutdown(reason=reason),
                    audit=True,
                )
        self._pending_models.clear()
