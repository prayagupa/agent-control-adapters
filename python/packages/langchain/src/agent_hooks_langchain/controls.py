"""One concrete native AgentMiddleware supplied to the application's create_agent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from agent_hooks_adapters.contracts import validate_agent_id
from agent_hooks_langgraph._native import messages_value, tool_content
from agent_hooks_langgraph._scope import NativeScope
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolCall, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command  # pyright: ignore[reportMissingTypeStubs]

from ._profile import NativeProfile


class LangChainMiddleware(AgentMiddleware[AgentState[object], None, object]):
    """Guard actual native model/tool handlers, not merely observer callbacks."""

    def __init__(self, controls: LangChainControls, label: str) -> None:
        self._controls = controls
        self._label = label

    def _request(self, request: ModelRequest[None]) -> list[AnyMessage]:
        scope = self._controls.scope()
        profile = self._controls.profile()
        if (
            type(request) is not ModelRequest
            or request.model is not profile.model
            or request.response_format is not None
            or request.tool_choice is not None
            or request.model_settings
            or len(request.tools) != len(profile.tools)
            or any(
                tool is not captured.tool
                for tool, captured in zip(request.tools, profile.tools, strict=True)
            )
            or request.system_message is not profile.system
        ):
            raise ValueError("Native model request escaped the captured plain/tool-only profile")
        messages = messages_value(request.messages)
        if request.system_message is not None:
            messages = [request.system_message, *messages]
        scope.checked(lambda: None)
        return messages

    def _override(
        self, request: ModelRequest[None], messages: list[AnyMessage]
    ) -> ModelRequest[None]:
        if request.system_message is None:
            return request.override(messages=messages)
        system = messages[0]
        if not isinstance(system, SystemMessage):
            raise ValueError("Native system-message placement cannot change")
        return request.override(system_message=system, messages=messages[1:])

    def _response(self, response: object) -> AIMessage:
        if (
            type(response) is not ModelResponse
            or response.structured_response is not None
            or len(response.result) != 1
            or type(response.result[0]) is not AIMessage
        ):
            raise ValueError("Exactly one AIMessage in a plain ModelResponse is supported")
        return response.result[0]

    def before_agent(self, state: AgentState[object], runtime: Runtime[None]) -> None:
        """Prove the native before-agent node actually executed in the active scope."""
        scope = self._controls.scope()

        def observe() -> None:
            messages_value(state["messages"])
            scope.before_agent_calls += 1
            if scope.before_agent_calls != 1:
                raise ValueError("Duplicate native before-agent activation")

        scope.checked(observe)

    async def abefore_agent(self, state: AgentState[object], runtime: Runtime[None]) -> None:
        """Observe native async activation without blocking policy work inline."""
        self.before_agent(state, runtime)

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[object]],
    ) -> ModelResponse[object]:
        """Write model transforms into the actual native request and returned response."""
        scope = self._controls.scope()
        profile = self._controls.profile()
        messages = scope.checked(lambda: self._request(request))
        step, replacement = scope.before_model(
            profile.graph, profile.model, profile.tools, messages
        )
        override = scope.checked(lambda: self._override(request, replacement))
        response: object = handler(override)
        message = scope.checked(lambda: self._response(response))
        transformed = scope.after_model(step, message)
        return ModelResponse(result=[transformed])

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[object]]],
    ) -> ModelResponse[object]:
        """Apply model transforms on native ainvoke using the shared async policy lane."""
        scope = self._controls.scope()
        profile = self._controls.profile()
        messages = scope.checked(lambda: self._request(request))
        step, replacement = await scope.abefore_model(
            profile.graph, profile.model, profile.tools, messages
        )
        override = scope.checked(lambda: self._override(request, replacement))
        response: object = await handler(override)
        message = scope.checked(lambda: self._response(response))
        transformed = await scope.aafter_model(step, message)
        return ModelResponse(result=[transformed])

    def _tool_response(self, value: object, request: ToolCallRequest) -> ToolMessage:
        if (
            type(value) is not ToolMessage
            or not isinstance(value.content, str)
            or value.status != "success"
            or value.tool_call_id != request.tool_call["id"]
            or value.name != request.tool_call["name"]
            or value.artifact is not None
        ):
            raise ValueError("Only the actual plain successful ToolMessage is supported")
        return value

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[object]],
    ) -> ToolMessage:
        """Replace executable tool arguments/result; let unrelated native errors escape."""
        scope = self._controls.scope()
        profile = self._controls.profile()
        step, arguments = scope.before_tool(profile.graph, request.tool_call, profile.tools)
        scope.checked(lambda: self._tool_identity(request, step.tool.tool))
        replacement: ToolCall = {
            "id": step.identifier,
            "name": step.tool.name,
            "args": arguments,
            "type": "tool_call",
        }
        response: object = handler(request.override(tool_call=replacement))
        message = scope.checked(lambda: self._tool_response(response, request))
        value = scope.after_tool(step, arguments, message.content)
        return message.model_copy(update={"content": tool_content(value)})

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[object]]],
    ) -> ToolMessage:
        """Deny before the actual async tool body or return its transformed ToolMessage."""
        scope = self._controls.scope()
        profile = self._controls.profile()
        step, arguments = await scope.abefore_tool(profile.graph, request.tool_call, profile.tools)
        scope.checked(lambda: self._tool_identity(request, step.tool.tool))
        replacement: ToolCall = {
            "id": step.identifier,
            "name": step.tool.name,
            "args": arguments,
            "type": "tool_call",
        }
        response: object = await handler(request.override(tool_call=replacement))
        message = scope.checked(lambda: self._tool_response(response, request))
        value = await scope.aafter_tool(step, arguments, message.content)
        return message.model_copy(update={"content": tool_content(value)})

    def _tool_identity(self, request: ToolCallRequest, tool: object) -> None:
        if type(request) is not ToolCallRequest or request.tool is not tool:
            raise ValueError("Executable tool request does not own the registered tool")

    @property
    def label(self) -> str:
        """Captured application-owned graph actor label."""
        return self._label


class LangChainControls:
    """Create exactly one inactive middleware for visible native agent construction."""

    def __init__(self) -> None:
        self._middleware: LangChainMiddleware | None = None
        self._sealed = False
        self._construction_violation = False
        self._scope: NativeScope | None = None
        self._profile: NativeProfile | None = None

    def middleware(self, agent_id: str) -> LangChainMiddleware:
        """Return the concrete middleware to pass as the only create_agent middleware."""
        validate_agent_id(agent_id)
        if self._sealed or self._middleware is not None:
            raise ValueError("Exactly one LangChain actor/middleware is supported")
        self._middleware = LangChainMiddleware(self, agent_id)
        return self._middleware

    def scope(self) -> NativeScope:
        """Reject native guard execution during construction or outside the run."""
        if not self._sealed:
            self._construction_violation = True
            raise RuntimeError("Middleware execution during trusted construction is forbidden")
        if self._scope is None:
            raise RuntimeError("Native middleware has no admitted execution scope")
        return self._scope

    def profile(self) -> NativeProfile:
        """Return the captured actual model/tool/graph membership."""
        if self._profile is None:
            raise RuntimeError("Native middleware has no validated graph profile")
        return self._profile

    @property
    def middleware_instance(self) -> LangChainMiddleware | None:
        """The one concrete middleware returned during native construction."""
        return self._middleware

    @property
    def construction_violation(self) -> bool:
        """Retain a caught construction-time middleware execution attempt."""
        return self._construction_violation

    def seal(self) -> None:
        """Close middleware registration before validating native installation."""
        self._sealed = True

    def attach(self, scope: NativeScope, profile: NativeProfile) -> None:
        """Attach one inactive native scope and its captured immutable profile."""
        if not self._sealed or self._scope is not None:
            raise RuntimeError("Native middleware scope can be attached exactly once")
        self._scope = scope
        self._profile = profile
