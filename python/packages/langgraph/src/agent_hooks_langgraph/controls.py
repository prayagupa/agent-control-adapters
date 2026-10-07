"""Application-facing concrete guarded nodes for native StateGraph construction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from agent_hooks_adapters.codecs import wire_equal
from agent_hooks_adapters.contracts import validate_agent_id
from agent_hooks_adapters.json import object_value
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.language_models.chat_models import (
    _ChatModelBinding,  # pyright: ignore[reportPrivateUsage]
)
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import MessagesState  # pyright: ignore[reportMissingTypeStubs]

from ._native import (
    ModelProof,
    ToolProfile,
    messages_value,
    model_profile,
    tool_content,
    tools_profile,
)
from ._scope import NativeScope


class GuardedModelNode(Runnable[MessagesState, MessagesState]):
    """Invoke an explicit native model only within an activated, proven graph scope."""

    def __init__(
        self,
        controls: LangGraphControls,
        label: str,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
    ) -> None:
        self._controls = controls
        self._label = label
        self._model = model_profile(model)
        self._tools = tools_profile(tools)
        bound: object = (
            model.bind_tools([item.tool for item in self._tools], parallel_tool_calls=False)
            if self._tools
            else model
        )
        if bound is not model:
            if type(bound) not in (RunnableBinding, _ChatModelBinding):
                raise ValueError("Only plain or explicit tool-only model bindings are supported")
            binding = cast("RunnableBinding[LanguageModelInput, BaseMessage]", bound)
            if (
                binding.bound is not model
                or binding.config
                or set(binding.kwargs) - {"tools", "tool_choice", "parallel_tool_calls", "strict"}
                or not wire_equal(
                    binding.kwargs.get("tools"),
                    [convert_to_openai_tool(item.tool) for item in self._tools],
                )
            ):
                raise ValueError("Only plain or explicit tool-only model bindings are supported")
        self._runner = cast("Runnable[LanguageModelInput, BaseMessage]", bound)
        self._model_proof = ModelProof.capture(self._model)
        self._binding_snapshot = (
            object_value(cast("RunnableBinding[LanguageModelInput, BaseMessage]", bound).kwargs)
            if bound is not model
            else None
        )

    def revalidate(self) -> None:
        """Prove guarded method inventory and immutable actual tool-only binding options."""
        if (
            set(vars(self))
            != {
                "_controls",
                "_label",
                "_model",
                "_tools",
                "_runner",
                "_model_proof",
                "_binding_snapshot",
            }
            or self._model is not self._model_proof.model
        ):
            raise ValueError("Guarded native model configuration changed")
        self._model_proof.revalidate()
        if self._binding_snapshot is None:
            if self._runner is not self._model:
                raise ValueError("Guarded native model binding changed")
        else:
            binding = self._runner
            if type(binding) not in (RunnableBinding, _ChatModelBinding):
                raise ValueError("Guarded native model binding changed")
            binding = cast("RunnableBinding[LanguageModelInput, BaseMessage]", binding)
            if (
                binding.bound is not self._model
                or binding.config
                or binding.config_factories
                or binding.custom_input_type is not None
                or binding.custom_output_type is not None
                or not wire_equal(binding.kwargs, self._binding_snapshot)
            ):
                raise ValueError("Guarded native model binding changed")

    @property
    def label(self) -> str:
        """Captured application-owned node label."""
        return self._label

    @property
    def model(self) -> BaseChatModel:
        """Actual explicit model retained by this guarded node."""
        return self._model

    @property
    def tools(self) -> tuple[ToolProfile, ...]:
        """Exact registered native tool profiles."""
        return self._tools

    def _input(self, value: MessagesState, kwargs: dict[str, object]) -> MessagesState:
        if set(value) != {"messages"} or kwargs:
            raise ValueError("Guarded nodes accept only ordinary native messages state")
        return {"messages": messages_value(value["messages"])}

    def invoke(
        self, input: MessagesState, config: RunnableConfig | None = None, **kwargs: object
    ) -> MessagesState:
        """Execute native synchronous model transport with actual pre/post write-back."""
        scope = self._controls.scope()
        state = scope.checked(lambda: self._input(input, kwargs))
        step, messages = scope.before_model(self, self._model, self._tools, state["messages"])
        response: object = self._runner.invoke(messages, config)
        return {"messages": [scope.after_model(step, response)]}

    async def ainvoke(
        self, input: MessagesState, config: RunnableConfig | None = None, **kwargs: object
    ) -> MessagesState:
        """Await explicit native async transport on the caller's loop."""
        scope = self._controls.scope()
        state = scope.checked(lambda: self._input(input, kwargs))
        step, messages = await scope.abefore_model(
            self, self._model, self._tools, state["messages"]
        )
        response: object = await self._runner.ainvoke(messages, config)
        return {"messages": [await scope.aafter_model(step, response)]}


class GuardedToolNode(Runnable[MessagesState, MessagesState]):
    """Invoke one registered native tool after checking actual destination ownership."""

    def __init__(self, controls: LangGraphControls, label: str, tools: Sequence[BaseTool]) -> None:
        self._controls = controls
        self._label = label
        self._tools = tools_profile(tools)
        if not self._tools:
            raise ValueError("A guarded tool node requires at least one native tool")

    @property
    def label(self) -> str:
        """Captured application-owned tool-node label."""
        return self._label

    @property
    def tools(self) -> tuple[ToolProfile, ...]:
        """Exact registered native tool profiles."""
        return self._tools

    def revalidate(self) -> None:
        """Reject replaced guard methods and changed tool configuration."""
        if set(vars(self)) != {"_controls", "_label", "_tools"}:
            raise ValueError("Guarded native tool configuration changed")
        for tool in self._tools:
            tool.revalidate()

    def _call(self, value: MessagesState, kwargs: dict[str, object]) -> object:
        if set(value) != {"messages"} or kwargs:
            raise ValueError("Guarded tool nodes accept only native messages state")
        messages = messages_value(value["messages"])
        message = messages[-1]
        if not isinstance(message, AIMessage) or len(message.tool_calls) != 1:
            raise ValueError("A tool node requires exactly one actual model tool call")
        return message.tool_calls[0]

    def invoke(
        self, input: MessagesState, config: RunnableConfig | None = None, **kwargs: object
    ) -> MessagesState:
        """Execute one actual sync tool with proven source accounting and replacements."""
        scope = self._controls.scope()
        call = scope.checked(lambda: self._call(input, kwargs))
        step, arguments = scope.before_tool(self, call, self._tools)
        scope.checked(scope.revalidate)
        response: object = step.tool.tool.invoke(arguments, config)
        value = scope.after_tool(step, arguments, response)
        return {
            "messages": [
                ToolMessage(
                    content=tool_content(value),
                    tool_call_id=step.identifier,
                    name=step.tool.name,
                )
            ]
        }

    async def ainvoke(
        self, input: MessagesState, config: RunnableConfig | None = None, **kwargs: object
    ) -> MessagesState:
        """Await one explicit native async tool, never the default executor fallback."""
        scope = self._controls.scope()
        call = scope.checked(lambda: self._call(input, kwargs))
        step, arguments = await scope.abefore_tool(self, call, self._tools)
        scope.checked(scope.revalidate)
        response: object = await step.tool.tool.ainvoke(arguments, config)
        value = await scope.aafter_tool(step, arguments, response)
        return {
            "messages": [
                ToolMessage(
                    content=tool_content(value),
                    tool_call_id=step.identifier,
                    name=step.tool.name,
                )
            ]
        }


class LangGraphControls:
    """Create inactive guarded nodes; the application owns native graph construction."""

    def __init__(self) -> None:
        self._nodes: list[GuardedModelNode | GuardedToolNode] = []
        self._sealed = False
        self._construction_violation = False
        self._scope: NativeScope | None = None

    def _label(self, agent_id: str) -> None:
        validate_agent_id(agent_id)
        if (
            self._sealed
            or len(self._nodes) >= 16
            or any(node.label == agent_id for node in self._nodes)
        ):
            raise ValueError("Guarded node labels must be unique and bounded")

    def model_node(
        self, agent_id: str, model: BaseChatModel, tools: Sequence[BaseTool] = ()
    ) -> GuardedModelNode:
        """Return one concrete Runnable owning all native model effects for this actor."""
        self._label(agent_id)
        node = GuardedModelNode(self, agent_id, model, tools)
        self._nodes.append(node)
        return node

    def tool_node(self, agent_id: str, tools: Sequence[BaseTool]) -> GuardedToolNode:
        """Return one concrete Runnable owning one correlated native tool step."""
        self._label(agent_id)
        node = GuardedToolNode(self, agent_id, tools)
        self._nodes.append(node)
        return node

    def scope(self) -> NativeScope:
        """Reject guard execution during construction or outside the admitted run."""
        if not self._sealed:
            self._construction_violation = True
            raise RuntimeError("Guard execution during trusted graph construction is forbidden")
        if self._scope is None:
            raise RuntimeError("Guarded native graph has no admitted execution scope")
        return self._scope

    @property
    def nodes(self) -> tuple[GuardedModelNode | GuardedToolNode, ...]:
        """Captured distinct actual guarded nodes, never a mutable registry."""
        return tuple(self._nodes)

    @property
    def construction_violation(self) -> bool:
        """Retain a caught construction-time guard execution attempt."""
        return self._construction_violation

    def seal(self) -> None:
        """Close builder registration before inspecting the returned native graph."""
        self._sealed = True

    def attach(self, scope: NativeScope) -> None:
        """Attach one inactive validated scope without executing any guard."""
        if not self._sealed or self._scope is not None:
            raise RuntimeError("Native graph scope can be attached exactly once after construction")
        self._scope = scope
