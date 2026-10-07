"""Read-only, exact pinned create_agent closure and middleware installation proof."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from types import FunctionType
from typing import TYPE_CHECKING, Protocol, cast

from agent_hooks_adapters.codecs import wire_equal
from agent_hooks_adapters.json import JSON, object_value
from agent_hooks_langgraph._inspection import (
    Graph,
    GraphProof,
    attribute,
    inspect_graph,
    native_builder,
    nested_codes,
)
from agent_hooks_langgraph._native import ModelProof, ToolProfile, model_profile, tools_profile
from langchain.agents import factory as native_factory
from langchain.agents.middleware import _trace_policy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import tool_node as native_tool_node
from langgraph.prebuilt.tool_node import ToolNode

if TYPE_CHECKING:
    from .controls import LangChainMiddleware


class InjectedView(Protocol):
    """Read-only pinned injection inventory; no untrusted injection is supported."""

    state: Mapping[str, object]
    store: object
    runtime: object
    all_injected_keys: set[str]


create_agent = cast("FunctionType", attribute(native_factory, "create_agent"))
default_tool_errors: object = attribute(native_tool_node, "_default_handle_tool_errors")


def _native_function(value: object, name: str) -> FunctionType:
    owners = (
        create_agent,
        cast("FunctionType", attribute(native_factory, "_make_model_to_tools_edge")),
        cast("FunctionType", attribute(native_factory, "_make_tools_to_model_edge")),
    )
    codes = tuple(code for owner in owners for code in nested_codes(owner.__code__))
    if not inspect.isfunction(value) or value.__code__ not in codes or value.__name__ != name:
        raise ValueError("Foreign native agent executable or routing function")
    return value


def _single(value: object, middleware: LangChainMiddleware, asynchronous: bool) -> None:
    chain = cast(
        "FunctionType",
        attribute(
            native_factory,
            "_chain_async_model_call_handlers" if asynchronous else "_chain_model_call_handlers",
        ),
    )
    if not inspect.isfunction(value) or value.__code__ not in nested_codes(chain.__code__):
        raise ValueError("Model middleware is not a native normalized single-handler chain")
    closure = inspect.getclosurevars(value).nonlocals
    if set(closure) != {"_to_composed_result", "single_handler"}:
        raise ValueError("Composed/multiple model middleware handlers are unsupported")
    single: object = closure["single_handler"]
    if not callable(single):
        raise ValueError("Missing executable model middleware")
    unwrapped = inspect.unwrap(single)
    if getattr(unwrapped, "__self__", None) is not middleware:
        raise ValueError("Native model wrapper is not bound to the exact controls middleware")


@dataclass(frozen=True)
class NativeProfile:
    """Strong ownership of exact create_agent executables and immutable system text."""

    graph: Graph
    proof: GraphProof
    middleware: LangChainMiddleware
    model: BaseChatModel
    tools: tuple[ToolProfile, ...]
    system: SystemMessage | None
    system_snapshot: dict[str, JSON] | None
    model_proof: ModelProof

    def revalidate(self) -> None:
        """Reject callback/default-handler/model/system mutation before every effect."""
        self.model_proof.revalidate()
        for tool in self.tools:
            tool.revalidate()
        if (
            attribute(_trace_policy, "_DEFAULT_TRACE_POLICY") is not None
            or self.middleware.trace_policy is not None
        ):
            raise ValueError("Custom native middleware tracing callbacks are unsupported")
        if self.system is not None and not wire_equal(
            self.system.model_dump(mode="json"), self.system_snapshot
        ):
            raise ValueError("Native system-message configuration changed")
        _installation(self.graph, self.middleware)


def _installation(
    graph: Graph, middleware: LangChainMiddleware
) -> tuple[BaseChatModel, tuple[ToolProfile, ...], SystemMessage | None]:
    specifications = native_builder(graph).nodes
    model_node = specifications["model"].runnable
    sync: object = getattr(model_node, "func", None)
    asynchronous: object = getattr(model_node, "afunc", None)
    sync_closure = inspect.getclosurevars(_native_function(sync, "model_node")).nonlocals
    async_closure = inspect.getclosurevars(_native_function(asynchronous, "amodel_node")).nonlocals
    _single(sync_closure.get("wrap_model_call_handler"), middleware, False)
    _single(async_closure.get("awrap_model_call_handler"), middleware, True)
    model = model_profile(sync_closure.get("model"))
    if (
        async_closure.get("model") is not model
        or sync_closure.get("initial_response_format") is not None
    ):
        raise ValueError("Only an explicit plain/tool-only native model is supported")
    system: object = sync_closure.get("system_message")
    if system is not async_closure.get("system_message") or (
        system is not None
        and (type(system) is not SystemMessage or not isinstance(system.content, str))
    ):
        raise ValueError("Only separate plain-text native system messages are supported")
    tools_node = specifications.get("tools")
    profiles: tuple[ToolProfile, ...] = ()
    if tools_node is not None:
        node = tools_node.runnable
        if type(node) is not ToolNode:
            raise ValueError("The exact native ToolNode is required")
        if (
            attribute(node, "_handle_tool_errors") is not default_tool_errors
            or attribute(node, "_messages_key") != "messages"
        ):
            raise ValueError("Default native tool error behavior must be unchanged")
        for property_name in ("_wrap_tool_call", "_awrap_tool_call"):
            wrapper: object = getattr(node, property_name)
            if (
                not callable(wrapper)
                or getattr(inspect.unwrap(wrapper), "__self__", None) is not middleware
            ):
                raise ValueError("Native tool wrapper must belong to the same controls middleware")
        if any(
            injected.state
            or injected.store is not None
            or injected.runtime is not None
            or injected.all_injected_keys
            for injected in cast(
                "Mapping[str, InjectedView]", attribute(node, "_injected_args")
            ).values()
        ):
            raise ValueError("State/store/runtime injected tools are unsupported")
        profiles = tools_profile(tuple(node.tools_by_name.values()))
    defaults: object = sync_closure.get("default_tools")
    if not isinstance(defaults, list):
        raise ValueError("Native tools must have a captured concrete list")
    defaults = cast("list[object]", defaults)
    if len(defaults) != len(profiles) or any(
        tool is not captured.tool for tool, captured in zip(defaults, profiles, strict=True)
    ):
        raise ValueError("Remote, structured or dynamic native tools are unsupported")
    before = specifications[f"{middleware.name}.before_agent"]
    for property_name, method in (("func", "before_agent"), ("afunc", "abefore_agent")):
        function: object = getattr(before.runnable, property_name, None)
        if getattr(function, "__self__", None) is not middleware or getattr(
            function, "__func__", None
        ) is not getattr(type(middleware), method):
            raise ValueError("Native before-agent executable is not the exact controls middleware")
    if before.trace_policy is None:
        raise ValueError("Pinned native before-agent trace policy is missing")
    for property_name in ("process_inputs", "process_outputs"):
        function: object = getattr(before.trace_policy, property_name)
        transform = cast("FunctionType", attribute(_trace_policy, "_resolved_transform"))
        if not inspect.isfunction(function) or function.__code__ not in nested_codes(
            transform.__code__
        ):
            raise ValueError("Custom middleware trace processor is unsupported")
        closure = inspect.getclosurevars(function).nonlocals
        if closure.get("mw_policy") is not None or closure.get("field") != property_name:
            raise ValueError("Custom middleware trace callback is unsupported")
    return model, profiles, system


def profile(value: object, middleware: LangChainMiddleware) -> NativeProfile:
    """Prove the complete native create_agent graph before policy loading."""
    if version("langchain") != "1.4.3":
        raise ValueError("Unsupported langchain version; expected 1.4.3")
    builder: object = getattr(value, "builder", None)
    specifications: object = getattr(builder, "nodes", None)
    if not isinstance(specifications, dict):
        raise ValueError("Builder must return the native create_agent compiled graph")
    specifications = cast("dict[str, object]", specifications)
    names = {"model", f"{middleware.name}.before_agent"}
    if "tools" in specifications:
        names.add("tools")
    if set(specifications) != names:
        raise ValueError(
            "Additional middleware, delegation or foreign native nodes are unsupported"
        )
    expected: dict[str, object] = {
        name: attribute(specifications[name], "runnable") for name in names
    }
    proof = inspect_graph(value, expected, langchain=True)
    graph = proof.graph
    builder_view = native_builder(graph)
    model, tools, system = _installation(graph, middleware)
    before = f"{middleware.name}.before_agent"
    edges = {("__start__", before), (before, "model")}
    if "tools" in specifications:
        if builder_view.edges != edges or set(builder_view.branches) != {"model", "tools"}:
            raise ValueError("Unexpected native agent edges/branches")
        for source, function_name, destinations in (
            ("model", "model_to_tools", {"tools": "tools", "__end__": "__end__", "model": "model"}),
            ("tools", "tools_to_model", {"model": "model"}),
        ):
            branches = builder_view.branches[source]
            if len(branches) != 1:
                raise ValueError("Native agent requires one exact closed routing branch")
            branch = next(iter(branches.values()))
            _native_function(getattr(branch.path, "func", None), function_name)
            if branch.ends != destinations or branch.input_schema is not None:
                raise ValueError("Native agent routing must remain closed")
    elif builder_view.edges != edges | {("model", "__end__")} or builder_view.branches:
        raise ValueError("Unexpected tool-free native agent routing")
    result = NativeProfile(
        graph,
        proof,
        middleware,
        model,
        tools,
        system,
        object_value(system.model_dump(mode="json")) if system is not None else None,
        ModelProof.capture(model),
    )
    result.revalidate()
    return result
