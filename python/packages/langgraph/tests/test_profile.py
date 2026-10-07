"""Initial native profile/options/membership rejection before policy file I/O."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from agent_hooks_adapters import NativeRuntime
from agent_hooks_langgraph import LangGraphControls
from agent_hooks_langgraph._inspection import attribute
from agent_hooks_langgraph.portable import create_backend
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    MessagesState,
    StateGraph,
)
from langgraph.prebuilt import tools_condition
from langgraph_adapter_test_helpers import graph_case, model_and_tool


def configured(builder: object, path: Path) -> object:
    return create_backend(
        runtime=builder,
        interceptors=None,
        policy_file=path,
        agents=None,
        agent_policies=(),
        resolver=None,
        timeout=5,
        emit_timeout=30,
        max_records=1000,
    )


@pytest.mark.parametrize(
    "field",
    [
        "metadata",
        "input_schema",
        "retry_policy",
        "cache_policy",
        "is_error_handler",
        "error_handler_node",
        "ends",
        "defer",
        "timeout",
        "trace_policy",
    ],
)
def test_every_state_node_spec_option_rejected(field: str, tmp_path: Path) -> None:
    case = graph_case()

    def build(controls: LangGraphControls) -> NativeRuntime:
        native = case.build(controls)
        specifications = cast(
            "dict[str, object]", attribute(attribute(native.runtime, "builder"), "nodes")
        )
        spec = specifications["model"]
        setattr(spec, field, object())
        return native

    with pytest.raises((ValueError, TypeError, AttributeError)):
        configured(build, tmp_path / "must-not-open.yaml")
    assert case.model.calls == 0
    assert case.effects == []


@pytest.mark.parametrize(
    "field",
    [
        "bound",
        "mapper",
        "writers",
        "triggers",
        "channels",
        "retry_policy",
        "cache_policy",
        "error_handler_node",
        "is_error_handler",
        "subgraphs",
        "timeout",
        "trace_policy",
    ],
)
def test_every_pregel_option_rejected_before_io(field: str, tmp_path: Path) -> None:
    case = graph_case()

    def build(controls: LangGraphControls) -> NativeRuntime:
        native = case.build(controls)
        nodes = cast("dict[str, object]", attribute(native.runtime, "nodes"))
        setattr(nodes["model"], field, object())
        return native

    with pytest.raises((ValueError, TypeError, AttributeError)):
        configured(build, tmp_path / "must-not-open.yaml")
    assert case.model.calls == 0


@pytest.mark.parametrize(
    "field",
    [
        "checkpointer",
        "store",
        "cache",
        "interrupt_before_nodes",
        "interrupt_after_nodes",
        "config",
        "context_schema",
        "step_timeout",
        "stream_transformers",
    ],
)
def test_graph_options_rejected_before_io(field: str, tmp_path: Path) -> None:
    case = graph_case()

    def build(controls: LangGraphControls) -> NativeRuntime:
        native = case.build(controls)
        setattr(native.runtime, field, object())
        return native

    with pytest.raises((ValueError, TypeError)):
        configured(build, tmp_path / "must-not-open.yaml")
    assert case.model.calls == 0


def test_foreign_plain_callable_rejected_before_io(tmp_path: Path) -> None:
    model, _, effects = model_and_tool(use_tool=False)

    def build(controls: LangGraphControls) -> NativeRuntime:
        guarded = controls.model_node("model", model)
        builder = StateGraph(MessagesState)
        builder.add_node("model", guarded)  # pyright: ignore[reportUnknownMemberType]
        builder.add_node("foreign", lambda state: state)  # pyright: ignore[reportUnknownMemberType, reportUnknownLambdaType]
        builder.add_edge(START, "model")
        builder.add_edge("model", "foreign")
        builder.add_edge("foreign", END)
        graph = builder.compile()  # pyright: ignore[reportUnknownMemberType]
        return NativeRuntime(graph, {"model": guarded})

    with pytest.raises(ValueError, match="Foreign"):
        configured(build, tmp_path / "must-not-open.yaml")
    assert effects == []


def test_custom_route_rejected_before_io(tmp_path: Path) -> None:
    model, _, _ = model_and_tool(use_tool=False)

    def build(controls: LangGraphControls) -> NativeRuntime:
        node = controls.model_node("model", model)
        builder = StateGraph(MessagesState)
        builder.add_node("model", node)  # pyright: ignore[reportUnknownMemberType]
        builder.add_edge(START, "model")

        def foreign_route(state: MessagesState) -> str:
            return END

        builder.add_conditional_edges("model", foreign_route, {"__end__": END})
        return NativeRuntime(builder.compile(), {"model": node})  # pyright: ignore[reportUnknownMemberType]

    with pytest.raises(ValueError, match="tools_condition"):
        configured(build, tmp_path / "must-not-open.yaml")
    assert model.calls == 0


def test_guards_cannot_execute_during_construction_even_if_caught(tmp_path: Path) -> None:
    model, _, _ = model_and_tool(use_tool=False)

    def build(controls: LangGraphControls) -> NativeRuntime:
        node = controls.model_node("model", model)
        with pytest.raises(RuntimeError, match="construction"):
            node.invoke({"messages": []})
        builder = StateGraph(MessagesState)
        builder.add_node("model", node)  # pyright: ignore[reportUnknownMemberType]
        builder.add_edge(START, "model")
        builder.add_edge("model", END)
        return NativeRuntime(builder.compile(), {"model": node})  # pyright: ignore[reportUnknownMemberType]

    with pytest.raises(ValueError, match="without executing"):
        configured(build, tmp_path / "must-not-open.yaml")
    assert model.calls == 0


@pytest.mark.parametrize("mapping", ["missing", "foreign", "renamed"])
def test_every_guard_must_have_exact_native_mapping(mapping: str, tmp_path: Path) -> None:
    case = graph_case()

    def build(controls: LangGraphControls) -> NativeRuntime:
        native = case.build(controls)
        members = dict(native.agents)
        if mapping == "missing":
            del members["tools"]
        elif mapping == "foreign":
            members["tools"] = object()
        else:
            members["renamed"] = members.pop("tools")
        return NativeRuntime(native.runtime, members)

    with pytest.raises(ValueError, match="every executable"):
        configured(build, tmp_path / "must-not-open.yaml")


def test_closed_tool_route_destination_must_match_actual_tools(tmp_path: Path) -> None:
    model, tool, _ = model_and_tool()
    _, foreign_tool, _ = model_and_tool()

    def build(controls: LangGraphControls) -> NativeRuntime:
        model_node = controls.model_node("model", model, [tool])
        tool_node = controls.tool_node("tools", [foreign_tool])
        builder = StateGraph(MessagesState)
        builder.add_node("model", model_node)  # pyright: ignore[reportUnknownMemberType]
        builder.add_node("tools", tool_node)  # pyright: ignore[reportUnknownMemberType]
        builder.add_edge(START, "model")
        builder.add_conditional_edges("model", tools_condition, {"tools": "tools", "__end__": END})
        builder.add_edge("tools", "model")
        return NativeRuntime(builder.compile(), {"model": model_node, "tools": tool_node})  # pyright: ignore[reportUnknownMemberType]

    with pytest.raises(ValueError, match="exact registered"):
        configured(build, tmp_path / "must-not-open.yaml")


def test_actor_cap_exact() -> None:
    from agent_hooks_adapters.runtime import validate_topology

    validate_topology(actors=16, nodes=16, edges=64)
    for counts in ((17, 16, 64), (16, 17, 64), (16, 16, 65)):
        with pytest.raises(ValueError):
            validate_topology(actors=counts[0], nodes=counts[1], edges=counts[2])
    controls = LangGraphControls()
    model, _, _ = model_and_tool(use_tool=False)
    for index in range(16):
        controls.model_node(f"actor-{index}", model)
    with pytest.raises(ValueError, match="bounded"):
        controls.model_node("seventeenth", model)


@pytest.mark.parametrize("runtime", [object(), None])
def test_existing_graph_is_not_a_builder(runtime: object, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="builder"):
        configured(runtime, tmp_path / "must-not-open.yaml")


def test_async_builder_rejected(tmp_path: Path) -> None:
    async def build(controls: LangGraphControls) -> NativeRuntime:
        raise AssertionError("Async builder must not be invoked")

    with pytest.raises(TypeError, match="synchronous"):
        configured(build, tmp_path / "must-not-open.yaml")
