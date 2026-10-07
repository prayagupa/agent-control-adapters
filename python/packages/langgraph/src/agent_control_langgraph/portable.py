"""Portable constructor with strict application-owned native graph validation."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from agent_control_adapters import AgentPolicy, AsyncAdapter, NativeRuntime
from agent_control_adapters.bindings import prepare_controls
from agent_hooks import ApprovalResolver, Interceptor
from langgraph.graph import END, START  # pyright: ignore[reportMissingTypeStubs]
from langgraph.prebuilt import tools_condition

from ._inspection import inspect_graph, native_builder
from ._routing import Route
from ._scope import GraphAdapter, NativeScope
from .controls import GuardedModelNode, GuardedToolNode, LangGraphControls


def create_backend(
    *,
    runtime: object,
    interceptors: Sequence[Interceptor] | None,
    policy_file: str | Path | None,
    agents: Mapping[str, object] | None,
    agent_policies: Sequence[AgentPolicy],
    resolver: ApprovalResolver | None,
    timeout: float,
    emit_timeout: float,
    max_records: int,
) -> AsyncAdapter:
    """Build inactive guards, prove native membership/routing, then load policies."""
    if (
        not callable(runtime)
        or inspect.iscoroutinefunction(runtime)
        or inspect.isasyncgenfunction(runtime)
    ):
        raise TypeError("LangGraph runtime must be a synchronous application-owned builder")
    controller = LangGraphControls()
    native: object = cast("Callable[[LangGraphControls], object]", runtime)(controller)
    controller.seal()
    if inspect.iscoroutine(native):
        native.close()
        raise TypeError("LangGraph builders must return NativeRuntime synchronously")
    if not isinstance(native, NativeRuntime) or controller.construction_violation:
        raise ValueError("Builder must return NativeRuntime without executing any guarded node")
    nodes = controller.nodes
    for node in nodes:
        node.revalidate()
    expected_actors = {node.label: node for node in nodes}
    if set(native.agents) != set(expected_actors) or any(
        native.agents[label] is not node for label, node in expected_actors.items()
    ):
        raise ValueError("NativeRuntime.agents must map every executable guarded node exactly once")
    if agents is not None and (
        set(agents) != set(native.agents)
        or any(agents[label] is not node for label, node in native.agents.items())
    ):
        raise ValueError("Caller agents must exactly match the captured native mapping")
    graph_value = native.runtime
    builder: object = getattr(graph_value, "builder", None)
    specifications: object = getattr(builder, "nodes", None)
    if not isinstance(specifications, Mapping):
        raise ValueError("The builder must compile a native StateGraph(MessagesState)")
    expected: dict[str, object] = {}
    for name, specification in cast("Mapping[str, object]", specifications).items():
        runnable: object = getattr(specification, "runnable", None)
        if not any(runnable is node for node in nodes):
            raise ValueError("Foreign/plain callable graph nodes are unsupported")
        expected[name] = runnable
    if len(expected) != len(nodes) or len({id(node) for node in expected.values()}) != len(nodes):
        raise ValueError("Each guarded node must appear exactly once in the native graph")
    proof = inspect_graph(graph_value, expected)
    graph = proof.graph
    native_graph_builder = native_builder(graph)
    starts = [destination for source, destination in native_graph_builder.edges if source == START]
    if len(starts) != 1 or not isinstance(expected[starts[0]], GuardedModelNode):
        raise ValueError("Graph must have exactly one guarded model entry")
    entry = expected[starts[0]]
    routes: list[Route] = []
    adjacency: dict[str, set[str]] = {START: set(starts)}
    for source, destination in native_graph_builder.edges:
        if (
            source not in {START, *expected}
            or destination not in {END, *expected}
            or (destination in expected and isinstance(expected[destination], GuardedToolNode))
        ):
            raise ValueError("Only ordinary closed edges may connect guarded model nodes")
        adjacency.setdefault(source, set()).add(destination)
    for source, branches in native_graph_builder.branches.items():
        node = expected.get(source)
        if not isinstance(node, GuardedModelNode) or len(branches) != 1:
            raise ValueError("Only model nodes may have the exact native tools_condition branch")
        branch = next(iter(branches.values()))
        if (
            getattr(branch.path, "func", None) is not tools_condition
            or branch.input_schema is not None
            or branch.ends is None
            or set(branch.ends) not in ({"__end__"}, {"tools", "__end__"})
            or branch.ends.get("__end__") != END
            or source in {edge[0] for edge in native_graph_builder.edges}
        ):
            raise ValueError(
                "Conditional routing must use exact tools_condition and a closed destination map"
            )
        destination = branch.ends.get("tools")
        if node.tools:
            target = expected.get(destination or "")
            if (
                not isinstance(target, GuardedToolNode)
                or len(node.tools) != len(target.tools)
                or any(
                    first.tool is not second.tool
                    for first, second in zip(node.tools, target.tools, strict=True)
                )
            ):
                raise ValueError("Model tools must route to the exact registered guarded tool node")
            routes.append(Route(node, target))
        elif destination is not None:
            raise ValueError("A tool-free model cannot route to a tool node")
        adjacency.setdefault(source, set()).update(branch.ends.values())
    for name, node in expected.items():
        if (
            isinstance(node, GuardedModelNode)
            and node.tools
            and not any(route.origin is node for route in routes)
        ):
            raise ValueError("Every tool-enabled model requires a proven static tool route")
        if isinstance(node, GuardedToolNode) and (
            not any(route.destination is node for route in routes)
            or len(adjacency.get(name, ())) != 1
            or not isinstance(
                expected.get(next(iter(adjacency.get(name, ())), "")), GuardedModelNode
            )
        ):
            raise ValueError("Tool nodes require a closed guarded-model continuation")
    reachable: set[str] = set()
    waiting = [START]
    while waiting:
        name = waiting.pop()
        if name not in reachable:
            reachable.add(name)
            waiting.extend(adjacency.get(name, ()))
    if not set(expected) <= reachable:
        raise ValueError("Every guarded executable must be reachable from the native entry")
    prepared = prepare_controls(
        native.agents,
        interceptors=interceptors,
        policy_file=policy_file,
        agents=agents,
        agent_policies=agent_policies,
        resolver=resolver,
        timeout=timeout,
        emit_timeout=emit_timeout,
        max_records=max_records,
    )

    def revalidate() -> None:
        for node in nodes:
            node.revalidate()

    scope = NativeScope(
        "langgraph",
        prepared,
        proof,
        entry,
        tuple((node, node.tools) for node in nodes),
        tuple(routes),
        revalidate,
    )
    controller.attach(scope)
    return GraphAdapter(scope)
