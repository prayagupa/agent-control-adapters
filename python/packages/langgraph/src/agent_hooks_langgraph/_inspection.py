"""Version-pinned, read-only proof of the actual executable compiled graph."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, fields
from importlib.metadata import version
from types import CodeType, FunctionType
from typing import Protocol, cast

from agent_hooks_adapters.runtime import validate_topology
from langgraph.channels import binop  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    START,
    MessagesState,
    StateGraph,
)
from langgraph.graph._branch import BranchSpec  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph._node import StateNodeSpec  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph.message import add_messages  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph.state import (  # pyright: ignore[reportMissingTypeStubs, reportPrivateUsage]
    CompiledStateGraph,
    _control_branch,  # pyright: ignore[reportPrivateUsage]
)
from langgraph.pregel._read import DEFAULT_BOUND  # pyright: ignore[reportMissingTypeStubs]
from langgraph.pregel._write import ChannelWrite  # pyright: ignore[reportMissingTypeStubs]

Graph = CompiledStateGraph[MessagesState, None, MessagesState, MessagesState]
BinaryOperatorAggregate = binop.BinaryOperatorAggregate


class NodeSpecView(Protocol):
    """Typed view of every pinned StateNodeSpec field read by the inspector."""

    runnable: object
    metadata: object
    input_schema: object
    retry_policy: object
    cache_policy: object
    is_error_handler: bool
    error_handler_node: object
    ends: object
    defer: bool
    timeout: object
    trace_policy: object


class PregelView(Protocol):
    """Typed view of every pinned PregelNode execution field."""

    channels: str | list[str]
    triggers: list[str]
    mapper: object
    writers: list[object]
    bound: object
    cache_policy: object
    retry_policy: object
    timeout: object
    tags: object
    metadata: object
    trace_policy: object
    is_error_handler: bool
    error_handler_node: object
    subgraphs: list[object]


class BranchView(Protocol):
    """Closed native branch inventory without SDK generic Unknown propagation."""

    path: object
    ends: dict[str, str] | None
    input_schema: object


class BuilderView(Protocol):
    """Read-only typed boundary over a nominally validated native StateGraph."""

    nodes: dict[str, NodeSpecView]
    edges: set[tuple[str, str]]
    branches: Mapping[str, Mapping[str, BranchView]]
    channels: Mapping[str, object]
    schemas: object
    managed: object
    waiting_edges: set[tuple[tuple[str, ...], str]]
    compiled: bool
    state_schema: object
    input_schema: object
    output_schema: object
    context_schema: object


def native_builder(graph: Graph) -> BuilderView:
    """Expose an explicit typed view only after nominal StateGraph ownership validation."""
    builder: object = graph.builder
    if type(builder) is not StateGraph:
        raise ValueError("A concrete native StateGraph builder is required")
    return cast("BuilderView", builder)


def native_nodes(graph: Graph) -> Mapping[str, PregelView]:
    """Expose actual compiled nodes through the complete pinned inventory."""
    return cast("Mapping[str, PregelView]", attribute(graph, "nodes"))


_SPEC_FIELDS = {
    "runnable",
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
}
_PREGEL_FIELDS = {
    "channels",
    "triggers",
    "mapper",
    "writers",
    "bound",
    "cache_policy",
    "retry_policy",
    "timeout",
    "tags",
    "metadata",
    "trace_policy",
    "is_error_handler",
    "error_handler_node",
    "subgraphs",
}
_CACHED_FIELDS = {"flat_writers", "node", "input_cache_key"}
_GRAPH_FIELDS = {
    "nodes",
    "channels",
    "stream_mode",
    "stream_eager",
    "output_channels",
    "stream_channels",
    "interrupt_after_nodes",
    "interrupt_before_nodes",
    "input_channels",
    "step_timeout",
    "debug",
    "checkpointer",
    "store",
    "cache",
    "retry_policy",
    "cache_policy",
    "context_schema",
    "config",
    "trigger_to_nodes",
    "node_error_handler_map",
    "name",
    "stream_transformers",
    "_serde_allowlist",
    "builder",
    "schema_to_mapper",
}
_GRAPH_OPTIONAL_FIELDS = {"__orig_class__", "_state_mapper", "_output_mapper"}


def attribute(value: object, name: str) -> object:
    """Read a required pinned native attribute without a silent absent-field fallback."""
    return getattr(value, name)


def nested_codes(code: CodeType) -> tuple[CodeType, ...]:
    """Inventory exact nested native code objects without executing any model/tool."""
    return (
        code,
        *(
            child
            for value in code.co_consts
            if isinstance(value, CodeType)
            for child in nested_codes(value)
        ),
    )


@dataclass(frozen=True, eq=False)
class Identity:
    """Keep captured objects strongly owned and compare only actual object identity."""

    value: object

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Identity) and self.value is other.value

    def __hash__(self) -> int:
        return id(self.value)


def shape(value: object, depth: int = 4) -> object:
    """Capture bounded container contents as well as their retained identities."""
    if value is None or isinstance(value, str | bool | int | float):
        return type(value), value
    if depth <= 0:
        return Identity(value)
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        return Identity(mapping), tuple(
            (shape(key, depth - 1), shape(item, depth - 1)) for key, item in mapping.items()
        )
    if isinstance(value, list | tuple):
        sequence = cast("list[object] | tuple[object, ...]", value)
        return Identity(sequence), tuple(shape(item, depth - 1) for item in sequence)
    if isinstance(value, set | frozenset):
        members = cast("set[object] | frozenset[object]", value)
        return Identity(members), frozenset(
            cast("str | tuple[object, ...]", shape(item, depth - 1)) for item in members
        )
    if inspect.ismethod(value):
        return Identity(value.__self__), Identity(value.__func__)
    if inspect.isfunction(value):
        closure = inspect.getclosurevars(value).nonlocals
        return (
            Identity(value),
            Identity(value.__code__),
            tuple((key, shape(item, depth - 1)) for key, item in closure.items()),
        )
    return Identity(value)


def _attributes(value: object, names: set[str]) -> object:
    return tuple((name, shape(getattr(value, name))) for name in sorted(names))


def _writers(value: object) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError("Unsupported native writer collection")
    result: list[object] = []
    for writer in cast("list[object]", value):
        module = type(writer).__module__
        if module not in ("langgraph.pregel._write", "langgraph._internal._runnable"):
            raise ValueError("Foreign executable graph writer")
        result.append((Identity(writer), shape(vars(writer))))
    return tuple(result)


def _validate_writers(name: str, node: PregelView, builder: BuilderView) -> None:
    if not isinstance(cast("object", node.writers), list) or not node.writers:
        raise ValueError("Native executable writers are required")
    expected_count = (
        1
        + len(builder.branches.get(name, {}))
        + int(
            any(
                source == name and destination != "__end__" for source, destination in builder.edges
            )
        )
    )
    if len(node.writers) != expected_count:
        raise ValueError("Native writer set does not match complete closed topology")
    branches = tuple(builder.branches.get(name, {}).values())
    codes = nested_codes(
        cast("FunctionType", attribute(CompiledStateGraph, "attach_node")).__code__
    )
    for writer in node.writers:
        if type(writer) is ChannelWrite:
            if (
                getattr(attribute(writer, "func"), "__self__", None) is not writer
                or getattr(attribute(writer, "func"), "__func__", None)
                is not attribute(ChannelWrite, "_write")
                or getattr(attribute(writer, "afunc"), "__func__", None)
                is not attribute(ChannelWrite, "_awrite")
            ):
                raise ValueError("Foreign native channel writer implementation")
            for entry in cast("tuple[object, ...]", attribute(writer, "writes")):
                mapper: object = getattr(entry, "mapper", None)
                if (
                    mapper is not None
                    and mapper is not _control_branch
                    and (not inspect.isfunction(mapper) or mapper.__code__ not in codes)
                ):
                    raise ValueError("Foreign native channel update mapper")
        else:
            sync = attribute(writer, "func")
            asynchronous = attribute(writer, "afunc")
            if (
                getattr(sync, "__func__", None) is not attribute(BranchSpec, "_route")
                or getattr(asynchronous, "__func__", None) is not attribute(BranchSpec, "_aroute")
                or not any(getattr(sync, "__self__", None) is branch for branch in branches)
                or getattr(sync, "__self__", None) is not getattr(asynchronous, "__self__", None)
            ):
                raise ValueError("Foreign executable routing writer")


def _writer_effects(value: object) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError("Unsupported cached graph writers")
    effects: list[object] = []
    for writer in cast("list[object]", value):
        if isinstance(writer, ChannelWrite):
            effects.extend(shape(entry) for entry in writer.writes)
        else:
            effects.append(Identity(writer))
    return tuple(effects)


def _cached(node: object) -> None:
    attributes = vars(node)
    writers: object = attributes["writers"]
    if "flat_writers" in attributes:
        flat: object = attributes["flat_writers"]
        if _writer_effects(flat) != _writer_effects(writers):
            raise ValueError("Cached native writer binding changed")
    if "node" in attributes:
        executable: object = attributes["node"]
        steps: object = getattr(executable, "steps", None)
        expected: object = attributes.get("flat_writers", writers)
        if not isinstance(expected, list):
            raise ValueError("Unsupported cached native executable")
        expected = cast("list[object]", expected)
        if attributes["bound"] is DEFAULT_BOUND and len(expected) == 1:
            if executable is not expected[0]:
                raise ValueError("Cached start writer differs from the native writer")
            return
        if not isinstance(steps, list | tuple):
            raise ValueError("Unsupported cached native executable")
        actual_steps = cast("list[object] | tuple[object, ...]", steps)
        expected_steps = expected
        if attributes["bound"] is not DEFAULT_BOUND:
            expected_steps = [attributes["bound"], *expected_steps]
        if len(actual_steps) != len(expected_steps) or any(
            first is not second for first, second in zip(actual_steps, expected_steps, strict=True)
        ):
            raise ValueError("Cached executable differs from the actual bound nodes")
    if "input_cache_key" in attributes:
        channels: object = attributes["channels"]
        expected_key = (
            attributes["mapper"],
            tuple(cast("list[object]", channels)) if isinstance(channels, list) else (channels,),
        )
        if attributes["input_cache_key"] != expected_key:
            raise ValueError("Cached native input mapper changed")


def _capture(graph: Graph) -> object:
    builder = native_builder(graph)
    specifications = tuple(
        (name, Identity(spec), _attributes(spec, _SPEC_FIELDS))
        for name, spec in builder.nodes.items()
    )
    actual: list[object] = []
    for name, node in native_nodes(graph).items():
        if set(vars(node)) - _CACHED_FIELDS != _PREGEL_FIELDS:
            raise ValueError("Pinned PregelNode field inventory changed")
        _cached(node)
        actual.append(
            (name, Identity(node), _attributes(node, _PREGEL_FIELDS), _writers(node.writers))
        )
    channels = tuple(
        (
            name,
            Identity(channel),
            Identity(type(channel)),
            shape(getattr(channel, "operator", None)),
        )
        for name, channel in cast("Mapping[str, object]", attribute(graph, "channels")).items()
    )
    return (
        Identity(graph),
        Identity(builder),
        specifications,
        tuple(actual),
        channels,
        _attributes(graph, _GRAPH_FIELDS - {"nodes", "channels", "schema_to_mapper"}),
        shape(builder.edges),
        shape(builder.branches),
        shape(builder.waiting_edges),
        shape(builder.schemas),
        shape(builder.channels),
        shape(builder.managed),
        Identity(builder.state_schema),
        Identity(builder.input_schema),
        Identity(builder.output_schema),
        Identity(builder.context_schema),
        tuple((name, shape(getattr(graph, name, None))) for name in sorted(_GRAPH_OPTIONAL_FIELDS)),
        _attributes(
            attribute(builder, "_node_defaults"),
            {"retry_policy", "cache_policy", "error_handler", "timeout"},
        ),
        tuple(
            (name, shape(vars(branch.path)), shape(branch.ends), Identity(branch.input_schema))
            for branches in builder.branches.values()
            for name, branch in branches.items()
        ),
        tuple(
            (name, shape(vars(spec.runnable)))
            for name, spec in builder.nodes.items()
            if hasattr(spec.runnable, "__dict__")
        ),
    )


@dataclass(frozen=True)
class GraphProof:
    """Hold an immutable proof of the actual graph; revalidate before every effect."""

    graph: Graph
    snapshot: object

    def revalidate(self) -> None:
        """Fail explicitly if any captured topology, channel or executable changed."""
        if _capture(self.graph) != self.snapshot:
            raise ValueError("Native compiled graph topology or executable binding changed")


def inspect_graph(
    value: object, expected: Mapping[str, object], *, langchain: bool = False
) -> GraphProof:
    """Validate every native node field and actual binding before policy file I/O."""
    for package, pin in (
        ("langgraph", "1.2.13"),
        ("langgraph-prebuilt", "1.1.0"),
        ("langchain-core", "1.6.6"),
    ):
        if version(package) != pin:
            raise ValueError(f"Unsupported {package} version; expected {pin}")
    if type(value) is not CompiledStateGraph:
        raise ValueError("A native CompiledStateGraph from the trusted builder is required")
    graph = cast("Graph", value)
    builder = native_builder(graph)
    captured_specifications = tuple(builder.nodes.items())
    if (
        set(vars(graph)) - _GRAPH_OPTIONAL_FIELDS != _GRAPH_FIELDS
        or {field.name for field in fields(StateNodeSpec)} != _SPEC_FIELDS
        or getattr(graph, "_state_mapper", None) is not None
        or getattr(graph, "_output_mapper", None) is not None
    ):
        raise ValueError("Pinned native graph field inventory changed")
    edge_count = len(builder.edges) + sum(
        len(branch.ends or {})
        for branches in builder.branches.values()
        for branch in branches.values()
    )
    validate_topology(actors=len(expected), nodes=len(builder.nodes), edges=edge_count)
    actual_nodes = native_nodes(graph)
    if set(builder.nodes) != set(expected) or set(actual_nodes) != {START, *expected}:
        raise ValueError("Complete executable graph membership must match guarded builder nodes")
    if (
        builder.waiting_edges
        or builder.managed
        or not builder.compiled
        or builder.context_schema is not None
        or graph.context_schema is not None
        or any(
            attribute(attribute(builder, "_node_defaults"), name) is not None
            for name in ("retry_policy", "cache_policy", "error_handler", "timeout")
        )
        or attribute(graph, "checkpointer") is not None
        or graph.store is not None
        or attribute(graph, "cache") is not None
        or attribute(graph, "cache_policy") is not None
        or graph.retry_policy
        or graph.interrupt_before_nodes
        or graph.interrupt_after_nodes
        or graph.step_timeout is not None
        or graph.debug
        or graph.node_error_handler_map
        or (graph.stream_transformers and not langchain)
        or attribute(graph, "_serde_allowlist") is not None
    ):
        raise ValueError(
            "Persistence, retries, callbacks, interrupts and custom graph options are unsupported"
        )
    if langchain:
        if tuple(
            (transformer.__module__, transformer.__name__)
            for transformer in graph.stream_transformers
        ) != (
            ("langgraph.prebuilt._tool_call_transformer", "ToolCallTransformer"),
            ("langchain.agents._subagent_transformer", "SubagentTransformer"),
        ):
            raise ValueError("Only the exact pinned native stream transformers are supported")
        if graph.config != {
            "recursion_limit": 9999,
            "metadata": {"ls_integration": "langchain_create_agent"},
            "configurable": {},
        }:
            raise ValueError("Custom native agent configuration is unsupported")
        schema_names = ("StateSchema", "InputSchema", "OutputSchema")
        for schema, name in zip(
            (builder.state_schema, builder.input_schema, builder.output_schema),
            schema_names,
            strict=True,
        ):
            if (
                getattr(schema, "__module__", None) != "langchain.agents.factory"
                or getattr(schema, "__name__", None) != name
            ):
                raise ValueError("Custom LangChain state/input/output schemas are unsupported")
        expected_channels = {"messages", "jump_to", "structured_response"}
    else:
        if (
            builder.state_schema is not MessagesState
            or builder.input_schema is not MessagesState
            or builder.output_schema is not MessagesState
            or graph.config is not None
        ):
            raise ValueError("Only ordinary StateGraph(MessagesState) is supported")
        expected_channels = {"messages"}
    captured_channels = tuple(native_builder(graph).channels.items())
    if set(builder.channels) != expected_channels:
        raise ValueError("Custom state channels/reducers are unsupported")
    message_channel = builder.channels["messages"]
    if (
        type(message_channel) is not BinaryOperatorAggregate
        or getattr(message_channel, "operator", None) is not add_messages
    ):
        raise ValueError("The exact native messages reducer is required")
    if actual_nodes[START].bound is not DEFAULT_BOUND:
        raise ValueError("Foreign native start executable")
    compiled_channels = cast("Mapping[str, object]", attribute(graph, "channels"))
    if set(compiled_channels) != {
        START,
        "__pregel_tasks",
        *expected_channels,
        *(f"branch:to:{name}" for name in expected),
    } or any(compiled_channels[name] is not channel for name, channel in captured_channels):
        raise ValueError("Actual compiled channel set differs from the native state graph")
    for name, actual in actual_nodes.items():
        if (
            actual.mapper is not None
            or actual.channels != (START if name == START else list(builder.channels))
            or actual.triggers != [START if name == START else f"branch:to:{name}"]
        ):
            raise ValueError("Actual mapper/channels/triggers differ from the native graph")
        _validate_writers(name, actual, builder)
    for name, specification in captured_specifications:
        actual = actual_nodes[name]
        if (
            specification.runnable is not expected[name]
            or actual.bound is not specification.runnable
        ):
            raise ValueError("Actual compiled binding differs from the guarded native object")
        if (
            specification.metadata is not None
            or specification.retry_policy is not None
            or specification.cache_policy is not None
            or specification.is_error_handler
            or specification.error_handler_node is not None
            or specification.ends not in ((), None)
            or specification.defer
            or specification.timeout is not None
            or (not langchain and specification.trace_policy is not None)
            or specification.input_schema is not builder.state_schema
            or actual.retry_policy
            or actual.cache_policy is not None
            or actual.timeout is not None
            or actual.is_error_handler
            or actual.error_handler_node is not None
            or actual.subgraphs
            or actual.metadata is not None
            or actual.tags is not None
            or actual.trace_policy is not specification.trace_policy
        ):
            raise ValueError("Unsupported StateNodeSpec or PregelNode option")
    return GraphProof(graph, _capture(graph))
