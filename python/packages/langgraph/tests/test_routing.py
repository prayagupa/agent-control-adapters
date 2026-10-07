"""Real native model/tool guard receipts and bounded static-route attack vectors."""

from __future__ import annotations

import asyncio

import pytest
from agent_control_adapters import AdapterBlocked, NativeRuntime, RunRequest
from agent_control_langgraph import GuardedModelNode, GuardedToolNode, LangGraphControls
from agent_control_langgraph._native import model_value
from agent_control_langgraph._routing import Route, RoutingLedger
from agent_control_langgraph.portable import create_backend
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    MessagesState,
    StateGraph,
)
from langgraph.prebuilt import tools_condition
from langgraph_adapter_test_helpers import NativePolicy, graph_adapter, graph_case, model_and_tool


def test_wrong_node_and_wrong_origin_do_not_consume_source_budget() -> None:
    case = graph_case()
    graph_adapter(case, NativePolicy())
    assert case.controls is not None
    scope = case.controls.scope()
    model = next(node for node in case.controls.nodes if isinstance(node, GuardedModelNode))
    tool = next(node for node in case.controls.nodes if isinstance(node, GuardedToolNode))
    scope.session.begin()
    scope.active = True
    try:
        state = model.invoke({"messages": [HumanMessage(content="actual native call")]})
        message = state["messages"][0]
        assert isinstance(message, AIMessage)
        projected = model_value(message, model.tools)
        calls = projected["tool_calls"]
        assert isinstance(calls, list) and isinstance(calls[0], dict)
        call = calls[0]
        assert scope.session.budget.snapshot.pending == scope.ledger.pending == 1
        with pytest.raises(ValueError, match="Wrong node"):
            scope.ledger.before(object(), call)
        with pytest.raises(ValueError, match="Unproven"):
            scope.ledger.reserve(object(), call)
        assert scope.session.budget.snapshot.pending == scope.ledger.pending == 1
        result = tool.invoke(state)
        assert result["messages"][0].type == "tool"
        assert case.effects == ["original"]
        assert scope.session.budget.snapshot.pending == scope.ledger.pending == 0
        with pytest.raises(AdapterBlocked):
            tool.invoke(state)
        assert case.effects == ["original"]
    finally:
        scope.active = False
        scope.ledger.close()
        scope.session.close(RuntimeError("unit terminal cleanup"))
    assert scope.ledger.pending == 0


def test_duplicate_pending_and_cross_actor_replay_are_blocked() -> None:
    model, tool, effects = model_and_tool()
    original = model.responses[0]
    model.responses[1] = original.model_copy()

    def build(controls: LangGraphControls) -> NativeRuntime:
        first = controls.model_node("first", model, [tool])
        first_tools = controls.tool_node("first-tools", [tool])
        second = controls.model_node("second", model, [tool])
        second_tools = controls.tool_node("second-tools", [tool])
        builder = StateGraph(MessagesState)
        for name, node in (
            ("first", first),
            ("first-tools", first_tools),
            ("second", second),
            ("second-tools", second_tools),
        ):
            builder.add_node(name, node)  # pyright: ignore[reportUnknownMemberType]
        builder.add_edge(START, "first")
        builder.add_conditional_edges(
            "first", tools_condition, {"tools": "first-tools", "__end__": END}
        )
        builder.add_conditional_edges(
            "second", tools_condition, {"tools": "second-tools", "__end__": END}
        )
        builder.add_edge("first-tools", "second")
        builder.add_edge("second-tools", "second")
        return NativeRuntime(
            builder.compile(),  # pyright: ignore[reportUnknownMemberType]
            {
                "first": first,
                "first-tools": first_tools,
                "second": second,
                "second-tools": second_tools,
            },
        )

    adapter = create_backend(
        runtime=build,
        interceptors=[NativePolicy()],
        policy_file=None,
        agents=None,
        agent_policies=(),
        resolver=None,
        timeout=5,
        emit_timeout=30,
        max_records=1000,
    )
    with pytest.raises(AdapterBlocked) as raised:
        asyncio.run(adapter.arun(RunRequest({"prompt": "real graph route"})))
    assert isinstance(raised.value.__cause__, ValueError)
    assert model.calls == 2
    assert effects == ["original"]
    assert (
        sum(record.interception_point.value == "pre_tool_call" for record in adapter.records) == 1
    )


def test_duplicate_pending_model_completion_blocks_before_any_tool() -> None:
    case = graph_case()
    case.model.responses[1] = case.model.responses[0].model_copy()
    graph_adapter(case, NativePolicy())
    assert case.controls is not None
    scope = case.controls.scope()
    model = next(node for node in case.controls.nodes if isinstance(node, GuardedModelNode))
    scope.session.begin()
    scope.active = True
    primary = RuntimeError("unit cleanup")
    try:
        state: MessagesState = {"messages": [HumanMessage(content="actual transport")]}
        model.invoke(state)
        with pytest.raises(AdapterBlocked):
            model.invoke(state)
        assert case.model.calls == 2
        assert case.effects == []
        assert scope.ledger.pending == 1
    finally:
        scope.active = False
        scope.ledger.close()
        scope.session.close(primary)
    assert scope.ledger.pending == scope.session.budget.snapshot.pending == 0


def test_pending_route_cap_and_cleanup_no_fabricated_records() -> None:
    case = graph_case()
    graph_adapter(case, NativePolicy())
    assert case.controls is not None
    scope = case.controls.scope()
    model = next(node for node in case.controls.nodes if isinstance(node, GuardedModelNode))
    tool = next(node for node in case.controls.nodes if isinstance(node, GuardedToolNode))
    ledger = RoutingLedger(scope.session, (Route(model, tool),))
    # This is an isolated ledger threshold test, not fabricated model completion/audit.
    for index in range(64):
        ledger.reserve(
            model, {"id": f"correlation-{index}", "name": "ping", "args": {"value": "x"}}
        )
    assert ledger.pending == 64
    with pytest.raises(ValueError, match="oversized"):
        ledger.reserve(model, {"id": "sixty-fifth", "name": "ping", "args": {"value": "x"}})
    assert scope.session.records == ()
    ledger.close()
    assert ledger.pending == 0
