"""Reject unsupported create_agent surfaces and require exact installed wrapper identity."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from agent_control_adapters import NativeRuntime
from agent_control_langchain import LangChainControls
from agent_control_langchain.portable import create_backend
from agent_control_langgraph._inspection import attribute
from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
from langchain.agents.middleware import AgentMiddleware
from langchain_adapter_test_helpers import chain_case
from langgraph_adapter_test_helpers import model_and_tool


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


def foreign_handler(value: object) -> object:
    return value


def foreign_recovery(value: object) -> str:
    return "recover"


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
    ],
)
def test_native_node_options_rejected_before_io(field: str, tmp_path: Path) -> None:
    case = chain_case()

    def build(controls: LangChainControls) -> NativeRuntime:
        native = case.build(controls)
        specifications = cast(
            "dict[str, object]", attribute(attribute(native.runtime, "builder"), "nodes")
        )
        setattr(specifications["model"], field, object())
        return native

    with pytest.raises((ValueError, TypeError, AttributeError)):
        configured(build, tmp_path / "must-not-open.yaml")
    assert case.model.calls == 0


@pytest.mark.parametrize("field", ["func", "afunc"])
def test_model_closure_must_be_actual_native(field: str, tmp_path: Path) -> None:
    case = chain_case()

    def build(controls: LangChainControls) -> NativeRuntime:
        native = case.build(controls)
        specifications = cast(
            "dict[str, object]", attribute(attribute(native.runtime, "builder"), "nodes")
        )
        setattr(attribute(specifications["model"], "runnable"), field, foreign_handler)
        return native

    with pytest.raises(ValueError, match="executable"):
        configured(build, tmp_path / "must-not-open.yaml")
    assert case.effects == []


@pytest.mark.parametrize("field", ["_wrap_tool_call", "_awrap_tool_call", "_handle_tool_errors"])
def test_exact_tool_wrapper_and_default_handler_identity(field: str, tmp_path: Path) -> None:
    case = chain_case()

    def build(controls: LangChainControls) -> NativeRuntime:
        native = case.build(controls)
        specifications = cast(
            "dict[str, object]", attribute(attribute(native.runtime, "builder"), "nodes")
        )
        setattr(attribute(specifications["tools"], "runnable"), field, foreign_recovery)
        return native

    with pytest.raises(ValueError):
        configured(build, tmp_path / "must-not-open.yaml")
    assert case.effects == []


def test_multiple_middleware_rejected(tmp_path: Path) -> None:
    model, tool, effects = model_and_tool()

    class ForeignMiddleware(AgentMiddleware):
        def before_model(self, state: object, runtime: object) -> None:
            raise AssertionError("Foreign middleware must not execute")

    def build(controls: LangChainControls) -> NativeRuntime:
        graph = create_agent(
            model, [tool], middleware=[controls.middleware("agent"), ForeignMiddleware()]
        )
        return NativeRuntime(graph, {"agent": graph})

    with pytest.raises(ValueError, match="Additional middleware"):
        configured(build, tmp_path / "must-not-open.yaml")
    assert effects == []


def test_bare_graph_and_wrong_actor_rejected(tmp_path: Path) -> None:
    case = chain_case()
    native = case.build(LangChainControls())
    with pytest.raises(TypeError, match="builder"):
        configured(native.runtime, tmp_path / "must-not-open.yaml")

    def build(controls: LangChainControls) -> NativeRuntime:
        proper = case.build(controls)
        return NativeRuntime(proper.runtime, {"agent": object()})

    with pytest.raises(ValueError, match="one exact"):
        configured(build, tmp_path / "must-not-open.yaml")


def test_no_native_async_executor_fallback(tmp_path: Path) -> None:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    model = FakeMessagesListChatModel(responses=[AIMessage(content="unit I/O")])

    def build(controls: LangChainControls) -> NativeRuntime:
        graph = create_agent(model, middleware=[controls.middleware("agent")])
        return NativeRuntime(graph, {"agent": graph})

    with pytest.raises(ValueError, match="explicit sync/async"):
        configured(build, tmp_path / "must-not-open.yaml")


def test_tool_async_fallback_rejected(tmp_path: Path) -> None:
    from langchain_core.tools import StructuredTool

    model, _, _ = model_and_tool()

    def ping(value: str) -> str:
        return value

    tool = StructuredTool.from_function(ping, description="Unit local tool")

    def build(controls: LangChainControls) -> NativeRuntime:
        graph = create_agent(model, [tool], middleware=[controls.middleware("agent")])
        return NativeRuntime(graph, {"agent": graph})

    with pytest.raises(ValueError, match="explicit sync/async"):
        configured(build, tmp_path / "must-not-open.yaml")
