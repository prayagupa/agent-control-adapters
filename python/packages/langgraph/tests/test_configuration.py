"""Reject sequential configuration and actual guarded binding mutation before native effects."""

from typing import cast

import pytest
from agent_control_adapters import AdapterBlocked, RunRequest
from agent_control_langgraph import LangGraphControls
from agent_control_langgraph._inspection import attribute
from agent_control_langgraph._native import ModelProof
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableBinding
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph_adapter_test_helpers import NativePolicy, NativeTransport, graph_adapter, graph_case


def replacement(value: object) -> object:
    return value


@pytest.mark.parametrize("field", ["name", "metadata"])
def test_model_settings_changed_before_run(field: str) -> None:
    case = graph_case()
    adapter = graph_adapter(case, NativePolicy())
    setattr(case.model, field, "changed" if field == "name" else {"lc_versions": {"x": "changed"}})
    with pytest.raises((ValueError, AdapterBlocked), match="configuration changed"):
        adapter.run(RunRequest({"prompt": "actual native admission"}))
    assert case.model.calls == 0 and case.effects == []


@pytest.mark.parametrize("field", ["description", "verbose", "tags", "metadata"])
def test_tool_settings_changed_before_run(field: str) -> None:
    case = graph_case()
    adapter = graph_adapter(case, NativePolicy())
    value: object = {
        "description": "changed",
        "verbose": True,
        "tags": ["changed"],
        "metadata": {"changed": True},
    }[field]
    setattr(case.tool, field, value)
    with pytest.raises((ValueError, AdapterBlocked), match="configuration changed"):
        adapter.run(RunRequest({"prompt": "actual native admission"}))
    assert case.model.calls == 0 and case.effects == []


def test_instance_transport_replacement_rejected() -> None:
    model = NativeTransport(responses=[AIMessage(content="unit transport")])
    proof = ModelProof.capture(model)
    setattr(model, "_generate", replacement)  # noqa: B010 - intentional SDK mutation
    with pytest.raises(ValueError, match="configuration changed"):
        proof.revalidate()


def test_guard_method_replacement_rejected() -> None:
    case = graph_case()
    adapter = graph_adapter(case, NativePolicy())
    assert case.controls is not None
    node = case.controls.nodes[0]
    setattr(node, "invoke", replacement)  # noqa: B010 - intentional guarded-method attack
    with pytest.raises((ValueError, AdapterBlocked), match="binding changed"):
        adapter.run(RunRequest({"prompt": "actual admission"}))
    assert case.model.calls == 0 and case.effects == []


def test_bound_runner_kwargs_mutation_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    model = NativeTransport(responses=[AIMessage(content="unit transport")])

    def bind(
        self: NativeTransport, tools: list[StructuredTool], **kwargs: object
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        return RunnableBinding(
            bound=self, kwargs={"tools": [convert_to_openai_tool(tool) for tool in tools]}
        )

    monkeypatch.setattr(NativeTransport, "bind_tools", bind)
    case = graph_case()
    controls = LangGraphControls()
    node = controls.model_node("model", model, [case.tool])
    runner = cast("RunnableBinding[LanguageModelInput, BaseMessage]", attribute(node, "_runner"))
    kwargs: object = runner.kwargs
    assert isinstance(kwargs, dict)
    cast("dict[str, object]", kwargs)["tool_choice"] = "changed"
    with pytest.raises(ValueError, match="binding changed"):
        node.revalidate()


def test_native_typed_tool_binding_preserves_closed_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = NativeTransport(responses=[AIMessage(content="unit transport")])

    def bind(
        self: NativeTransport, tools: list[StructuredTool], **kwargs: object
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        return self.bind(
            tools=[convert_to_openai_tool(tool) for tool in tools],
            parallel_tool_calls=False,
        )

    monkeypatch.setattr(NativeTransport, "bind_tools", bind)
    case = graph_case()
    controls = LangGraphControls()
    node = controls.model_node("model", model, [case.tool])
    runner = cast("RunnableBinding[LanguageModelInput, BaseMessage]", attribute(node, "_runner"))
    assert type(runner).__name__ == "_ChatModelBinding"
    assert runner.bound is model
    node.revalidate()
    assert runner.invoke("native typed binding").content == "unit transport"
    assert model.calls == 1
    kwargs: object = runner.kwargs
    assert isinstance(kwargs, dict)
    cast("dict[str, object]", kwargs)["parallel_tool_calls"] = True
    with pytest.raises(ValueError, match="binding changed"):
        node.revalidate()
