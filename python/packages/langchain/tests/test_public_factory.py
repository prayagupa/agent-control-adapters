"""Actual public LangChain dispatch exercises registered enum and typed async overload."""

import asyncio
from pathlib import Path

import pytest
from agent_adapter_conformance import make_rego_manifest
from agent_control_adapters import (
    AsyncAdapter,
    Backend,
    NativeRuntime,
    RunRequest,
    with_agent_controls,
)
from agent_control_langchain import LangChainControls
from graph_adapter_contract_vectors import Mode
from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
from langchain_adapter_test_helpers import chain_case
from langgraph_adapter_test_helpers import NativePolicy, model_and_tool


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("source", ["native", "acs"])
def test_registered_public_factory(mode: Mode, source: str, tmp_path: Path) -> None:
    case = chain_case()
    adapter: AsyncAdapter = with_agent_controls(
        backend=Backend.LANGCHAIN,
        runtime=case.build,
        interceptors=[NativePolicy()] if source == "native" else None,
        policy_file=make_rego_manifest(tmp_path) if source == "acs" else None,
        agents=None,
        agent_policies=(),
        resolver=None,
        timeout=5,
        emit_timeout=30,
        max_records=1000,
    )
    request = RunRequest({"prompt": "actual public LangChain"})
    result = adapter.run(request) if mode == "sync" else asyncio.run(adapter.arun(request))
    assert result.output == "native final" and case.effects == ["original"]
    assert case.model.calls == 2 and adapter.cleanup_error is None
    with pytest.raises(RuntimeError, match="single-use"):
        adapter.run(request)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("source", ["native", "acs"])
def test_tool_free_public_factory_actual_empty_tool_node(
    mode: Mode, source: str, tmp_path: Path
) -> None:
    """Pinned create_agent installs a wrapped empty ToolNode; prove its real closed routing."""
    model, _, effects = model_and_tool(use_tool=False)

    def build(controls: LangChainControls) -> NativeRuntime:
        graph = create_agent(
            model,
            tools=[],
            system_prompt="trusted system",
            middleware=[controls.middleware("agent")],
        )
        return NativeRuntime(graph, {"agent": graph})

    adapter: AsyncAdapter = with_agent_controls(
        backend=Backend.LANGCHAIN,
        runtime=build,
        interceptors=[NativePolicy()] if source == "native" else None,
        policy_file=make_rego_manifest(tmp_path) if source == "acs" else None,
    )
    request = RunRequest({"prompt": "actual public tool-free LangChain"})
    result = adapter.run(request) if mode == "sync" else asyncio.run(adapter.arun(request))
    assert result.output == "native final"
    assert model.calls == 1 and effects == []
    assert [record.interception_point.value for record in result.records] == [
        "agent_startup",
        "input",
        "pre_model_call",
        "post_model_call",
        "output",
        "agent_shutdown",
    ]
    assert adapter.cleanup_error is None
