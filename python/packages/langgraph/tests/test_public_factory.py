"""Public enum/async overload dispatch into actual native LangGraph, not a patched registry."""

import asyncio
from pathlib import Path

import pytest
from agent_adapter_conformance import make_rego_manifest
from agent_control_adapters import AsyncAdapter, Backend, RunRequest, with_agent_controls
from graph_adapter_contract_vectors import Mode
from langgraph_adapter_test_helpers import NativePolicy, graph_case


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("source", ["native", "acs"])
def test_registered_public_factory(mode: Mode, source: str, tmp_path: Path) -> None:
    case = graph_case()
    adapter: AsyncAdapter = with_agent_controls(
        backend=Backend.LANGGRAPH,
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
    request = RunRequest({"prompt": "actual public LangGraph"})
    result = adapter.run(request) if mode == "sync" else asyncio.run(adapter.arun(request))
    assert result.output == "native final" and case.effects == ["original"]
    assert case.model.calls == 2 and adapter.cleanup_error is None
    with pytest.raises(RuntimeError, match="single-use"):
        adapter.run(request)
