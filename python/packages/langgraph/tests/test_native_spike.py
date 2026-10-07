"""C4: prove actual runner model replacement and wrapper denial before registration."""

from __future__ import annotations

import asyncio

import pytest
from agent_adapter_conformance import assert_governed_success
from agent_control_adapters import AdapterBlocked, RunRequest
from agent_hooks.exceptions import InterceptionBlocked
from langchain_core.messages import ToolMessage
from langgraph_adapter_test_helpers import POINTS, NativePolicy, graph_adapter, graph_case


def test_actual_native_transforms() -> None:
    case = graph_case()
    adapter = graph_adapter(case, NativePolicy(transform=True))
    result = asyncio.run(adapter.arun(RunRequest({"prompt": "native input"})))
    assert result.output == "transformed output"
    assert case.model.seen[0][0].content == "transformed request"
    assert case.effects == ["transformed arguments"]
    assert any(
        message.type == "ai" and message.content == "transformed model"
        for message in case.model.seen[1]
    )
    assert any(
        isinstance(message, ToolMessage) and message.content == "transformed tool result"
        for message in case.model.seen[1]
    )
    assert_governed_success(result, points=POINTS)


@pytest.mark.parametrize("point", ["pre_model_call", "pre_tool_call"])
def test_actual_native_wrapper_denial(point: str) -> None:
    case = graph_case()
    adapter = graph_adapter(case, NativePolicy(deny=point))
    with pytest.raises(AdapterBlocked) as raised:
        asyncio.run(adapter.arun(RunRequest({"prompt": "native input"})))
    assert isinstance(raised.value.__cause__, InterceptionBlocked)
    assert case.effects == []
    assert case.model.calls == (0 if point == "pre_model_call" else 1)
    assert any(
        record.interception_point.value == point and record.verdict.decision.value == "deny"
        for record in adapter.records
    )
