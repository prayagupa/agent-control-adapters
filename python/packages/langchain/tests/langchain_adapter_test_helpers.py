"""Native create_agent setup paired with the graph-owned deterministic I/O boundary."""

from __future__ import annotations

from dataclasses import dataclass

from agent_control_adapters import AsyncAdapter, NativeRuntime
from agent_control_langchain import LangChainControls
from agent_control_langchain.portable import create_backend
from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
from langchain_core.tools import StructuredTool
from langgraph_adapter_test_helpers import NativePolicy, NativeTransport, model_and_tool


@dataclass
class ChainCase:
    model: NativeTransport
    tool: StructuredTool
    effects: list[str]
    graph: object | None = None
    controls: LangChainControls | None = None

    def build(self, controls: LangChainControls) -> NativeRuntime:
        self.controls = controls
        middleware = controls.middleware("agent")
        self.graph = create_agent(
            model=self.model,
            tools=[self.tool],
            system_prompt="trusted system",
            middleware=[middleware],
        )
        return NativeRuntime(self.graph, {"agent": self.graph})


def chain_case() -> ChainCase:
    model, tool, effects = model_and_tool()
    return ChainCase(model, tool, effects)


def chain_adapter(
    case: ChainCase, policy: NativePolicy, *, max_records: int = 1000
) -> AsyncAdapter:
    return create_backend(
        runtime=case.build,
        interceptors=[policy],
        policy_file=None,
        agents=None,
        agent_policies=(),
        resolver=None,
        timeout=5,
        emit_timeout=30,
        max_records=max_records,
    )
