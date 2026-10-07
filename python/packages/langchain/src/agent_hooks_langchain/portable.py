"""Portable constructor for one application-owned native create_agent graph."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from agent_hooks import ApprovalResolver, Interceptor
from agent_hooks_adapters import AgentPolicy, AsyncAdapter, NativeRuntime
from agent_hooks_adapters.bindings import prepare_controls
from agent_hooks_langgraph._routing import Route
from agent_hooks_langgraph._scope import GraphAdapter, NativeScope

from ._profile import profile
from .controls import LangChainControls


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
    """Validate inactive native wrappers and actual membership before policy file I/O."""
    if (
        not callable(runtime)
        or inspect.iscoroutinefunction(runtime)
        or inspect.isasyncgenfunction(runtime)
    ):
        raise TypeError("LangChain runtime must be a synchronous application-owned builder")
    controller = LangChainControls()
    native: object = cast("Callable[[LangChainControls], object]", runtime)(controller)
    controller.seal()
    if inspect.iscoroutine(native):
        native.close()
        raise TypeError("LangChain builders must return NativeRuntime synchronously")
    middleware = controller.middleware_instance
    if (
        not isinstance(native, NativeRuntime)
        or middleware is None
        or controller.construction_violation
        or set(native.agents) != {middleware.label}
        or native.agents[middleware.label] is not native.runtime
    ):
        raise ValueError(
            "NativeRuntime must bind the one exact middleware-owned compiled graph actor"
        )
    if agents is not None and (
        set(agents) != set(native.agents)
        or any(agents[label] is not actor for label, actor in native.agents.items())
    ):
        raise ValueError("Caller agents must exactly match the captured native graph mapping")
    captured = profile(native.runtime, middleware)
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
    scope = NativeScope(
        "langchain",
        prepared,
        captured.proof,
        captured.graph,
        ((captured.graph, captured.tools),),
        (Route(captured.graph, captured.graph),) if captured.tools else (),
        captured.revalidate,
    )
    controller.attach(scope, captured)
    return GraphAdapter(scope, before_agent=True)
