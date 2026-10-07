"""Explicit backend dispatch; no model fallback, plugin discovery, or runtime patching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Literal, Protocol, TypedDict, Unpack, cast, overload

from agent_hooks import ApprovalResolver, Interceptor

from .acs import normalize_policy_file
from .contracts import Adapter, AgentPolicy, AsyncAdapter, Backend
from .errors import UnsupportedBackendError


class _BackendConstructor(Protocol):
    def __call__(
        self,
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
    ) -> Adapter: ...


class _ControlOptions(TypedDict, total=False):
    interceptors: Sequence[Interceptor] | None
    policy_file: str | Path | None
    agents: Mapping[str, object] | None
    agent_policies: Sequence[AgentPolicy]
    resolver: ApprovalResolver | None
    timeout: float
    emit_timeout: float
    max_records: int


@overload
def with_agent_controls(
    *,
    backend: Literal[
        Backend.GOOGLE_ADK,
        Backend.LANGCHAIN,
        Backend.LANGGRAPH,
        Backend.OPENAI_AGENTS,
        Backend.MAF,
    ],
    runtime: object,
    **options: Unpack[_ControlOptions],
) -> AsyncAdapter: ...


@overload
def with_agent_controls(
    *, backend: Backend, runtime: object, **options: Unpack[_ControlOptions]
) -> Adapter: ...


def with_agent_controls(
    *,
    backend: Backend,
    runtime: object,
    interceptors: Sequence[Interceptor] | None = None,
    policy_file: str | Path | None = None,
    agents: Mapping[str, object] | None = None,
    agent_policies: Sequence[AgentPolicy] = (),
    resolver: ApprovalResolver | None = None,
    timeout: float = 5,
    emit_timeout: float = 30,
    max_records: int = 1000,
) -> Adapter:
    """Govern a native runtime with Python interceptors or an ACS manifest, never both."""
    if not isinstance(cast("object", backend), Backend):
        raise UnsupportedBackendError("Unsupported adapter backend; use a Backend enum member.")
    if (interceptors is None) == (policy_file is None):
        raise ValueError("Provide exactly one of interceptors or policy_file")
    path = normalize_policy_file(policy_file) if policy_file is not None else None
    namespace = {
        Backend.CREWAI: "agent_control_crewai",
        Backend.GOOGLE_ADK: "agent_control_google_adk",
        Backend.LANGCHAIN: "agent_control_langchain",
        Backend.LANGGRAPH: "agent_control_langgraph",
        Backend.OPENAI_AGENTS: "agent_control_openai_agents",
        Backend.MAF: "agent_control_maf",
    }[backend]
    try:
        module = import_module(f"{namespace}.portable")
    except ModuleNotFoundError as error:
        if error.name != namespace:
            raise
        raise UnsupportedBackendError(f"Install the {backend.value} adapter backend.") from error

    constructor: object = getattr(module, "create_backend", None)
    if not callable(constructor):
        raise TypeError("The selected backend does not provide a callable portable constructor")
    return cast("_BackendConstructor", constructor)(
        runtime=runtime,
        interceptors=interceptors,
        policy_file=path,
        agents=agents,
        agent_policies=agent_policies,
        resolver=resolver,
        timeout=timeout,
        emit_timeout=emit_timeout,
        max_records=max_records,
    )


with_agent_hooks = with_agent_controls
create_adapter = with_agent_controls
