"""Explicit backend dispatch; no model fallback, plugin discovery, or runtime patching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from agent_hooks import ApprovalResolver, Interceptor

from .contracts import Adapter, AgentPolicy, Backend
from .errors import UnsupportedBackendError


def with_agent_hooks(
    *,
    backend: Backend,
    runtime: object,
    interceptors: Sequence[Interceptor],
    agents: Mapping[str, object] | None = None,
    agent_policies: Sequence[AgentPolicy] = (),
    resolver: ApprovalResolver | None = None,
    timeout: float = 5,
    emit_timeout: float = 30,
    max_records: int = 1000,
) -> Adapter:
    """Bind native objects; invalid selectors or missing backends raise UnsupportedBackendError."""
    if backend is not Backend.CREWAI:
        raise UnsupportedBackendError("Unsupported adapter backend; use a Backend enum member.")
    try:
        from agent_hooks_crewai.portable import create_backend
    except ModuleNotFoundError as error:
        if error.name != "agent_hooks_crewai":
            raise
        raise UnsupportedBackendError("Install the CrewAI adapter backend.") from error

    return create_backend(
        runtime,
        interceptors=interceptors,
        agents=agents,
        agent_policies=agent_policies,
        resolver=resolver,
        timeout=timeout,
        emit_timeout=emit_timeout,
        max_records=max_records,
    )


create_adapter = with_agent_hooks
