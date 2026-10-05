"""Explicit backend dispatch; no model fallback, plugin discovery, or runtime patching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from agent_hooks import ApprovalResolver, Interceptor

from .acs import normalize_policy_file
from .contracts import Adapter, AgentPolicy, Backend
from .errors import UnsupportedBackendError


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
    if backend is not Backend.CREWAI:
        raise UnsupportedBackendError("Unsupported adapter backend; use a Backend enum member.")
    if (interceptors is None) == (policy_file is None):
        raise ValueError("Provide exactly one of interceptors or policy_file")
    path = normalize_policy_file(policy_file) if policy_file is not None else None
    try:
        from agent_hooks_crewai.portable import create_backend
    except ModuleNotFoundError as error:
        if error.name != "agent_hooks_crewai":
            raise
        raise UnsupportedBackendError("Install the CrewAI adapter backend.") from error

    return create_backend(
        runtime,
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
