"""Portable constructor with the common keywords, independent of enum registration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from agent_hooks import ApprovalResolver, Interceptor
from agent_hooks_adapters import AgentPolicy, AsyncAdapter
from agent_hooks_adapters.bindings import prepare_controls

from ._profile import profile
from .adapter import OpenAIAgentsAdapter


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
    """Validate native profile/ownership before constructing native or ACS policies."""
    native = profile(runtime)
    controls = prepare_controls(
        {"agent": native.agent},
        interceptors=interceptors,
        policy_file=policy_file,
        agents=agents,
        agent_policies=agent_policies,
        resolver=resolver,
        timeout=timeout,
        emit_timeout=emit_timeout,
        max_records=max_records,
    )
    return OpenAIAgentsAdapter(native, controls)
