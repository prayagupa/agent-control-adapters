"""Compile trusted application policy bindings into the existing CrewAI adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from agent_hooks import ApprovalResolver, Interceptor
from agent_hooks_adapters import AgentPolicy
from agent_hooks_adapters.contracts import validate_agent_id
from crewai import Agent, Crew

from .adapter import AgentPolicy as NativeAgentPolicy
from .adapter import CrewAIAdapter


def create_backend(
    runtime: object,
    *,
    interceptors: Sequence[Interceptor],
    agents: Mapping[str, object] | None = None,
    agent_policies: Sequence[AgentPolicy] = (),
    resolver: ApprovalResolver | None = None,
    timeout: float = 5,
    emit_timeout: float = 30,
    max_records: int = 1000,
) -> CrewAIAdapter:
    """Validate trusted aliases and freeze controls without changing native identities."""
    if not isinstance(runtime, Crew):
        raise TypeError("The CrewAI backend requires a Crew runtime")
    if agents is not None and not isinstance(cast("object", agents), Mapping):
        raise TypeError("agents must be a mapping of application IDs to native agents")
    bindings: dict[str, Agent] = {}
    for agent_id, native_agent in (dict(agents) if agents is not None else {}).items():
        validate_agent_id(agent_id)
        if not isinstance(native_agent, Agent) or not any(
            native_agent is member for member in runtime.agents
        ):
            raise ValueError("Agent bindings must reference actual agents in the crew")
        if any(native_agent is member for member in bindings.values()):
            raise ValueError("Each native agent must have one application binding")
        bindings[agent_id] = native_agent
    policies: list[NativeAgentPolicy] = []
    seen: set[str] = set()
    for policy in tuple(agent_policies):
        if not isinstance(cast("object", policy), AgentPolicy):
            raise TypeError("agent_policies must contain portable AgentPolicy objects")
        native_agent = bindings.get(policy.agent_id)
        if native_agent is None:
            raise ValueError("Agent policy has no trusted native binding")
        if policy.agent_id in seen:
            raise ValueError("Each application agent must have one policy assignment")
        seen.add(policy.agent_id)
        policies.append(NativeAgentPolicy(native_agent, policy.interceptors))
    return CrewAIAdapter(
        runtime,
        interceptors=tuple(interceptors),
        agent_policies=tuple(policies),
        resolver=resolver,
        timeout=timeout,
        emit_timeout=emit_timeout,
        max_records=max_records,
    )
