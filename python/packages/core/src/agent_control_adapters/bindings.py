"""Validated, immutable host bindings prepared before policy loading or native work."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

from agent_hooks import ApprovalResolver, Interceptor

from ._config import validate_run_options
from .acs import load_acs_policy, normalize_policy_file, validate_acs_interceptor
from .contracts import AgentPolicy, validate_agent_id


@dataclass(frozen=True)
class BoundActor:
    """An actual native actor with host-owned identity, session and ordered controls."""

    label: str
    native: object
    identity: str
    session_id: str
    interceptors: tuple[Interceptor, ...]


@dataclass(frozen=True)
class PreparedControls:
    """Frozen controls whose validation and policy loading have completed."""

    actors: tuple[BoundActor, ...]
    resolver: ApprovalResolver | None
    timeout: float
    emit_timeout: float
    max_records: int


def _interceptors(values: Sequence[Interceptor]) -> tuple[Interceptor, ...]:
    if (
        not isinstance(cast("object", values), Sequence)
        or isinstance(cast("object", values), str | bytes)
        or not 1 <= len(values) <= 64
    ):
        raise ValueError("Provide between 1 and 64 interceptors")
    captured = tuple(values)
    if any(not callable(getattr(value, "intercept", None)) for value in captured):
        raise TypeError("Interceptors must provide a callable intercept method")
    return captured


def prepare_controls(
    native_actors: Mapping[str, object],
    *,
    interceptors: Sequence[Interceptor] | None = None,
    policy_file: str | Path | None = None,
    agents: Mapping[str, object] | None = None,
    agent_policies: Sequence[AgentPolicy] = (),
    resolver: ApprovalResolver | None = None,
    timeout: float = 5,
    emit_timeout: float = 30,
    max_records: int = 1000,
    acs_only: bool = False,
) -> PreparedControls:
    """Validate all sources and actual membership before opening any manifest."""
    if (interceptors is None) == (policy_file is None):
        raise ValueError("Provide exactly one of interceptors or policy_file")
    validate_run_options(timeout, emit_timeout, max_records)
    if not isinstance(cast("object", native_actors), Mapping) or not 1 <= len(native_actors) <= 16:
        raise ValueError("A native runtime must contain between 1 and 16 actors")
    native = dict(native_actors)
    for label, actor in native.items():
        validate_agent_id(label)
        if actor is None or sum(actor is value for value in native.values()) != 1:
            raise ValueError("Native actor membership must be distinct")
    if agents is not None and (not isinstance(cast("object", agents), Mapping) or len(agents) > 16):
        raise TypeError("agents must be a bounded mapping of labels to native actors")
    supplied = dict(agents) if agents is not None else native
    seen: list[object] = []
    for label, actor in supplied.items():
        validate_agent_id(label)
        if not any(actor is member for member in native.values()) or any(
            actor is member for member in seen
        ):
            raise ValueError("Agent bindings must uniquely reference actual native actors")
        seen.append(actor)
    if not isinstance(cast("object", agent_policies), Sequence) or len(agent_policies) > 16:
        raise TypeError("agent_policies must be a bounded sequence")
    assignments: dict[str, AgentPolicy] = {}
    for policy in tuple(agent_policies):
        if not isinstance(cast("object", policy), AgentPolicy):
            raise TypeError("agent_policies must contain portable AgentPolicy objects")
        if policy.agent_id not in supplied or policy.agent_id in assignments:
            raise ValueError("Agent policy must have a unique trusted native binding")
        assignments[policy.agent_id] = policy
    shared = _interceptors(interceptors) if interceptors is not None else ()
    path = normalize_policy_file(policy_file) if policy_file is not None else None
    for policy in assignments.values():
        if policy.interceptors:
            _interceptors(policy.interceptors)
    labeled: list[tuple[str, object]] = [
        (
            next((name for name, member in supplied.items() if member is actor), label),
            actor,
        )
        for label, actor in native.items()
    ]
    if len({label for label, _ in labeled}) != len(labeled):
        raise ValueError("Agent labels must remain unique across the native runtime")
    if acs_only:
        for interceptor in shared:
            validate_acs_interceptor(interceptor)
        for policy in assignments.values():
            for interceptor in policy.interceptors:
                validate_acs_interceptor(interceptor)
    if path is not None:
        shared = (load_acs_policy(path),)
    extras: dict[str, tuple[Interceptor, ...]] = {
        label: (load_acs_policy(policy.policy_file),)
        if policy.policy_file is not None
        else policy.interceptors
        for label, policy in assignments.items()
    }
    actors: list[BoundActor] = []
    for label, actor in labeled:
        actors.append(
            BoundActor(
                label=label,
                native=actor,
                identity=str(uuid4()),
                session_id=str(uuid4()),
                interceptors=shared + extras.get(label, ()),
            )
        )
    return PreparedControls(tuple(actors), resolver, timeout, emit_timeout, max_records)
