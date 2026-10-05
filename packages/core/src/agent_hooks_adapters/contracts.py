"""Framework-neutral request, output, policy assignment, and execution contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, cast

from agent_hooks import InterceptionRecord, Interceptor

from .errors import AdapterCleanupError
from .json import JSON, object_value


class Backend(StrEnum):
    """Execution backends implemented by the adapter factory."""

    CREWAI = "crewai"


def validate_agent_id(agent_id: object) -> None:
    """Require a bounded application-owned label, never a model-derived identity."""
    if (
        not isinstance(agent_id, str)
        or not agent_id
        or len(agent_id) > 128
        or not agent_id.isascii()
        or any(not (character.isalnum() or character in "_.-") for character in agent_id)
    ):
        raise ValueError("Agent IDs must be 1-128 ASCII alphanumeric/_.- characters")


@dataclass(frozen=True, init=False)
class RunRequest:
    """Own a bounded JSON input snapshot; reads return detached copies."""

    _inputs: dict[str, JSON] | None = field(repr=False)

    def __init__(self, inputs: dict[str, object] | None = None) -> None:
        object.__setattr__(self, "_inputs", object_value(inputs) if inputs is not None else None)

    @property
    def inputs(self) -> dict[str, object] | None:
        """Return a fresh input object without exposing request-owned mutable state."""
        if self._inputs is None:
            return None
        return {key: value for key, value in object_value(self._inputs).items()}


@dataclass(frozen=True)
class RunResult:
    """A successful text output and the actual bounded audit snapshot for that run."""

    output: str
    records: tuple[InterceptionRecord, ...]
    records_dropped: int


@dataclass(frozen=True)
class AgentPolicy:
    """Additional controls selected by a trusted application-owned agent binding."""

    agent_id: str
    interceptors: tuple[Interceptor, ...]

    def __post_init__(self) -> None:
        validate_agent_id(self.agent_id)
        if not isinstance(cast("object", self.interceptors), tuple):
            raise TypeError("AgentPolicy.interceptors must be a tuple")
        if not self.interceptors:
            raise ValueError("AgentPolicy.interceptors must be nonempty")


class Adapter(Protocol):
    """One synchronous, single-use governed run; backend setup stays outside this API."""

    @property
    def records(self) -> tuple[InterceptionRecord, ...]:
        """Actual audit records, also available after denial."""
        ...

    @property
    def records_dropped(self) -> int:
        """Number of older records evicted by the configured audit cap."""
        ...

    @property
    def cleanup_error(self) -> AdapterCleanupError | None:
        """Incomplete cleanup without replacing an existing execution failure."""
        ...

    def run(self, request: RunRequest | None = None) -> RunResult:
        """Execute once; governance denial raises AdapterBlocked instead of a result."""
        ...
