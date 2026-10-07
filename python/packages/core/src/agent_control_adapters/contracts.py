"""Framework-neutral request, output, policy assignment, and execution contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast

from agent_hooks import InterceptionRecord, Interceptor

from .acs import normalize_policy_file
from .errors import AdapterCleanupError
from .json import JSON, object_value


class Backend(StrEnum):
    """Execution backends implemented by the adapter factory."""

    CREWAI = "crewai"
    GOOGLE_ADK = "google-adk"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    OPENAI_AGENTS = "openai-agents"
    MAF = "maf"


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
    """Additional controls selected by a trusted application-owned agent binding.

    A supplied policy_file is captured as an absolute Path, without resolving symlinks.
    """

    agent_id: str
    interceptors: tuple[Interceptor, ...] = ()
    policy_file: str | Path | None = None

    def __post_init__(self) -> None:
        validate_agent_id(self.agent_id)
        if not isinstance(cast("object", self.interceptors), tuple):
            raise TypeError("AgentPolicy.interceptors must be a tuple")
        if bool(self.interceptors) == (self.policy_file is not None):
            raise ValueError("AgentPolicy requires either nonempty interceptors or policy_file")
        if self.policy_file is not None:
            object.__setattr__(self, "policy_file", normalize_policy_file(self.policy_file))


@dataclass(frozen=True, init=False)
class NativeRuntime:
    """A builder's native runtime and detached, trusted actor membership.

    Native objects remain application-owned; only the label mapping is copied.
    """

    runtime: object
    _agents: Mapping[str, object] = field(repr=False)

    def __init__(self, runtime: object, agents: Mapping[str, object]) -> None:
        if runtime is None:
            raise ValueError("A native runtime is required")
        if not isinstance(cast("object", agents), Mapping):
            raise TypeError("agents must be a mapping of labels to native objects")
        if not 1 <= len(agents) <= 16:
            raise ValueError("Native runtime must contain between 1 and 16 actors")
        owned: dict[str, object] = {}
        for label, native in agents.items():
            validate_agent_id(label)
            if native is None or any(native is actor for actor in owned.values()):
                raise ValueError("Native runtime requires distinct native actor objects")
            owned[label] = native
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "_agents", MappingProxyType(owned))

    @property
    def agents(self) -> Mapping[str, object]:
        """Expose read-only labels without copying or disguising native identities."""
        return self._agents


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


class AsyncAdapter(Adapter, Protocol):
    """A single-use adapter supporting execution on the caller's running event loop."""

    async def arun(self, request: RunRequest | None = None) -> RunResult:
        """Execute once without owning or closing caller clients or their event loop."""
        ...
