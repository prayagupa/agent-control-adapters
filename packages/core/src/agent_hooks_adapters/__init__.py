"""Portable execution API for native Agent Hooks policies and ACS manifests."""

from .contracts import Adapter, AgentPolicy, Backend, RunRequest, RunResult
from .errors import (
    ACSUnavailableError,
    AdapterBlocked,
    AdapterCleanupError,
    UnsupportedBackendError,
)
from .factory import create_adapter, with_agent_controls, with_agent_hooks

__all__ = [
    "ACSUnavailableError",
    "Adapter",
    "AdapterBlocked",
    "AdapterCleanupError",
    "AgentPolicy",
    "Backend",
    "RunRequest",
    "RunResult",
    "UnsupportedBackendError",
    "create_adapter",
    "with_agent_controls",
    "with_agent_hooks",
]
