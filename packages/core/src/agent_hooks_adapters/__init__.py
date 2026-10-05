"""Portable execution API with framework-specific backends and shared SDK policies."""

from .contracts import Adapter, AgentPolicy, Backend, RunRequest, RunResult
from .errors import AdapterBlocked, AdapterCleanupError, UnsupportedBackendError
from .factory import create_adapter, with_agent_hooks

__all__ = [
    "Adapter",
    "AdapterBlocked",
    "AdapterCleanupError",
    "AgentPolicy",
    "Backend",
    "RunRequest",
    "RunResult",
    "UnsupportedBackendError",
    "create_adapter",
    "with_agent_hooks",
]
