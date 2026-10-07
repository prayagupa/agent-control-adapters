"""CrewAI configuration of the shared bounded SDK policy worker."""

from __future__ import annotations

from agent_control_adapters._worker import Binding as Binding
from agent_control_adapters._worker import ContextFactory as ContextFactory
from agent_control_adapters._worker import Worker as SharedWorker
from agent_hooks import ApprovalResolver

from .errors import AdapterBlocked, AdapterCleanupError


class Worker(SharedWorker):
    """Preserve CrewAI's positional constructor and HookAborted-compatible errors."""

    def __init__(
        self,
        bindings: dict[int, Binding],
        resolver: ApprovalResolver | None,
        timeout: float,
        emit_timeout: float,
        max_records: int,
    ) -> None:
        super().__init__(
            bindings,
            resolver,
            timeout,
            emit_timeout,
            max_records,
            framework="crewai",
            blocked_type=AdapterBlocked,
            cleanup_type=AdapterCleanupError,
        )
