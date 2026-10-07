"""Safe public errors, without policy or model content."""

from agent_hooks_adapters import AdapterBlocked as CommonAdapterBlocked
from agent_hooks_adapters import AdapterCleanupError as CommonAdapterCleanupError
from crewai.hooks.dispatch import HookAborted


class AdapterBlocked(HookAborted, CommonAdapterBlocked):
    """The adapter denied execution; inspect audit records for policy decisions."""

    def __init__(self) -> None:
        super().__init__("CrewAI execution blocked by governance.", source="agent-hooks-crewai")


class AdapterCleanupError(CommonAdapterCleanupError):
    """A policy worker has not exited within the finite cleanup deadline."""

    def __init__(self) -> None:
        super().__init__("Governance worker cleanup incomplete; process isolation may be required.")
