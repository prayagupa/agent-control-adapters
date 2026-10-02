"""Safe public errors, without policy or model content."""

from crewai.hooks.dispatch import HookAborted


class AdapterBlocked(HookAborted):
    """The adapter denied execution; inspect audit records for policy decisions."""

    def __init__(self) -> None:
        super().__init__("CrewAI execution blocked by governance.", source="agent-hooks-crewai")


class AdapterCleanupError(RuntimeError):
    """A policy worker has not exited within the finite cleanup deadline."""

    def __init__(self) -> None:
        super().__init__("Governance worker cleanup incomplete; process isolation may be required.")
