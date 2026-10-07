"""Common error categories; backends retain safe messages and enforcement signals."""


class AdapterBlocked(RuntimeError):
    """Governance denied execution; no successful RunResult is returned."""


class AdapterCleanupError(RuntimeError):
    """Owned governance work did not finish within its cleanup deadline."""


class UnsupportedBackendError(ValueError):
    """The requested backend is not implemented or installed."""


class ACSUnavailableError(ImportError):
    """The optional pinned ACS runtime is missing or incompatible."""
