"""Trusted manifest paths and lazy loading of the pinned native ACS interceptor."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from os import fspath
from pathlib import Path
from typing import Protocol, cast

from agent_hooks import Interceptor

from .errors import ACSUnavailableError

ACS_VERSION = "0.4.0a4"
_INSTALL_MESSAGE = "Install 'agent-hooks-adapters[acs]' to use ACS policy files."


class _ACSModule(Protocol):
    """Typed boundary for the published ACS package, which ships no typing marker."""

    def AcsInterceptor(self, manifest_path: str) -> Interceptor: ...


def normalize_policy_file(value: object) -> Path:
    """Own an absolute local path; reject invalid types, empty strings and NULs."""
    if not isinstance(value, str | Path):
        raise TypeError("policy_file must be a string or pathlib.Path")
    text = fspath(value)
    if not text or "\x00" in text:
        raise ValueError("policy_file must be a nonempty path without NUL characters")
    return Path(text).absolute()


def _acs_module() -> _ACSModule:
    try:
        installed = version("agent-control-spec")
    except PackageNotFoundError as error:
        raise ACSUnavailableError(_INSTALL_MESSAGE) from error
    if installed != ACS_VERSION:
        raise ACSUnavailableError(
            f"ACS policy files require agent-control-spec=={ACS_VERSION}; found {installed}. "
            + _INSTALL_MESSAGE
        )
    try:
        module = import_module("agent_control_spec")
    except ModuleNotFoundError as error:
        if error.name != "agent_control_spec":
            raise
        raise ACSUnavailableError(_INSTALL_MESSAGE) from error
    return cast(_ACSModule, module)


def load_acs_policy(policy_file: str | Path) -> Interceptor:
    """Load a native interceptor; missing SDKs fail explicitly and engine errors propagate."""
    path = normalize_policy_file(policy_file)
    return _acs_module().AcsInterceptor(str(path))


def validate_acs_interceptor(interceptor: Interceptor) -> None:
    """Require the pinned native ACS class for a backend that accepts ACS only."""
    module = _acs_module()
    native_type: object = module.AcsInterceptor
    if not isinstance(native_type, type) or type(interceptor) is not native_type:
        raise TypeError("This backend requires native ACS interceptors or policy_file")
