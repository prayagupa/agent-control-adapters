"""Required-mode local-model gate, adapted from the local-model-testing fixture."""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Generator
from typing import NoReturn, cast
from urllib.parse import urlsplit

import httpx
import pytest
from openai import OpenAI

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"

_required_for_session = False
_e2e_passed = False
_e2e_skipped = False


def _required_model_tests() -> bool:
    value = os.getenv("LOCAL_MODEL_REQUIRED", "0")
    if value not in {"0", "1"}:
        raise pytest.UsageError("LOCAL_MODEL_REQUIRED must be 0 or 1")
    return value == "1"


def _local_base_url() -> str:
    value = os.getenv("LOCAL_MODEL_BASE_URL", "http://localhost:11434/v1")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"/v1", "/v1/"}
        or parsed.port == 0
        or any(ord(char) <= 32 or ord(char) >= 127 for char in value)
    ):
        raise ValueError("LOCAL_MODEL_BASE_URL must be a loopback /v1 URL")
    return value.rstrip("/")


def _model_name() -> str:
    value = os.getenv("LOCAL_MODEL", "qwen2.5:latest")
    if (
        len(value) > 200
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?", value)
        is None
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("LOCAL_MODEL must be a valid Ollama model identifier")
    return value if ":" in value else f"{value}:latest"


def _unavailable(reason: str) -> NoReturn:
    if _required_for_session:
        pytest.fail(f"{reason}; required local-model evidence is missing", pytrace=False)
    pytest.skip(f"{reason}; skipped e2e is missing evidence")


def pytest_configure(config: pytest.Config) -> None:
    """Reset required-run evidence, including when the whole suite is deselected."""
    global _required_for_session, _e2e_passed, _e2e_skipped
    _required_for_session = _required_model_tests()
    _e2e_passed = False
    _e2e_skipped = False
    config.addinivalue_line("markers", "e2e: requires the local model endpoint")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Count skips even when they occur before local-model fixture setup."""
    global _e2e_passed, _e2e_skipped
    if "e2e" in report.keywords:
        if report.skipped or hasattr(report, "wasxfail"):
            _e2e_skipped = True
        if report.when == "call" and report.passed:
            _e2e_passed = True


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """A required run cannot pass with skipped or absent e2e evidence."""
    if exitstatus == 0 and _required_for_session and (not _e2e_passed or _e2e_skipped):
        session.exitstatus = 1
        warnings.warn(
            pytest.PytestWarning("Required local-model evidence is incomplete"),
            stacklevel=2,
        )


@pytest.fixture(scope="session")
def local_model_name() -> str:
    """The exact model identifier used by examples and the probe."""
    return _model_name()


@pytest.fixture(scope="session")
def local_model_client() -> Generator[OpenAI, None, None]:
    """Probe without proxies/redirects; own and close a no-retry local client."""
    base_url = _local_base_url()
    with httpx.Client(
        timeout=httpx.Timeout(60, connect=3),
        trust_env=False,
        follow_redirects=False,
    ) as transport:
        try:
            response = transport.get(f"{base_url}/models", timeout=5)
        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
            _unavailable(f"Local model endpoint unavailable ({type(error).__name__})")
        response.raise_for_status()
        payload: object = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Local /models must return a JSON object")
        data = cast(dict[object, object], payload).get("data")
        if not isinstance(data, list):
            raise ValueError("Local /models must return a data list")
        identifiers: list[str] = []
        for item in cast(list[object], data):
            if not isinstance(item, dict):
                raise ValueError("Local model entries must be JSON objects")
            identifier = cast(dict[object, object], item).get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("Local model entries need nonempty string IDs")
            identifiers.append(identifier)
        if _model_name() not in identifiers:
            _unavailable(f"Selected local model {_model_name()} is not installed")
        with OpenAI(
            base_url=base_url,
            api_key="ollama",
            http_client=transport,
            max_retries=0,
            timeout=httpx.Timeout(60, connect=3),
        ) as client:
            yield client
