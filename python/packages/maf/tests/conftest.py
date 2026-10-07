"""Pinned ACS settings and fail-not-skip, loopback-only local-model evidence."""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import NoReturn, cast
from urllib.parse import urlsplit

import httpx2
import pytest
from agent_adapter_conformance import make_rego_manifest

_required = False
_e2e_passed = False
_e2e_skipped = False


def pytest_configure(config: pytest.Config) -> None:
    """Validate required mode and reset this session's model evidence."""
    global _required, _e2e_passed, _e2e_skipped
    value = os.getenv("LOCAL_MODEL_REQUIRED", "0")
    if value not in {"0", "1"}:
        raise pytest.UsageError("LOCAL_MODEL_REQUIRED must be 0 or 1")
    _required = value == "1"
    _e2e_passed = _e2e_skipped = False


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Track skips even when they bypass the local-model fixture."""
    global _e2e_passed, _e2e_skipped
    if "e2e" in report.keywords:
        _e2e_skipped |= report.skipped
        _e2e_passed |= report.when == "call" and report.passed


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Reject required runs with any skipped e2e or no passing e2e."""
    if exitstatus == 0 and _required and (_e2e_skipped or not _e2e_passed):
        session.exitstatus = 1
        warnings.warn(
            pytest.PytestWarning("Required local-model evidence is incomplete"),
            stacklevel=2,
        )


def _unavailable(reason: str) -> NoReturn:
    if _required:
        pytest.fail(f"{reason}; required local-model evidence is missing", pytrace=False)
    pytest.skip(f"{reason}; skipped e2e is missing evidence")


@pytest.fixture(scope="session")
def local_model_settings() -> tuple[str, str]:
    """Probe the real selected model using the pinned OpenAI 3 HTTPX2 transport."""
    base = os.getenv("LOCAL_MODEL_BASE_URL", "http://localhost:11434/v1")
    parsed = urlsplit(base)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"/v1", "/v1/"}
        or parsed.port == 0
        or any(ord(character) <= 32 or ord(character) >= 127 for character in base)
    ):
        raise ValueError("LOCAL_MODEL_BASE_URL must be a credential-free loopback /v1 URL")
    model = os.getenv("LOCAL_MODEL", "qwen2.5:latest")
    if (
        len(model) > 200
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?", model)
        is None
        or any(part in {"", ".", ".."} for part in model.split("/"))
    ):
        raise ValueError("LOCAL_MODEL must name an Ollama model")
    if ":" not in model:
        model += ":latest"
    with httpx2.Client(
        timeout=httpx2.Timeout(5, connect=2), trust_env=False, follow_redirects=False
    ) as client:
        try:
            response = client.get(f"{base.rstrip('/')}/models")
        except (httpx2.ConnectError, httpx2.ConnectTimeout) as error:
            _unavailable(f"Local endpoint unavailable ({type(error).__name__})")
        if response.is_redirect:
            raise ValueError("Local endpoint redirects are not supported")
        response.raise_for_status()
        payload: object = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Local /models response must be an object")
    data = cast("dict[str, object]", payload).get("data")
    if not isinstance(data, list):
        raise ValueError("Local /models response must contain a data list")
    identifiers: list[str] = []
    for entry in cast("list[object]", data):
        if not isinstance(entry, dict):
            raise ValueError("Local model entries must be objects")
        identifier = cast("dict[str, object]", entry).get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Local model entries must contain an id")
        identifiers.append(identifier)
    if model not in identifiers:
        _unavailable(f"Selected model {model} is not listed")
    return base.rstrip("/"), model


@pytest.fixture(autouse=True)
def deterministic_acs_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACS_OPA_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("ACS_REGO_V0", raising=False)


@pytest.fixture
def allow_manifest(tmp_path: Path) -> Path:
    return make_rego_manifest(tmp_path)
