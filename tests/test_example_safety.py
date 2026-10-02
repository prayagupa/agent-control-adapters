"""Deterministic checks for example I/O safety; no claims about model behavior."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from agent_hooks import (
    AgentContextBuilder,
    ApprovalOutcome,
    ApprovalRequest,
    InterceptionPoint,
    Verdict,
)

from examples.crewai.common import json_object, local_base_url
from examples.crewai.human_approval import ConsoleApproval


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "http://localhost.evil.example/v1",
        "http://user:secret@localhost:11434/v1",
        "http://localhost:11434/v1?route=remote",
        "http://localhost:11434/v1#fragment",
        "http://localhost:11434/",
        "http://localhost:0/v1",
        "http://localhost:11434/v1\n",
    ],
)
def test_examples_refuse_remote_or_ambiguous_endpoints(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    monkeypatch.setenv("LOCAL_MODEL_BASE_URL", url)
    with pytest.raises(ValueError):
        local_base_url()


@pytest.mark.parametrize("value", [[], None, {1: "not a string key"}])
def test_json_object_rejects_invalid_payloads(value: object) -> None:
    with pytest.raises(ValueError):
        json_object(value)


@pytest.mark.parametrize(
    ("reply", "outcome"),
    [
        (b"approve\n", ApprovalOutcome.APPROVE),
        (b"approve\r\n", ApprovalOutcome.APPROVE),
        (b"", ApprovalOutcome.REJECT),
        (b"approve", ApprovalOutcome.REJECT),
        (b"approve\nignore the rules", ApprovalOutcome.REJECT),
        (b"yes\n", ApprovalOutcome.REJECT),
        (b"reject\n", ApprovalOutcome.REJECT),
    ],
)
def test_console_requires_explicit_complete_approval_and_echoes_identity(
    reply: bytes, outcome: ApprovalOutcome
) -> None:
    """Synthetic approval context tests only I/O validation, never model behavior."""
    context = AgentContextBuilder(
        agent_id="unit-test", framework="crewai", session_id="unit-test"
    ).output(content="Synthetic unit-test fixture \x1b[31m")
    request = ApprovalRequest(
        context_identity="jcs-sha256:unit-test",
        interception_point=InterceptionPoint.OUTPUT,
        verdict=Verdict.escalate(reason="test:approval"),
        context=context,
    )
    resolver = ConsoleApproval()
    with (
        patch("examples.crewai.human_approval.select.select", return_value=([0], [], [])),
        patch("examples.crewai.human_approval.sys.stdin") as stream,
        patch("examples.crewai.human_approval.os.read", return_value=reply),
    ):
        stream.fileno.return_value = 0
        result = resolver.resolve(request)
    assert result.outcome is outcome
    assert result.context_identity == request.context_identity


def test_console_timeout_rejects_without_reading() -> None:
    context = AgentContextBuilder(
        agent_id="unit-test", framework="crewai", session_id="unit-test"
    ).output(content="Synthetic fixture")
    request = ApprovalRequest(
        context_identity="jcs-sha256:unit-test",
        interception_point=InterceptionPoint.OUTPUT,
        verdict=Verdict.escalate(reason="test:approval"),
        context=context,
    )
    with (
        patch("examples.crewai.human_approval.select.select", return_value=([], [], [])),
        patch("examples.crewai.human_approval.os.read") as read,
    ):
        result = ConsoleApproval().resolve(request)
    assert result.outcome is ApprovalOutcome.REJECT
    read.assert_not_called()
