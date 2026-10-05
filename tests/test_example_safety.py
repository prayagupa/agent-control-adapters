"""Deterministic example safety checks; synthetic responses are only unit-test fixtures."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Generator
from pathlib import Path
from typing import cast
from unittest.mock import patch

import httpx
import pytest
from agent_hooks import (
    AgentContextBuilder,
    ApprovalOutcome,
    ApprovalRequest,
    InterceptionPoint,
    Verdict,
)
from crewai.llms.providers.openai.completion import OpenAICompletion

from examples.crewai import human_approval, llm_hooks, multi_agent, tool_hooks
from examples.crewai.human_approval import ConsoleApproval

ROOT: Path = Path(__file__).resolve().parents[1]
EXAMPLE_NAMES: tuple[str, ...] = ("llm_hooks", "tool_hooks", "human_approval", "multi_agent")
EXAMPLE_MAINS: tuple[Callable[[], None], ...] = (
    llm_hooks.main,
    tool_hooks.main,
    human_approval.main,
    multi_agent.main,
)
JSON_VALIDATORS: tuple[Callable[[object], dict[str, object]], ...] = (
    llm_hooks.json_object,
    tool_hooks.json_object,
    human_approval.json_object,
    multi_agent.json_object,
)


@pytest.mark.parametrize("validate_json", JSON_VALIDATORS, ids=EXAMPLE_NAMES)
@pytest.mark.parametrize("value", [[], None, {1: "not a string key"}])
def test_json_object_rejects_invalid_payloads(
    value: object, validate_json: Callable[[object], dict[str, object]]
) -> None:
    """Policies and approval must reject malformed external objects."""
    with pytest.raises(ValueError):
        validate_json(value)


@pytest.mark.parametrize("validate_json", JSON_VALIDATORS, ids=EXAMPLE_NAMES)
def test_json_object_preserves_valid_payloads(
    validate_json: Callable[[object], dict[str, object]],
) -> None:
    """Preserve valid JSON instead of rewriting trusted identity or content."""
    value: dict[str, object] = {"content": "Synthetic fixture"}
    assert validate_json(value) is value


@pytest.mark.parametrize(
    ("example", "max_lines"),
    [("llm_hooks", 100), ("tool_hooks", 105), ("human_approval", 160), ("multi_agent", 150)],
)
def test_example_source_is_simple_and_self_contained(example: str, max_lines: int) -> None:
    """Keep complete examples bounded in size and free of test-evidence machinery."""
    text = (ROOT / "examples" / "crewai" / f"{example}.py").read_text()
    assert len(text.splitlines()) <= max_lines
    tree = ast.parse(text)
    definitions = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef | ast.FunctionDef)
    }
    assert "main" in definitions
    assert not {"LocalTextLLM", "local_llm", "ModelRun", "report", "local_base_url"} & definitions
    assert "/api/tags" not in text and "/api/version" not in text
    constructors = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"OpenAICompletion", "Agent", "Crew", "with_agent_hooks", "RunRequest"} <= constructors
    assert not {"CrewAIAdapter", "create_adapter"} & constructors
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "with_agent_hooks"
        ):
            backend = next((value.value for value in node.keywords if value.arg == "backend"), None)
            assert isinstance(backend, ast.Attribute)
            assert isinstance(backend.value, ast.Name)
            assert backend.value.id == "Backend" and backend.attr == "CREWAI"
    methods = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "run" in methods and "kickoff" not in methods
    assert not {"make_agent", "make_crew"} & constructors
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                alias.name.split(".")[0] not in {"examples", "tests"} for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert node.module is not None and node.module.split(".")[0] not in {
                "examples",
                "tests",
            }


@pytest.fixture
def unit_model_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[list[httpx.Request], None, None]:
    """Stub only HTTP responses; run real native clients, crews and policies."""
    requests: list[httpx.Request] = []
    transports: set[httpx.HTTPTransport] = set()

    def handle_request(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://localhost:11434/v1/chat/completions"
        timeouts = human_approval.json_object(request.extensions["timeout"])
        assert all(timeouts[phase] == 60 for phase in ("connect", "read", "write", "pool"))
        requests.append(request)
        transports.add(transport)
        parameters = human_approval.json_object(json.loads(request.content))
        model = parameters["model"]
        assert isinstance(model, str)
        message: dict[str, object] = {
            "role": "assistant",
            "content": "Final Answer: Synthetic unit-test response. Identifier 000-00-0000.",
        }
        finish_reason = "stop"
        native_tools = parameters.get("tools")
        if native_tools:
            assert isinstance(native_tools, list)
            schema = human_approval.json_object(cast(list[object], native_tools)[0])
            function = human_approval.json_object(schema["function"])
            name = function["name"]
            assert name in {"delete_file", "send_email"}
            messages = parameters["messages"]
            assert isinstance(messages, list)
            if not any(
                human_approval.json_object(value).get("role") == "tool"
                for value in cast(list[object], messages)
            ):
                arguments = (
                    {"path": "demo.txt"}
                    if name == "delete_file"
                    else {"message": "Synthetic unit-test preview."}
                )
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "unit-test-call",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
                finish_reason = "tool_calls"
        return httpx.Response(
            200,
            json={
                "id": "unit-test",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            },
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    try:
        yield requests
    finally:
        for transport in transports:
            transport.close()


@pytest.mark.parametrize(
    ("example", "run_example"), zip(EXAMPLE_NAMES, EXAMPLE_MAINS, strict=True), ids=EXAMPLE_NAMES
)
@pytest.mark.parametrize(
    ("model_env", "expected_model"),
    [
        (None, "qwen2.5:latest"),
        ("qwen2.5", "qwen2.5:latest"),
        ("qwen2.5:unit-test", "qwen2.5:unit-test"),
    ],
)
def test_example_uses_native_llm_and_preserves_governance(
    example: str,
    run_example: Callable[[], None],
    model_env: str | None,
    expected_model: str,
    monkeypatch: pytest.MonkeyPatch,
    unit_model_requests: list[httpx.Request],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The actual examples use bounded native settings and real policy wiring."""
    if model_env is None:
        monkeypatch.delenv("LOCAL_MODEL", raising=False)
    else:
        monkeypatch.setenv("LOCAL_MODEL", model_env)
    monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "https://remote.invalid/v1")
    with (
        patch(f"examples.crewai.{example}.OpenAICompletion", wraps=OpenAICompletion) as factory,
        patch("examples.crewai.human_approval.select.select", return_value=([0], [], [])),
        patch("examples.crewai.human_approval.sys.stdin") as stream,
        patch("examples.crewai.human_approval.os.read", return_value=b"approve\n"),
    ):
        stream.fileno.return_value = 0
        run_example()
    factory.assert_called_once_with(
        model=expected_model,
        custom_openai=True,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0,
        max_tokens=256,
        timeout=60,
        max_retries=0,
    )
    assert len(unit_model_requests) == (2 if example in {"multi_agent", "human_approval"} else 1)
    for request in unit_model_requests:
        parameters = human_approval.json_object(json.loads(request.content))
        assert parameters["model"] == expected_model
        assert parameters["temperature"] == 0
        assert parameters["max_tokens"] == 256
        assert bool(parameters.get("tools")) is (example in {"tool_hooks", "human_approval"})
    output = capsys.readouterr().out
    if example == "tool_hooks":
        assert "Blocked: delete_file did not execute." in output
        assert "Synthetic unit-test response." not in output
    else:
        assert "Synthetic unit-test response." in output
        if example == "human_approval":
            assert "Approved preview (not sent):" in output
            assert "Answer after approval:" in output
        elif example == "llm_hooks":
            assert "[REDACTED]" in output and "000-00-0000" not in output
        elif example == "multi_agent":
            assert "Per-agent policies stayed isolated." in output


@pytest.mark.parametrize(
    ("limit", "size", "denied"),
    [(800, 800, False), (800, 801, True), (1600, 1600, False), (1600, 1601, True)],
)
def test_agent_text_limit_enforces_exact_boundary(limit: int, size: int, denied: bool) -> None:
    """Per-agent limits retain the exact inclusive threshold and host identity."""
    context = AgentContextBuilder(
        agent_id="unit-test", framework="crewai", session_id="unit-test"
    ).post_model_call(
        model_id="synthetic-unit-test", content="x" * size, tool_calls=[], finish_reason="stop"
    )
    policy = multi_agent.AgentTextLimit(max_characters=limit)
    verdict = policy.intercept(context)
    assert (verdict.decision.value == "deny") is denied
    assert policy.seen_agent_ids == {"unit-test"}


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
    reply: bytes, outcome: ApprovalOutcome, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validate terminal input and escaping using synthetic unit-test context only."""
    context = AgentContextBuilder(
        agent_id="unit-test", framework="crewai", session_id="unit-test"
    ).pre_tool_call(
        call_id="unit-test-call",
        name="send_email",
        args={"message": "Synthetic unit-test fixture \x1b[31m"},
    )
    request = ApprovalRequest(
        context_identity="jcs-sha256:unit-test",
        interception_point=InterceptionPoint.PRE_TOOL_CALL,
        verdict=Verdict.escalate(reason="test:approval"),
        context=context,
    )
    with (
        patch("examples.crewai.human_approval.select.select", return_value=([0], [], [])),
        patch("examples.crewai.human_approval.sys.stdin") as stream,
        patch("examples.crewai.human_approval.os.read", return_value=reply),
    ):
        stream.fileno.return_value = 0
        result = ConsoleApproval().resolve(request)
    assert result.outcome is outcome
    assert result.context_identity == request.context_identity
    displayed = capsys.readouterr().out
    assert "\x1b" not in displayed
    assert "\\u001b" in displayed


def test_console_timeout_rejects_without_reading() -> None:
    """Timeout must reject without consuming future terminal input."""
    context = AgentContextBuilder(
        agent_id="unit-test", framework="crewai", session_id="unit-test"
    ).pre_tool_call(
        call_id="unit-test-call", name="send_email", args={"message": "Synthetic fixture"}
    )
    request = ApprovalRequest(
        context_identity="jcs-sha256:unit-test",
        interception_point=InterceptionPoint.PRE_TOOL_CALL,
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


@pytest.mark.parametrize("identifier", ["000-00-0000", "123-45-6789"])
def test_model_policy_transforms_only_the_response_content(identifier: str) -> None:
    context = AgentContextBuilder(
        agent_id="unit-test", framework="crewai", session_id="unit-test"
    ).post_model_call(
        model_id="synthetic-unit-test",
        content=f"Test identifier {identifier}.",
        tool_calls=[],
        finish_reason="stop",
    )
    verdict = llm_hooks.RedactIdentifiers().intercept(context)
    assert verdict.decision.value == "transform"
    assert verdict.transform is not None
    assert verdict.transform.to_wire() == {
        "path": "$target.content",
        "value": "Test identifier [REDACTED].",
    }
    assert context["target"]["content"] == f"Test identifier {identifier}."


@pytest.mark.parametrize("content", [None, 1, [], {}])
def test_model_policy_rejects_non_text_responses(content: object) -> None:
    context = AgentContextBuilder(
        agent_id="unit-test", framework="crewai", session_id="unit-test"
    ).post_model_call(
        model_id="synthetic-unit-test", content=content, tool_calls=[], finish_reason="stop"
    )
    assert llm_hooks.RedactIdentifiers().intercept(context).decision.value == "deny"


@pytest.mark.parametrize(
    ("name", "denied"),
    [("delete_file", True), ("drop_table", True), ("system_shutdown", True), ("web_search", False)],
)
def test_tool_policy_uses_host_tool_name_not_forged_arguments(name: str, denied: bool) -> None:
    context = AgentContextBuilder(
        agent_id="unit-test", framework="crewai", session_id="unit-test"
    ).pre_tool_call(
        call_id="unit-test-call", name=name, args={"name": "web_search", "path": "demo.txt"}
    )
    verdict = tool_hooks.BlockDestructiveTools().intercept(context)
    assert (verdict.decision.value == "deny") is denied


@pytest.mark.parametrize(("name", "liftable"), [("send_email", True), ("delete_file", False)])
def test_approval_policy_escalates_only_the_expected_tool(name: str, liftable: bool) -> None:
    context = AgentContextBuilder(
        agent_id="unit-test", framework="crewai", session_id="unit-test"
    ).pre_tool_call(call_id="unit-test-call", name=name, args={"message": "Synthetic fixture"})
    verdict = human_approval.ReviewBeforeSend().intercept(context)
    assert verdict.decision.value == "deny"
    assert verdict.is_liftable is liftable


@pytest.mark.parametrize("reply", [b"reject\n", b""])
def test_rejected_tool_example_never_runs_its_body(
    reply: bytes,
    unit_model_requests: list[httpx.Request],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch("examples.crewai.human_approval.select.select", return_value=([0], [], [])),
        patch("examples.crewai.human_approval.sys.stdin") as stream,
        patch("examples.crewai.human_approval.os.read", return_value=reply),
    ):
        stream.fileno.return_value = 0
        human_approval.main()
    output = capsys.readouterr().out
    assert "Blocked: send_email did not execute." in output
    assert "Approved preview (not sent):" not in output
    assert len(unit_model_requests) == 1
