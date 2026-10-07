"""Real native callback enforcement, executable transforms and honest SDK audit."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest
from adk_adapter_test_helpers import (
    POINTS,
    BoundaryModel,
    Control,
    controlled,
    native_case,
    text_response,
    tool_response,
)
from agent_adapter_conformance import assert_governed_success
from agent_hooks import ALLOW, AgentContext, Decision, Verdict
from agent_hooks_adapters import AdapterBlocked, AgentPolicy, RunRequest
from agent_hooks_adapters.json import object_value
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types


def test_all_eight_points_order_actual_records_and_correlations() -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        observer = Control("never", deny=True)
        adapter = controlled(case, interceptors=[observer])
        result = await adapter.arun(RunRequest({"prompt": "safe"}))
        assert_governed_success(result, points=POINTS)
        assert case.effects == ["native"]
        assert [record.interception_point.value for record in result.records] == [
            "agent_startup",
            "input",
            "pre_model_call",
            "post_model_call",
            "pre_tool_call",
            "post_tool_call",
            "pre_model_call",
            "post_model_call",
            "output",
            "agent_shutdown",
        ]
        assert len({record.session_id for record in result.records}) == 1
        assert [record.sequence for record in result.records] == list(range(10))
        contexts = [object_value(context) for context in observer.seen]
        for pre_point, post_point, key in (
            ("pre_model_call", "post_model_call", "model"),
            ("pre_tool_call", "post_tool_call", "tool_call"),
        ):
            pre = [
                context[key] for context in contexts if context["interception_point"] == pre_point
            ]
            post = [
                context[key] for context in contexts if context["interception_point"] == post_point
            ]
            assert pre == post
        for context in contexts:
            agent = object_value(context["agent"])
            assert agent["framework"] == "google-adk"
            assert isinstance(agent["id"], str) and agent["id"]
        assert adapter.cleanup_error is None

    asyncio.run(scenario())


@pytest.mark.parametrize("point", POINTS)
def test_each_real_point_deny_aborts_except_audit_only_shutdown(point: str) -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        adapter = controlled(case, interceptors=[Control(point, deny=True)])
        if point == "agent_shutdown":
            result = await adapter.arun(RunRequest({"prompt": "safe"}))
            assert result.output == "native answer"
        else:
            with pytest.raises(AdapterBlocked):
                await adapter.arun(RunRequest({"prompt": "safe"}))
        assert any(
            record.interception_point.value == point and record.verdict.decision is Decision.DENY
            for record in adapter.records
        )
        assert case.effects == (
            ["native"]
            if point
            in (
                "post_tool_call",
                "output",
                "agent_shutdown",
            )
            else []
        )
        if point in ("agent_startup", "input", "pre_model_call"):
            assert case.model.seen == []
        if point in ("post_model_call", "pre_tool_call"):
            assert len(case.model.seen) == 1
        if point == "post_tool_call":
            assert len(case.model.seen) == 1
        assert adapter.records[-1].interception_point.value == "agent_shutdown"
        assert adapter.cleanup_error is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "point,replacement,expected",
    [
        (
            "input",
            {"content": "actual input replacement", "role": "user"},
            "actual input replacement",
        ),
        ("output", {"content": "actual final replacement"}, "actual final replacement"),
        ("pre_tool_call", {"value": "actual tool argument"}, "actual tool argument"),
        ("post_tool_call", {"result": "actual tool result"}, "actual tool result"),
    ],
)
def test_transforms_reach_actual_native_values(
    point: str, replacement: dict[str, str], expected: str
) -> None:
    async def scenario() -> None:
        case = native_case(use_tool=point in ("pre_tool_call", "post_tool_call"))
        adapter = controlled(case, interceptors=[Control(point, replacement=replacement)])
        result = await adapter.arun(RunRequest({"prompt": "original"}))
        assert any(record.verdict.decision is Decision.TRANSFORM for record in result.records)
        if point == "input":
            assert case.model.seen[0].contents[0].parts is not None
            assert case.model.seen[0].contents[0].parts[0].text == expected
        elif point == "output":
            assert result.output == expected
        elif point == "pre_tool_call":
            assert case.effects == [expected]
        else:
            responses = [
                part.function_response
                for content in case.model.seen[1].contents
                for part in content.parts or []
                if part.function_response is not None
            ]
            assert responses[0].response == {"result": expected}
            assert responses[0].id == "call-0"
            assert case.effects == ["native"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "point,replacement",
    [
        ("agent_startup", {"tools_registered": []}),
        ("input", {"content": "changed", "role": "system"}),
        ("input", {"content": "changed", "role": "user", "agent_id": "spoof"}),
        ("pre_model_call", []),
        ("post_model_call", {"content": "changed", "tool_calls": [], "finish_reason": "spoof"}),
        ("post_model_call", {"content": "changed", "tool_calls": [], "finish_reason": "stop"}),
        ("pre_tool_call", {"value": 42}),
        ("pre_tool_call", {"value": "valid", "unknown": "must not drop"}),
        ("post_tool_call", {"result": float("nan")}),
        ("output", {"content": None}),
        ("output", {"content": "changed", "session_id": "spoof"}),
    ],
)
def test_unappliable_transform_fails_closed_without_fabricating_deny(
    point: str, replacement: object
) -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        adapter = controlled(case, interceptors=[Control(point, replacement=replacement)])
        with pytest.raises(AdapterBlocked):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        actual = next(
            record for record in adapter.records if record.interception_point.value == point
        )
        expected = (
            Decision.DENY if point in ("agent_startup", "post_tool_call") else Decision.TRANSFORM
        )
        assert actual.verdict.decision is expected
        assert case.effects == (["native"] if point in ("post_tool_call", "output") else [])
        assert adapter.cleanup_error is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "arguments",
    [
        {"value": 12},
        {"value": True},
        {"value": None},
        {},
        {"value": "valid", "unknown": "must not drop"},
        {"value": "valid", "agent_id": "spoof"},
    ],
)
def test_raw_original_tool_arguments_rejected_before_native_sanitizer(
    arguments: dict[str, object],
) -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        case.model.responses[0] = tool_response(arguments)
        adapter = controlled(case)
        with pytest.raises(AdapterBlocked):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert case.effects == []
        assert not any(
            record.interception_point.value == "post_model_call" for record in adapter.records
        )
        assert not any(record.verdict.decision is Decision.DENY for record in adapter.records)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "response",
    [
        LlmResponse(),
        LlmResponse(content=types.Content(role="model", parts=[])),
        LlmResponse(content=types.Content(role="model", parts=[types.Part(text="")])),
        LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        inline_data=types.Blob(mime_type="text/plain", data=b"rich"),
                    )
                ],
            )
        ),
        LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="text"),
                    types.Part(
                        function_call=types.FunctionCall(
                            name="ping",
                            id="call-rich",
                            args={"value": "native"},
                        )
                    ),
                ],
            )
        ),
        LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name="ping", args={"value": "native"}),
                    )
                ],
            )
        ),
        LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="text")]),
            input_transcription=types.Transcription(text="rich"),
        ),
    ],
)
def test_null_rich_and_uncorrelated_model_outputs_fail_closed(response: LlmResponse) -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        case.model.responses[0] = response
        adapter = controlled(case)
        with pytest.raises(AdapterBlocked):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert case.effects == []
        assert not any(
            record.interception_point.value == "post_model_call" for record in adapter.records
        )

    asyncio.run(scenario())


def test_shared_controls_precede_per_actor_controls_and_mapping_is_captured() -> None:
    order: list[tuple[str, str]] = []

    class Ordered:
        def __init__(self, label: str) -> None:
            self.label = label

        def intercept(self, context: AgentContext) -> Verdict:
            order.append((self.label, context["interception_point"]))
            return ALLOW

    async def scenario() -> None:
        case = native_case()
        bindings: dict[str, object] = {"application_label": case.agent}
        adapter = controlled(
            case,
            interceptors=[Ordered("shared")],
            agents=bindings,
            agent_policies=[AgentPolicy("application_label", interceptors=(Ordered("actor"),))],
        )
        bindings.clear()
        result = await adapter.arun(RunRequest({"prompt": "safe"}))
        assert order == [
            (label, record.interception_point.value)
            for record in result.records
            for label in ("shared", "actor")
        ]
        assert all(record.interceptors_registered == 2 for record in result.records)

    asyncio.run(scenario())


def test_actual_transformed_response_event_preserves_usage_version_ids_and_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Event] = []
    original_delete = InMemorySessionService.delete_session

    async def observe_delete(
        self: InMemorySessionService, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        session = await self.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
        assert session is not None
        events.extend(session.events)
        await original_delete(self, app_name=app_name, user_id=user_id, session_id=session_id)

    monkeypatch.setattr(InMemorySessionService, "delete_session", observe_delete)

    async def scenario() -> None:
        case = native_case()
        adapter = controlled(
            case,
            interceptors=[
                Control(
                    "post_model_call",
                    replacement={
                        "content": "actual transformed native response",
                        "tool_calls": [],
                        "finish_reason": "stop",
                    },
                )
            ],
        )
        result = await adapter.arun(RunRequest({"prompt": "safe"}))
        actual = next(
            event
            for event in events
            if event.author == "trusted_actor" and event.usage_metadata is not None
        )
        assert actual.id
        assert actual.invocation_id
        assert actual.model_version == "native-version"
        assert actual.usage_metadata is not None
        assert actual.usage_metadata.total_token_count == 5
        assert actual.content is not None and actual.content.parts is not None
        assert actual.content.parts[0].text == result.output

    asyncio.run(scenario())


def test_native_transport_failure_is_not_misclassified(monkeypatch: pytest.MonkeyPatch) -> None:
    class TransportFault(RuntimeError):
        pass

    fault = TransportFault("actual transport fault")

    async def fail(
        self: BoundaryModel, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        raise fault
        yield text_response()  # pragma: no cover

    monkeypatch.setattr(BoundaryModel, "generate_content_async", fail)

    async def scenario() -> None:
        adapter = controlled(native_case())
        with pytest.raises(TransportFault) as caught:
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert caught.value is fault
        assert not any(record.verdict.decision is Decision.DENY for record in adapter.records)
        assert adapter.cleanup_error is None

    asyncio.run(scenario())


def test_exact_bounded_real_record_retention() -> None:
    async def scenario() -> None:
        adapter = controlled(native_case(use_tool=True), max_records=3)
        result = await adapter.arun(RunRequest({"prompt": "safe"}))
        assert len(result.records) == 3
        assert result.records_dropped == 7
        assert [record.sequence for record in result.records] == [7, 8, 9]

    asyncio.run(scenario())


def test_governance_boundary_logs_exclude_payload_and_private_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        adapter = controlled(
            native_case(use_tool=True),
            interceptors=[
                Control("pre_tool_call", deny=True),
            ],
        )
        with pytest.raises(AdapterBlocked) as caught:
            await adapter.arun(RunRequest({"prompt": "private-prompt-marker"}))
        assert caught.value.__cause__ is not None
        assert "private-prompt-marker" not in caplog.text
        assert "private policy test detail" not in caplog.text

    asyncio.run(scenario())
