"""Actual native rejection of null output and silently discarded tool arguments."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from agent_adapter_conformance import make_rego_manifest
from agent_control_adapters import AdapterBlocked, RunRequest
from agent_framework import ChatResponse, Content, Message
from agent_hooks.exceptions import InterceptionBlocked
from maf_adapter_test_helpers import controlled, native_case


@pytest.mark.parametrize("empty_messages", [False, True])
def test_null_model_content_without_tool_calls_is_unsupported(
    empty_messages: bool,
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch)

        async def empty_response(
            *,
            messages: Sequence[Message],
            options: Mapping[str, object],
            stream: bool = False,
            **kwargs: object,
        ) -> ChatResponse[None]:
            case.model.calls += 1
            return ChatResponse(
                messages=[] if empty_messages else [Message(role="assistant", contents=[])],
                response_id="empty-response",
            )

        monkeypatch.setattr(case.agent.client, "_inner_get_response", empty_response)
        adapter = controlled(case, allow_manifest)
        try:
            with pytest.raises(AdapterBlocked) as blocked:
                await adapter.arun(RunRequest({"prompt": "Hello"}))
            assert isinstance(blocked.value.__cause__, ValueError)
            assert case.model.calls == 1
            assert case.effects == []
            assert not any(
                record.interception_point.value == "output" for record in adapter.records
            )
            assert adapter.cleanup_error is None
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("transformed", [False, True])
def test_unknown_tool_argument_fields_never_reach_native_tool(
    transformed: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = make_rego_manifest(
        tmp_path,
        """verdict := {
        "decision": "transform",
        "transform": {
            "path": "$target",
            "value": {"value": "native", "unexpected": "must not be dropped"}
        }
    } if { input.snapshot.interception_point == "pre_tool_call" }"""
        if transformed
        else "",
    )

    async def scenario() -> None:
        case = native_case(monkeypatch, use_tool=True)
        if not transformed:

            async def unknown_arguments(
                *,
                messages: Sequence[Message],
                options: Mapping[str, object],
                stream: bool = False,
                **kwargs: object,
            ) -> ChatResponse[None]:
                case.model.calls += 1
                if case.model.calls > 1:
                    return ChatResponse(
                        messages=[
                            Message(
                                role="assistant",
                                contents=[Content.from_text("native response")],
                            )
                        ],
                        response_id="after-unknown-arguments",
                    )
                return ChatResponse(
                    messages=[
                        Message(
                            role="assistant",
                            contents=[
                                Content.from_function_call(
                                    "unknown-argument-call",
                                    "ping",
                                    arguments={
                                        "value": "native",
                                        "unexpected": "must not be dropped",
                                    },
                                )
                            ],
                        )
                    ],
                    response_id="unknown-arguments",
                )

            monkeypatch.setattr(case.agent.client, "_inner_get_response", unknown_arguments)
        adapter = controlled(case, path)
        try:
            with pytest.raises(AdapterBlocked) as blocked:
                await adapter.arun(RunRequest({"prompt": "Call ping"}))
            assert isinstance(blocked.value.__cause__, (ValueError, InterceptionBlocked))
            assert case.model.calls == 1
            assert case.effects == []
            if transformed:
                assert any(
                    record.interception_point.value == "pre_tool_call" for record in adapter.records
                )
            else:
                assert not any(
                    record.interception_point.value == "pre_tool_call" for record in adapter.records
                )
            assert not any(
                record.interception_point.value in {"post_tool_call", "output"}
                for record in adapter.records
            )
            assert adapter.cleanup_error is None
        finally:
            await case.transport.close()

    asyncio.run(scenario())
