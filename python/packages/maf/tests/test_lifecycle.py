"""Actual native-run bounds, cancellation, audit eviction and caller ownership."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx2
import pytest
from agent_adapter_conformance import make_rego_manifest
from agent_framework import ChatResponse, Content, Message
from agent_hooks_adapters import AdapterBlocked, RunRequest
from agent_hooks_adapters import runtime as runtime_module
from maf_adapter_test_helpers import controlled, native_case, native_tool_pair
from openai import APIConnectionError


def test_caller_client_reused_on_same_loop(
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch)
        loop = asyncio.get_running_loop()
        assert case.agent.client is case.client
        assert case.client.client is case.transport
        try:
            for _ in range(2):
                adapter = controlled(case, allow_manifest)
                assert (await adapter.arun(RunRequest({"prompt": "Hello"}))).output
                assert not case.transport.is_closed()
                assert asyncio.get_running_loop() is loop
                models = await case.transport.models.list()
                assert [model.id for model in models.data] == ["deterministic-model-boundary"]
            assert (await case.agent.run("Caller-owned ungoverned reuse")).text == "native response"
            assert case.model.calls == 3
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("cap", [1, 3, 10])
def test_exact_audit_eviction_and_drop(
    cap: int,
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch, use_tool=True)
        adapter = controlled(case, allow_manifest, max_records=cap)
        try:
            result = await adapter.arun(RunRequest({"prompt": "Call ping"}))
            assert len(result.records) == cap
            assert result.records_dropped == 10 - cap
            assert [record.sequence for record in result.records] == list(range(10 - cap, 10))
            assert result.records == adapter.records
            assert result.records_dropped == adapter.records_dropped
            assert result.records[-1].interception_point.value == "agent_shutdown"
        finally:
            await case.transport.close()

    asyncio.run(scenario())


def test_unknown_sdk_error_not_reclassified(
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch)
        original = APIConnectionError(request=httpx2.Request("POST", "http://localhost:1/v1"))

        async def fail(
            *,
            messages: Sequence[Message],
            options: Mapping[str, object],
            stream: bool = False,
            **kwargs: object,
        ) -> ChatResponse[None]:
            raise original

        monkeypatch.setattr(case.agent.client, "_inner_get_response", fail)
        adapter = controlled(case, allow_manifest)
        try:
            with pytest.raises(Exception) as caught:
                await adapter.arun(RunRequest({"prompt": "Hello"}))
            assert not isinstance(caught.value, AdapterBlocked)
            error: BaseException | None = caught.value
            chain: list[BaseException] = []
            while error is not None and error not in chain:
                chain.append(error)
                error = error.__cause__ or error.__context__
            assert original in chain
            assert adapter.cleanup_error is None
            assert not case.transport.is_closed()
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["deadline", "cancel", "cancel_twice"])
def test_native_deadline_cancel_repeated_cleanup(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    monkeypatch.setattr(runtime_module, "_NATIVE_RUN_TIMEOUT", 0.3)
    monkeypatch.setattr(runtime_module, "_NATIVE_CLEANUP_TIMEOUT", 1.0)

    async def scenario() -> None:
        for _ in range(3):
            entered, stopped = asyncio.Event(), asyncio.Event()
            draining, release = asyncio.Event(), asyncio.Event()
            case = native_case(monkeypatch)

            async def hold(
                *,
                messages: Sequence[Message],
                options: Mapping[str, object],
                stream: bool = False,
                entered: asyncio.Event = entered,
                stopped: asyncio.Event = stopped,
                draining: asyncio.Event = draining,
                release: asyncio.Event = release,
                **kwargs: object,
            ) -> ChatResponse[None]:
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    if kind == "cancel_twice":
                        draining.set()
                        await release.wait()
                    raise
                finally:
                    stopped.set()
                raise AssertionError("Unreachable")

            monkeypatch.setattr(case.agent.client, "_inner_get_response", hold)
            adapter = controlled(case, allow_manifest)
            caller = asyncio.create_task(adapter.arun(RunRequest({"prompt": "Hello"})))
            try:
                await asyncio.wait_for(entered.wait(), timeout=2)
                if kind in {"cancel", "cancel_twice"}:
                    caller.cancel("caller cancellation")
                    if kind == "cancel_twice":
                        await asyncio.wait_for(draining.wait(), timeout=1)
                        caller.cancel("secondary cancellation")
                        release.set()
                    with pytest.raises(asyncio.CancelledError, match="caller cancellation"):
                        await caller
                else:
                    with pytest.raises(AdapterBlocked, match="deadline"):
                        await caller
                assert stopped.is_set()
                assert adapter.cleanup_error is None
                assert not case.transport.is_closed()
                assert not any(
                    task.get_name().startswith("agent-controls-") and not task.done()
                    for task in asyncio.all_tasks()
                )
                assert all(
                    task is asyncio.current_task() or task.done() for task in asyncio.all_tasks()
                )
                assert not any(
                    thread.name.startswith("agent-hooks-") for thread in threading.enumerate()
                )
            finally:
                release.set()
                if not caller.done():
                    caller.cancel()
                    await asyncio.gather(caller, return_exceptions=True)
                await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("mutation", ["instructions", "tool_name_swap"])
def test_native_mutation_before_post_model_prevents_tool_effect(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch, use_tool=True)
        pair = native_tool_pair(case) if mutation == "tool_name_swap" else None

        async def mutate(
            *,
            messages: Sequence[Message],
            options: Mapping[str, object],
            stream: bool = False,
            **kwargs: object,
        ) -> ChatResponse[None]:
            response = await case.model.respond(
                messages=messages,
                options=options,
                stream=stream,
                **kwargs,
            )
            if mutation == "instructions":
                case.agent.default_options["instructions"] = "changed trusted configuration"
            else:
                assert pair is not None
                first, second = pair
                first.name, second.name = second.name, first.name
            return response

        monkeypatch.setattr(case.agent.client, "_inner_get_response", mutate)
        adapter = controlled(case, allow_manifest)
        try:
            with pytest.raises(AdapterBlocked) as blocked:
                await adapter.arun(RunRequest({"prompt": "Call ping"}))
            assert isinstance(blocked.value.__cause__, ValueError)
            assert case.model.calls == 1
            assert case.effects == []
            assert not any(
                record.interception_point.value == "pre_tool_call" for record in adapter.records
            )
            assert adapter.cleanup_error is None
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["image", "multiple_messages"])
def test_non_plain_model_response_rejected_by_actual_native_mapper(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch)

        async def rich_response(
            *,
            messages: Sequence[Message],
            options: Mapping[str, object],
            stream: bool = False,
            **kwargs: object,
        ) -> ChatResponse[None]:
            case.model.calls += 1
            if kind == "image":
                response_messages = [
                    Message(
                        role="assistant",
                        contents=[
                            Content.from_uri(
                                "https://example.invalid/not-fetched.png",
                                media_type="image/png",
                            )
                        ],
                    )
                ]
            else:
                response_messages = [
                    Message(role="assistant", contents=[Content.from_text("first")]),
                    Message(role="assistant", contents=[Content.from_text("second")]),
                ]
            return ChatResponse(messages=response_messages, response_id="non-plain")

        monkeypatch.setattr(case.agent.client, "_inner_get_response", rich_response)
        adapter = controlled(case, allow_manifest)
        try:
            with pytest.raises(AdapterBlocked) as blocked:
                await adapter.arun(RunRequest({"prompt": "Hello"}))
            assert isinstance(blocked.value.__cause__, ValueError)
            assert case.model.calls == 1
            assert case.effects == []
            assert not any(
                record.interception_point.value in {"pre_tool_call", "output"}
                for record in adapter.records
            )
            assert adapter.cleanup_error is None
        finally:
            await case.transport.close()

    asyncio.run(scenario())


def test_one_tool_call_per_model_output(
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch, use_tool=True, tool_calls_per_response=2)
        adapter = controlled(case, allow_manifest)
        try:
            with pytest.raises(AdapterBlocked):
                await adapter.arun(RunRequest({"prompt": "Call ping twice"}))
            assert case.model.calls == 1
            assert case.effects == []
            assert not any(
                record.interception_point.value == "pre_tool_call" for record in adapter.records
            )
        finally:
            await case.transport.close()

    asyncio.run(scenario())


def test_model_call_budget_exact_native_threshold(
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch, use_tool=True, repeat_tools=True)
        adapter = controlled(case, allow_manifest)
        try:
            with pytest.raises(AdapterBlocked):
                await adapter.arun(RunRequest({"prompt": "Keep calling ping"}))
            assert case.model.calls == 10
            assert case.effects == ["native"] * 10
            assert (
                sum(
                    record.interception_point.value == "pre_model_call"
                    for record in adapter.records
                )
                == 10
            )
            assert adapter.cleanup_error is None
        finally:
            await case.transport.close()

    asyncio.run(scenario())


def test_pre_model_changes_request_not_host_instructions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = make_rego_manifest(
        tmp_path,
        """verdict := {
        "decision": "transform",
        "transform": {"path": "$target[0].content", "value": "changed request"}
    } if { input.snapshot.interception_point == "pre_model_call" }""",
    )

    async def scenario() -> None:
        case = native_case(monkeypatch)
        try:
            await controlled(case, path).arun(RunRequest({"prompt": "original"}))
            assert case.model.seen[0][0].text == "changed request"
            assert case.model.seen_options[0]["instructions"] == "Trusted system text"
            assert case.agent.default_options["instructions"] == "Trusted system text"
        finally:
            await case.transport.close()

    asyncio.run(scenario())


def test_non_plain_native_output_transform_blocked_without_rollback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = make_rego_manifest(
        tmp_path,
        """verdict := {
        "decision": "transform",
        "transform": {"path": "$target.content", "value": "unsupported history replacement"}
    } if { input.snapshot.interception_point == "output" }""",
    )

    async def scenario() -> None:
        case = native_case(monkeypatch, use_tool=True)
        try:
            with pytest.raises(AdapterBlocked):
                await controlled(case, path).arun(RunRequest({"prompt": "Call ping"}))
            assert case.effects == ["native"]
            assert case.model.calls == 2
        finally:
            await case.transport.close()

    asyncio.run(scenario())
