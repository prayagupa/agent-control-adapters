"""Executable native thresholds, caller-loop liveness, cancellation and owned cleanup."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator

import pytest
from adk_adapter_test_helpers import (
    BoundaryModel,
    Control,
    controlled,
    native_case,
    text_response,
    tool_response,
)
from agent_control_adapters import AdapterBlocked, RunRequest
from agent_control_adapters.runtime import NativeExecutionLimits, RunBudget, validate_topology
from agent_control_google_adk._profile import native_field
from agent_hooks import ALLOW, AgentContext, Verdict
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService


def test_model_call_budget_exhausted_before_eleventh_actual_model_effect() -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        case.model.responses = []
        for index in range(10):
            response = tool_response()
            assert response.content is not None and response.content.parts is not None
            call = response.content.parts[0].function_call
            assert call is not None
            call.id = f"iteration-{index}"
            case.model.responses.append(response)
        adapter = controlled(case)
        with pytest.raises(AdapterBlocked):
            await adapter.arun(RunRequest({"prompt": "repeat tool"}))
        assert len(case.model.seen) == 10
        assert len(case.effects) == 10
        assert adapter.cleanup_error is None

    asyncio.run(scenario())


def test_one_tool_call_per_model_output_denies_before_any_body() -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        case.model.responses[0] = tool_response(calls=2)
        adapter = controlled(case)
        with pytest.raises(AdapterBlocked):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert case.effects == []
        assert len(case.model.seen) == 1
        assert not any(
            record.interception_point.value == "pre_tool_call" for record in adapter.records
        )

    asyncio.run(scenario())


def test_pending_correlation_cap_exact_threshold_and_sticky_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent_control_adapters.runtime._MAX_MODEL_CALLS", 65)
    budget = RunBudget("adk-pending-cap-test")
    for index in range(64):
        budget.before_model(0, f"model-{index}")
        budget.after_model(0, f"model-{index}", [f"tool-{index}"])
    assert budget.snapshot.pending == 64
    with pytest.raises(AdapterBlocked, match="Pending"):
        budget.before_model(0, "model-over-cap")
    budget.close()
    assert budget.snapshot.pending == 0
    with pytest.raises(AdapterBlocked):
        budget.event()


def test_native_event_cap_exact_threshold() -> None:
    budget = RunBudget("adk-event-cap-test")
    for _ in range(256):
        budget.event()
    assert budget.snapshot.native_events == 256
    with pytest.raises(AdapterBlocked, match="event limit"):
        budget.event()
    assert budget.snapshot.native_events == 256


@pytest.mark.parametrize(
    "actors,nodes,edges",
    [
        (0, 1, 0),
        (17, 1, 0),
        (1, 0, 0),
        (1, 17, 0),
        (1, 1, 65),
        (1, 1, -1),
    ],
)
def test_actor_node_edge_caps(actors: int, nodes: int, edges: int) -> None:
    validate_topology(actors=16, nodes=16, edges=64)
    with pytest.raises(ValueError):
        validate_topology(actors=actors, nodes=nodes, edges=edges)


def test_fixed_recursion_and_concurrency() -> None:
    limits = NativeExecutionLimits()
    assert limits.recursion == 30 and limits.tool_concurrency == 1


def test_native_run_deadline_and_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_control_adapters.runtime._NATIVE_RUN_TIMEOUT", 0.05)
    monkeypatch.setattr("agent_control_adapters.runtime._NATIVE_CLEANUP_TIMEOUT", 0.5)

    async def scenario() -> None:
        entered, exited = asyncio.Event(), asyncio.Event()

        async def blocked(
            self: BoundaryModel,
            llm_request: LlmRequest,
            stream: bool = False,
        ) -> AsyncGenerator[LlmResponse, None]:
            entered.set()
            try:
                await asyncio.Event().wait()
                yield text_response()
            finally:
                exited.set()

        monkeypatch.setattr(BoundaryModel, "generate_content_async", blocked)
        adapter = controlled(native_case())
        task = asyncio.create_task(adapter.arun(RunRequest({"prompt": "safe"})))
        await entered.wait()
        with pytest.raises(AdapterBlocked) as caught:
            await task
        assert "deadline" in str(caught.value) or "deadline" in str(caught.value.__cause__)
        assert exited.is_set()
        assert adapter.cleanup_error is None
        assert native_field(adapter, "_shutdown_error") is not None

    asyncio.run(scenario())


def test_cancellation_closes_only_owned_stream_session_and_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted: list[str] = []
    original_delete = InMemorySessionService.delete_session

    async def delete(
        self: InMemorySessionService,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> None:
        deleted.append(session_id)
        await original_delete(self, app_name=app_name, user_id=user_id, session_id=session_id)

    monkeypatch.setattr(InMemorySessionService, "delete_session", delete)

    async def scenario() -> None:
        entered, exited, caller_alive = asyncio.Event(), asyncio.Event(), asyncio.Event()
        caller_service = InMemorySessionService()
        caller_session = await caller_service.create_session(app_name="caller", user_id="caller")
        baseline = set(asyncio.all_tasks())

        async def blocked(
            self: BoundaryModel,
            llm_request: LlmRequest,
            stream: bool = False,
        ) -> AsyncGenerator[LlmResponse, None]:
            entered.set()
            try:
                await caller_alive.wait()
                yield text_response()
            finally:
                exited.set()

        monkeypatch.setattr(BoundaryModel, "generate_content_async", blocked)
        case = native_case()
        adapter = controlled(case)
        task = asyncio.create_task(adapter.arun(RunRequest({"prompt": "safe"})))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exited.is_set()
        assert len(deleted) == 1 and deleted[0] != caller_session.id
        assert (
            await caller_service.get_session(
                app_name="caller",
                user_id="caller",
                session_id=caller_session.id,
            )
            is not None
        )
        assert set(asyncio.all_tasks()) <= baseline
        assert adapter.cleanup_error is None
        caller_alive.set()
        assert not asyncio.get_running_loop().is_closed()

    asyncio.run(scenario())


def test_concurrent_arun_admission_one_effect_and_sync_refusal_does_not_admit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked(
            self: BoundaryModel,
            llm_request: LlmRequest,
            stream: bool = False,
        ) -> AsyncGenerator[LlmResponse, None]:
            self.seen.append(llm_request)
            entered.set()
            await release.wait()
            yield text_response()

        monkeypatch.setattr(BoundaryModel, "generate_content_async", blocked)
        case = native_case()
        adapter = controlled(case)
        task = asyncio.create_task(adapter.arun(RunRequest({"prompt": "safe"})))
        await entered.wait()
        with pytest.raises(RuntimeError, match="single-use"):
            await adapter.arun(RunRequest({"prompt": "race"}))
        with pytest.raises(RuntimeError, match="requires await"):
            adapter.run()
        release.set()
        assert (await task).output == "native answer"
        assert len(case.model.seen) == 1

    asyncio.run(scenario())


def test_async_native_policy_callbacks_leave_application_loop_live() -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        entered, release = threading.Event(), threading.Event()
        ready = asyncio.Event()

        class BlockingControl:
            def intercept(self, context: AgentContext) -> Verdict:
                if context["interception_point"] == "pre_model_call":
                    entered.set()
                    loop.call_soon_threadsafe(ready.set)
                    if not release.wait(2):
                        raise RuntimeError("Test policy barrier timeout")
                return ALLOW

        adapter = controlled(native_case(), interceptors=[BlockingControl()])
        task = asyncio.create_task(adapter.arun(RunRequest({"prompt": "safe"})))
        await ready.wait()
        assert entered.is_set()
        release.set()
        assert (await task).output == "native answer"
        assert adapter.cleanup_error is None

    asyncio.run(scenario())


def test_caller_runtime_and_model_reusable_same_loop_with_new_policy_scope() -> None:
    async def scenario() -> None:
        case = native_case()
        case.model.repeat = True
        first = await controlled(case).arun(RunRequest({"prompt": "first"}))
        second = await controlled(case).arun(RunRequest({"prompt": "second"}))
        assert first.output == second.output
        assert first.records[0].session_id != second.records[0].session_id
        assert len(case.model.seen) == 2
        assert case.agent.mode == "chat"
        assert case.runner.plugin_manager.plugins == []
        assert not asyncio.get_running_loop().is_closed()

    asyncio.run(scenario())


def test_post_tool_deny_does_not_roll_back_completed_effect() -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        adapter = controlled(case, interceptors=[Control("post_tool_call", deny=True)])
        with pytest.raises(AdapterBlocked):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert case.effects == ["native"]
        assert len(case.model.seen) == 1

    asyncio.run(scenario())
