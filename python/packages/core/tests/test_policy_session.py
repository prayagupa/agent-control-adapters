"""The shared execution scope uses the actual bounded SDK emitter and records."""

from __future__ import annotations

import asyncio
import threading

import agent_control_adapters._worker as worker_module
import pytest
from agent_control_adapters import AdapterBlocked, AdapterCleanupError
from agent_control_adapters.bindings import prepare_controls
from agent_control_adapters.codecs import input_text, unchanged
from agent_control_adapters.json import JSON, object_value
from agent_control_adapters.runtime import PolicySession
from agent_hooks import (
    ALLOW,
    AgentContext,
    AgentContextBuilder,
    Decision,
    InterceptionPoint,
    Verdict,
)


class Allow:
    def __init__(self) -> None:
        self.contexts: list[dict[str, JSON]] = []

    def intercept(self, context: AgentContext) -> Verdict:
        self.contexts.append(object_value(context))
        return ALLOW


class DenyInput:
    def intercept(self, context: AgentContext) -> Verdict:
        if context["interception_point"] == "input":
            return Verdict(decision=Decision.DENY, reason="private operator policy detail")
        return ALLOW


def test_session_is_lazy_and_matches_actual_native_objects() -> None:
    actor = object()
    session = PolicySession("test", prepare_controls({"trusted": actor}, interceptors=[Allow()]))
    assert session.records == ()
    assert session.records_dropped == 0
    assert session.binding(actor) == 0
    with pytest.raises(AdapterBlocked, match="Unknown"):
        session.binding(object())
    assert session.policy_failed


def test_sync_session_records_real_sdk_sequence_and_shared_actor_order() -> None:
    actor = object()
    session = PolicySession("test", prepare_controls({"agent": actor}, interceptors=[Allow()]))
    session.begin()
    with pytest.raises(RuntimeError, match="single-use"):
        session.begin()
    try:
        assert (
            session.emit(
                0,
                lambda builder: builder.input(content="native"),
                input_text,
            )
            == "native"
        )
        session.emit(
            0,
            lambda builder: builder.agent_shutdown(reason="completed"),
            audit=True,
        )
    finally:
        session.close()
    assert [record.interception_point for record in session.records] == [
        InterceptionPoint.INPUT,
        InterceptionPoint.AGENT_SHUTDOWN,
    ]
    assert [record.sequence for record in session.records] == [0, 1]
    assert session.records[0].session_id == session.controls.actors[0].session_id
    assert session.cleanup_error is None
    with pytest.raises(AdapterBlocked, match="not active"):
        session.emit(0, lambda builder: builder.input(content="later"))


def test_async_session_preserves_original_denial_and_only_audits_after_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        actor = object()
        session = PolicySession(
            "test", prepare_controls({"agent": actor}, interceptors=[DenyInput()])
        )
        session.begin()
        primary: BaseException | None = None
        try:
            with pytest.raises(AdapterBlocked) as denied:
                await session.aemit(0, lambda builder: builder.input(content="native"), input_text)
            primary = denied.value
            assert session.policy_failed
            assert session.last_error is not None
            with pytest.raises(AdapterBlocked):
                await session.aemit(0, lambda builder: builder.input(content="later"))
            await session.aemit(
                0, lambda builder: builder.agent_shutdown(reason="failed"), audit=True
            )
        finally:
            await session.aclose(primary)
        assert len(session.records) == 2
        assert session.records[0].verdict.decision is Decision.DENY
        assert session.records[1].interception_point is InterceptionPoint.AGENT_SHUTDOWN
        assert session.cleanup_error is None

    asyncio.run(scenario())
    assert "private operator policy detail" not in caplog.text


def test_unsupported_transform_poison_preserves_native_error_category() -> None:
    actor = object()
    session = PolicySession("test", prepare_controls({"agent": actor}, interceptors=[Allow()]))
    session.begin()
    error = ValueError("original native SDK failure")
    session.fail(error)
    assert not session.policy_failed
    assert session.last_error is error
    with pytest.raises(AdapterBlocked):
        session.emit(0, lambda builder: builder.input(content="later"))
    session.close(error)


def test_startup_notification_uses_actual_sdk_target() -> None:
    actor = object()
    session = PolicySession("test", prepare_controls({"agent": actor}, interceptors=[Allow()]))
    session.begin()
    try:
        original = {"tools_registered": ["ping"]}
        session.emit(
            0,
            lambda builder: builder.agent_startup(tools_registered=["ping"]),
            lambda target: unchanged(target, original),
        )
    finally:
        session.close()
    assert session.records[0].interception_point is InterceptionPoint.AGENT_STARTUP


def test_native_failure_poison_is_stronger_than_an_earlier_policy_denial() -> None:
    actor = object()
    session = PolicySession("test", prepare_controls({"agent": actor}, interceptors=[DenyInput()]))
    session.begin()
    try:
        with pytest.raises(AdapterBlocked):
            session.emit(0, lambda builder: builder.input(content="native"))
        session.fail(asyncio.CancelledError())
        with pytest.raises(AdapterBlocked):
            session.emit(0, lambda builder: builder.agent_shutdown(reason="cancelled"), audit=True)
    finally:
        session.close()
    assert len(session.records) == 1


def test_distinct_sessions_share_one_host_trace_across_all_actor_contexts() -> None:
    async def scenario() -> None:
        policy = Allow()
        actors = {"first": object(), "second": object()}
        sessions = [
            PolicySession("test", prepare_controls(actors, interceptors=[policy])) for _ in range(2)
        ]
        for session in sessions:
            session.begin()
        try:
            await asyncio.gather(
                *(
                    session.aemit(
                        binding,
                        lambda builder: builder.input(content='{"trace_id":"untrusted-payload"}'),
                        input_text,
                    )
                    for session in sessions
                    for binding in range(2)
                )
            )
        finally:
            await asyncio.gather(*(session.aclose() for session in sessions))
        traces = {session.trace_id for session in sessions}
        assert len(traces) == 2
        assert all(session.budget.trace_id == session.trace_id for session in sessions)
        assert len(policy.contexts) == 4
        assert all(
            context["trace"] in [{"trace_id": trace} for trace in traces]
            for context in policy.contexts
        )
        for session in sessions:
            identities = {actor.identity for actor in session.controls.actors}
            matching = [
                context
                for context in policy.contexts
                if object_value(context["agent"])["id"] in identities
            ]
            assert len(matching) == 2
            assert all(context["trace"] == {"trace_id": session.trace_id} for context in matching)
            assert session.cleanup_error is None
        assert not any(thread.name.startswith("agent-hooks-") for thread in threading.enumerate())

    asyncio.run(scenario())


def test_worker_and_budget_failure_logs_use_the_same_host_trace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = PolicySession(
        "test", prepare_controls({"agent": object()}, interceptors=[DenyInput()])
    )
    session.begin()
    try:
        with pytest.raises(AdapterBlocked):
            session.emit(0, lambda builder: builder.input(content="native"))
        with pytest.raises(AdapterBlocked):
            for _ in range(257):
                session.budget.event()
    finally:
        session.close()
    failures = [
        record
        for record in caplog.records
        if record.message in {"adapter_emit_failed", "adapter_budget_denied"}
    ]
    assert {record.message for record in failures} == {
        "adapter_emit_failed",
        "adapter_budget_denied",
    }
    assert all(record.__dict__["trace_id"] == session.trace_id for record in failures)


def test_actual_worker_cleanup_failure_retains_the_host_trace(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(worker_module, "_CLEANUP_TIMEOUT", 0.05)

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        entered, exited = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        session = PolicySession(
            "test", prepare_controls({"agent": object()}, interceptors=[Allow()])
        )
        session.begin()

        def factory(builder: AgentContextBuilder) -> AgentContext:
            loop.call_soon_threadsafe(entered.set)
            try:
                assert release.wait(3)
                return builder.input(content="native")
            finally:
                loop.call_soon_threadsafe(exited.set)

        emission = asyncio.create_task(session.aemit(0, factory))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            with pytest.raises(AdapterCleanupError):
                await session.aclose()
            assert session.cleanup_error is not None
            failures = [
                record
                for record in caplog.records
                if record.name == worker_module.__name__ and record.levelname == "ERROR"
            ]
            assert failures
            assert all(record.__dict__["trace_id"] == session.trace_id for record in failures)
        finally:
            release.set()
            await asyncio.wait_for(exited.wait(), 1)
            await session.aclose()
            with pytest.raises(AdapterBlocked):
                await emission
        assert not any(thread.name.startswith("agent-hooks-") for thread in threading.enumerate())

    asyncio.run(scenario())
