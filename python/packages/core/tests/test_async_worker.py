"""Real-SDK conformance for shared FIFO admission, cancellation, and cleanup."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, cast
from uuid import uuid4

import agent_hooks_adapters._worker as worker_module
import pytest
from agent_hooks import (
    ALLOW,
    AgentContext,
    AgentContextBuilder,
    Decision,
    EmitOutcome,
    InterceptionEmitter,
    InterceptionRecord,
    Interceptor,
    Transform,
    Verdict,
)
from agent_hooks_adapters._worker import Binding, Worker, _Job, _Notification
from agent_hooks_adapters.errors import AdapterBlocked, AdapterCleanupError
from agent_hooks_adapters.json import JSON, normalize, object_value

_IDENTITY = str(uuid4())
_SESSION = str(uuid4())


class AllowPolicy:
    def __init__(self) -> None:
        self.contexts: list[dict[str, JSON]] = []

    def intercept(self, context: AgentContext) -> Verdict:
        self.contexts.append(object_value(context))
        return ALLOW


class GatePolicy:
    def __init__(self, *, resist_cancel: bool = False) -> None:
        self.caller_loop = asyncio.get_running_loop()
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.exited = asyncio.Event()
        self.sdk_loop: asyncio.AbstractEventLoop | None = None
        self.gate: asyncio.Event | None = None
        self.contexts: list[dict[str, JSON]] = []
        self.resist_cancel = resist_cancel

    async def intercept(self, context: AgentContext) -> Verdict:
        self.contexts.append(object_value(context))
        if len(self.contexts) > 1:
            return ALLOW
        self.sdk_loop = asyncio.get_running_loop()
        self.gate = asyncio.Event()
        self.caller_loop.call_soon_threadsafe(self.entered.set)
        try:
            await self.gate.wait()
        except asyncio.CancelledError:
            self.caller_loop.call_soon_threadsafe(self.cancelled.set)
            if self.resist_cancel:
                await self.gate.wait()
            raise
        finally:
            self.caller_loop.call_soon_threadsafe(self.exited.set)
        return ALLOW

    def release(self) -> None:
        assert self.sdk_loop is not None and self.gate is not None
        if not self.sdk_loop.is_closed():
            self.sdk_loop.call_soon_threadsafe(self.gate.set)


class AsyncPolicy(Protocol):
    async def intercept(self, context: AgentContext) -> Verdict: ...


def make_worker(
    policy: Interceptor | AsyncPolicy,
    *,
    emit_timeout: float = 2,
    max_records: int = 32,
    trace_id: str | None = None,
) -> Worker:
    # SDK 0.1.0a5 awaits these policies, but its public protocol describes only sync returns.
    return Worker(
        {0: Binding(_IDENTITY, (cast("Interceptor", policy),), _SESSION)},
        None,
        5,
        emit_timeout,
        max_records,
        framework="test",
        trace_id=trace_id,
    )


def builder() -> AgentContextBuilder:
    return AgentContextBuilder(agent_id=_IDENTITY, framework="test", session_id=_SESSION)


def sync_emit(worker: Worker, label: str) -> object:
    return worker.emit(0, lambda context: context.input(content=label))


async def async_emit(worker: Worker, label: str) -> object:
    return await worker.aemit(0, lambda context: context.input(content=label))


async def checkpoint() -> None:
    reached = asyncio.Event()
    asyncio.get_running_loop().call_soon(reached.set)
    await reached.wait()


@pytest.mark.parametrize("trace_id", [None, "host-owned-run"])
def test_worker_context_trace_preserves_legacy_default(trace_id: str | None) -> None:
    policy = AllowPolicy()
    worker = make_worker(policy, trace_id=trace_id)
    try:
        sync_emit(worker, "input")
    finally:
        worker.close()
    expected = trace_id if trace_id is not None else _IDENTITY.replace("-", "")
    assert policy.contexts[0]["trace"] == {"trace_id": expected}


@pytest.mark.parametrize("trace_id", ["", " ", " trace", "trace ", "trace\nid", "x" * 129])
def test_supplied_worker_trace_rejected_before_thread_start(trace_id: str) -> None:
    before = set(threading.enumerate())
    with pytest.raises(ValueError, match="trace ID"):
        make_worker(AllowPolicy(), trace_id=trace_id)
    assert set(threading.enumerate()) == before


@pytest.mark.parametrize("unchecked", [False, True])
def test_external_context_uses_host_trace_without_mutating_caller_context(unchecked: bool) -> None:
    async def scenario() -> None:
        policy = AllowPolicy()
        worker = make_worker(policy, trace_id="host-owned-run")
        context = (
            builder()
            .with_l2(trace={"trace_id": "upstream-run", "span_id": "native-span"})
            .input(content="native")
        )
        original = object_value(context)
        try:
            if unchecked:
                await worker.aemit_unchecked(0, context)
            else:
                await worker.aemit_outcome(0, context)
        finally:
            await worker.aclose()
        assert policy.contexts[0]["trace"] == {
            "trace_id": "host-owned-run",
            "span_id": "native-span",
        }
        assert object_value(context) == original
        assert len(worker.records) == 1

    asyncio.run(scenario())


def test_external_malformed_trace_is_rejected_before_policy_io() -> None:
    async def scenario() -> None:
        policy = AllowPolicy()
        worker = make_worker(policy, trace_id="host-owned-run")
        context = builder().input(content="native")
        context["trace"] = "malformed"
        try:
            with pytest.raises(AdapterBlocked):
                await worker.aemit_outcome(0, context)
            assert isinstance(worker.last_error, ValueError)
            assert policy.contexts == []
            assert worker.records == ()
        finally:
            await worker.aclose()

    asyncio.run(scenario())


def watch_admission(worker: Worker, monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    changed = asyncio.Event()
    loop = asyncio.get_running_loop()
    original = worker._admit_locked

    def admit(job: _Job) -> None:
        original(job)
        loop.call_soon_threadsafe(changed.set)

    monkeypatch.setattr(worker, "_admit_locked", admit)
    return changed


async def wait_callers(worker: Worker, count: int, changed: asyncio.Event) -> None:
    while True:
        with worker._condition:
            if len(worker._callers) == count:
                return
        changed.clear()
        await asyncio.wait_for(changed.wait(), 2)


def test_async_policy_wait_leaves_caller_loop_live_without_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        policy = GatePolicy()
        worker = make_worker(policy)
        caller_loop = asyncio.get_running_loop()

        def forbidden(*args: object, **kwargs: object) -> object:
            raise AssertionError("Async policy submission must not use an executor.")

        monkeypatch.setattr(asyncio, "to_thread", forbidden)
        monkeypatch.setattr(caller_loop, "run_in_executor", forbidden)
        task = asyncio.create_task(async_emit(worker, "live"))
        try:
            await asyncio.wait_for(policy.entered.wait(), 1)
            await checkpoint()
            assert not task.done()
            assert policy.sdk_loop is not caller_loop
            policy.release()
            assert await task == {"content": "live", "role": "user"}
        finally:
            policy.release()
            await worker.aclose()
        await checkpoint()
        assert not caller_loop.is_closed()
        assert policy.sdk_loop is not None and policy.sdk_loop.is_closed()
        assert not worker._thread.is_alive()
        assert worker._loop is None and not worker._callers

    asyncio.run(scenario())


def test_nine_tasks_on_one_loop_are_distinct_and_fifo() -> None:
    async def scenario() -> None:
        policy = GatePolicy()
        worker = make_worker(policy)
        first = asyncio.create_task(async_emit(worker, "0"))
        try:
            await asyncio.wait_for(policy.entered.wait(), 1)
            waiting = [asyncio.create_task(async_emit(worker, str(index))) for index in range(1, 9)]
            await checkpoint()
            with worker._condition:
                assert len(worker._callers) == 9
                assert len(worker._queue) == 8 and worker._job is not None
            policy.release()
            results = await asyncio.gather(first, *waiting)
            assert results == [{"content": str(index), "role": "user"} for index in range(9)]
            assert [context["sequence"] for context in policy.contexts] == list(range(9))
            assert [record.sequence for record in worker.records] == list(range(9))
            assert not worker.denied
        finally:
            policy.release()
            await worker.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("tenth_async", [False, True])
def test_mixed_sync_async_exact_capacity_tenth_poisons(
    monkeypatch: pytest.MonkeyPatch, tenth_async: bool
) -> None:
    async def scenario() -> None:
        policy = GatePolicy(resist_cancel=True)
        worker = make_worker(policy)
        changed = watch_admission(worker, monkeypatch)
        with ThreadPoolExecutor(max_workers=5) as pool:
            first = asyncio.create_task(async_emit(worker, "first"))
            await asyncio.wait_for(policy.entered.wait(), 1)
            sync = [pool.submit(sync_emit, worker, f"sync-{index}") for index in range(4)]
            asynchronous = [
                asyncio.create_task(async_emit(worker, f"async-{index}")) for index in range(4)
            ]
            try:
                await wait_callers(worker, 9, changed)
                with worker._condition:
                    assert len(worker._queue) == 8
                with pytest.raises(AdapterBlocked):
                    if tenth_async:
                        await async_emit(worker, "tenth")
                    else:
                        await asyncio.wrap_future(pool.submit(sync_emit, worker, "tenth"))
                await asyncio.wait_for(policy.cancelled.wait(), 1)
                with worker._condition:
                    assert worker._job is not None
                    assert len(worker._callers) == 1
                    assert not worker._queue
                assert len(policy.contexts) == 1 and worker.denied
                policy.release()
                results = await asyncio.gather(
                    first,
                    *asynchronous,
                    *(asyncio.wrap_future(future) for future in sync),
                    return_exceptions=True,
                )
                assert all(isinstance(result, AdapterBlocked) for result in results)
            finally:
                policy.release()
                await worker.aclose()
        assert not worker._callers and worker._job is None

    asyncio.run(scenario())


def test_fifo_order_between_sync_and_async_waiters(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        policy = GatePolicy()
        worker = make_worker(policy)
        changed = watch_admission(worker, monkeypatch)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = asyncio.create_task(async_emit(worker, "async-0"))
            await asyncio.wait_for(policy.entered.wait(), 1)
            second = pool.submit(sync_emit, worker, "sync-1")
            await wait_callers(worker, 2, changed)
            third = asyncio.create_task(async_emit(worker, "async-2"))
            await wait_callers(worker, 3, changed)
            fourth = pool.submit(sync_emit, worker, "sync-3")
            await wait_callers(worker, 4, changed)
            fifth = asyncio.create_task(async_emit(worker, "async-4"))
            await wait_callers(worker, 5, changed)
            try:
                policy.release()
                results = await asyncio.gather(
                    first, asyncio.wrap_future(second), third, asyncio.wrap_future(fourth), fifth
                )
                assert results == [
                    {"content": label, "role": "user"}
                    for label in ["async-0", "sync-1", "async-2", "sync-3", "async-4"]
                ]
                assert [context["target"] for context in policy.contexts] == results
            finally:
                policy.release()
                await worker.aclose()

    asyncio.run(scenario())


def test_queued_cancellation_releases_only_its_lease() -> None:
    async def scenario() -> None:
        policy = GatePolicy()
        worker = make_worker(policy)
        first = asyncio.create_task(async_emit(worker, "first"))
        await asyncio.wait_for(policy.entered.wait(), 1)
        waiting = [asyncio.create_task(async_emit(worker, str(index))) for index in range(8)]
        try:
            await checkpoint()
            waiting[3].cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting[3]
            with worker._condition:
                assert len(worker._callers) == 8 and len(worker._queue) == 7
                assert worker._job is not None
            assert worker.available and not worker.denied
            replacement = asyncio.create_task(async_emit(worker, "replacement"))
            await checkpoint()
            with worker._condition:
                assert len(worker._callers) == 9
            policy.release()
            results = await asyncio.gather(
                first, *(task for task in waiting if task is not waiting[3]), replacement
            )
            assert len(results) == 9
            assert [record.sequence for record in worker.records] == list(range(9))
            assert policy.contexts[-1]["target"] == {"content": "replacement", "role": "user"}
        finally:
            policy.release()
            await worker.aclose()

    asyncio.run(scenario())


def test_queued_unchecked_rejected_by_host_deny_is_blocked_not_a_missing_record() -> None:
    async def scenario() -> None:
        policy = GatePolicy()
        worker = make_worker(policy)
        first = asyncio.create_task(async_emit(worker, "active"))
        await asyncio.wait_for(policy.entered.wait(), 1)
        queued = asyncio.create_task(worker.aemit_unchecked(0, builder().input(content="queued")))
        try:
            await checkpoint()
            with worker._condition:
                assert len(worker._callers) == 2 and len(worker._queue) == 1
            worker.deny()
            with pytest.raises(AdapterBlocked):
                await queued
            with worker._condition:
                assert len(worker._callers) == 1 and worker._job is not None
            assert worker.available
            policy.release()
            with pytest.raises(AdapterBlocked):
                await first
            assert len(policy.contexts) == 1 and len(worker.records) == 1
        finally:
            policy.release()
            await worker.aclose()

    asyncio.run(scenario())


def test_queued_wall_timeout_poisons_and_retains_real_active_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        policy = GatePolicy(resist_cancel=True)
        worker = make_worker(policy)
        first = asyncio.create_task(async_emit(worker, "active"))
        await asyncio.wait_for(policy.entered.wait(), 1)
        monkeypatch.setattr(worker, "_deadline", 0.05)
        try:
            with pytest.raises(AdapterBlocked):
                await async_emit(worker, "queued-timeout")
            await asyncio.wait_for(policy.cancelled.wait(), 1)
            with worker._condition:
                assert worker._job is not None and len(worker._callers) == 1
                assert not worker._queue
            assert len(policy.contexts) == 1 and isinstance(worker.last_error, TimeoutError)
            with pytest.raises(AdapterBlocked):
                await first
        finally:
            monkeypatch.setattr(worker, "_deadline", 2)
            policy.release()
            await worker.aclose()
        assert not worker._callers and not worker._thread.is_alive()

    asyncio.run(scenario())


def test_active_cancel_retains_job_until_sdk_cancellation_actually_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_module, "_CLEANUP_TIMEOUT", 0.05)

    async def scenario() -> None:
        policy = GatePolicy(resist_cancel=True)
        worker = make_worker(policy)
        task = asyncio.create_task(async_emit(worker, "active"))
        try:
            await asyncio.wait_for(policy.entered.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(policy.cancelled.wait(), 1)
            with worker._condition:
                assert worker._job is not None and len(worker._callers) == 1
            with pytest.raises(AdapterBlocked):
                await async_emit(worker, "not-admitted")
            with pytest.raises(AdapterCleanupError):
                await worker.aclose()
            assert worker._thread.is_alive()
        finally:
            policy.release()
            await asyncio.wait_for(policy.exited.wait(), 1)
            await worker.aclose()
        assert worker._job is None and not worker._callers
        assert len(worker.records) == 1 and not worker.records[0].proceeds
        assert isinstance(worker.last_error, asyncio.CancelledError)

    asyncio.run(scenario())


def test_public_poison_preserves_first_error_and_retains_actual_active_lease() -> None:
    async def scenario() -> None:
        policy = GatePolicy(resist_cancel=True)
        worker = make_worker(policy)
        first_error = RuntimeError("native boundary failed")
        active = asyncio.create_task(async_emit(worker, "active"))
        await asyncio.wait_for(policy.entered.wait(), 1)
        queued = asyncio.create_task(async_emit(worker, "queued"))
        try:
            await checkpoint()
            worker.poison(first_error)
            worker.poison(ValueError("later failure"))
            worker.poison()
            assert worker.denied and not worker.available
            assert worker.last_error is first_error
            with worker._condition:
                assert worker._job is not None and len(worker._callers) == 1
                assert not worker._queue
            with pytest.raises(AdapterBlocked) as caught:
                await queued
            assert caught.value.__cause__ is first_error
            with pytest.raises(AdapterBlocked):
                await active
            await asyncio.wait_for(policy.cancelled.wait(), 1)
            with worker._condition:
                assert worker._job is not None and len(worker._callers) == 1
            with pytest.raises(AdapterBlocked):
                await async_emit(worker, "after-poison")
            assert len(policy.contexts) == 1
        finally:
            policy.release()
            await worker.aclose()
        assert worker.last_error is first_error
        assert worker._job is None and not worker._callers
        assert not worker._thread.is_alive()

    asyncio.run(scenario())


@pytest.mark.parametrize("poison_instead", [False, True])
def test_completion_racing_cancel_or_sticky_denial_never_succeeds(
    monkeypatch: pytest.MonkeyPatch, poison_instead: bool
) -> None:
    async def scenario() -> None:
        worker = make_worker(AllowPolicy())
        loop = asyncio.get_running_loop()
        loop_errors: list[dict[str, object]] = []
        loop.set_exception_handler(lambda owner, context: loop_errors.append(context))
        original = worker._wake
        task: asyncio.Task[object]

        def wake(notification: _Notification) -> None:
            if poison_instead:
                worker.deny()
            else:
                task.cancel()
                notification.future.cancel()
            original(notification)

        monkeypatch.setattr(worker, "_wake", wake)
        task = asyncio.create_task(async_emit(worker, "race"))
        try:
            with pytest.raises(AdapterBlocked if poison_instead else asyncio.CancelledError):
                await task
            await checkpoint()
            assert worker.denied and not loop_errors
            assert worker._job is None and not worker._callers
        finally:
            monkeypatch.setattr(worker, "_wake", original)
            await worker.aclose()

    asyncio.run(scenario())


def test_closed_caller_loop_poison_releases_queue_but_retains_active_job() -> None:
    entered, release = threading.Event(), threading.Event()
    worker = make_worker(AllowPolicy())

    def blocking_factory(context: AgentContextBuilder) -> AgentContext:
        entered.set()
        assert release.wait(3)
        return context.input(content="active")

    caller_loop = asyncio.new_event_loop()
    queued = _Job(
        0,
        lambda context: context.input(content="queued"),
        normalize,
        False,
        object(),
        notification=_Notification(caller_loop, caller_loop.create_future()),
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(worker.emit, 0, blocking_factory)
        assert entered.wait(1)
        try:
            with worker._condition:
                worker._admit_locked(queued)
            caller_loop.close()
            worker.deny()
            with worker._condition:
                assert worker._poisoned
                assert worker._job is not None
                assert len(worker._callers) == 1 and not worker._queue
                assert queued.done.is_set()
            release.set()
            with pytest.raises(AdapterBlocked):
                first.result(timeout=2)
        finally:
            release.set()
            worker.close()
            if not caller_loop.is_closed():
                caller_loop.close()
    assert isinstance(worker.last_error, RuntimeError)
    assert not worker._thread.is_alive()


@pytest.mark.parametrize("stage", ["factory", "validator"])
@pytest.mark.parametrize("cancel", [False, True])
def test_nonpreemptible_factory_validator_retains_capacity(
    monkeypatch: pytest.MonkeyPatch, stage: str, cancel: bool
) -> None:
    monkeypatch.setattr(worker_module, "_CLEANUP_TIMEOUT", 0.05)

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        entered, exited = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        policy = AllowPolicy()
        worker = make_worker(policy, emit_timeout=2 if cancel else 0.1)

        def block(value: object) -> object:
            loop.call_soon_threadsafe(entered.set)
            try:
                assert release.wait(3)
                return value
            finally:
                loop.call_soon_threadsafe(exited.set)

        def factory(context: AgentContextBuilder) -> AgentContext:
            if stage == "factory":
                block(None)
            return context.input(content="blocked")

        task = asyncio.create_task(
            worker.aemit(0, factory, block if stage == "validator" else normalize)
        )
        try:
            await asyncio.wait_for(entered.wait(), 1)
            if cancel:
                task.cancel()
            with pytest.raises(asyncio.CancelledError if cancel else AdapterBlocked):
                await task
            with worker._condition:
                assert worker._job is not None and len(worker._callers) == 1
            await checkpoint()
            with pytest.raises(AdapterBlocked):
                await async_emit(worker, "later")
            with pytest.raises(AdapterCleanupError):
                await worker.aclose()
            assert worker._thread.is_alive() and not exited.is_set()
        finally:
            release.set()
            await asyncio.wait_for(exited.wait(), 1)
            await worker.aclose()
        assert not worker._thread.is_alive() and not worker._callers
        assert len(policy.contexts) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("_attempt", range(3))
def test_repeated_wall_timeout_cancels_and_drains(_attempt: int) -> None:
    async def scenario() -> None:
        policy = GatePolicy()
        worker = make_worker(policy, emit_timeout=0.05)
        started = time.monotonic()
        try:
            with pytest.raises(AdapterBlocked):
                await async_emit(worker, "timeout")
            assert time.monotonic() - started < 1
            await asyncio.wait_for(policy.exited.wait(), 1)
        finally:
            await worker.aclose()
        assert isinstance(worker.last_error, TimeoutError)
        assert not worker._thread.is_alive() and not worker._callers
        assert worker._job is None
        assert len(worker.records) == 1 and not worker.records[0].proceeds

    asyncio.run(scenario())


def test_aclose_drains_owned_tasks_and_asyncgens_and_leaves_caller_loop_usable() -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        background_closed, generator_closed = asyncio.Event(), asyncio.Event()
        generators: list[AsyncGenerator[str, None]] = []
        tasks: list[asyncio.Task[None]] = []

        async def generate() -> AsyncGenerator[str, None]:
            try:
                yield "value"
            finally:
                loop.call_soon_threadsafe(generator_closed.set)

        class Policy:
            async def intercept(self, context: AgentContext) -> Verdict:
                started = asyncio.Event()

                async def background() -> None:
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        loop.call_soon_threadsafe(background_closed.set)

                tasks.append(asyncio.create_task(background()))
                generator = generate()
                generators.append(generator)
                assert await anext(generator) == "value"
                await started.wait()
                return ALLOW

        worker = make_worker(Policy())
        await async_emit(worker, "cleanup")
        await worker.aclose()
        await asyncio.wait_for(background_closed.wait(), 1)
        await asyncio.wait_for(generator_closed.wait(), 1)
        await checkpoint()
        assert all(task.done() for task in tasks)
        assert not loop.is_closed()
        assert not worker._close_tasks and not worker._closers
        await worker.aclose()

    asyncio.run(scenario())


def test_shutdown_drains_tasks_created_during_background_cleanup() -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        child_closed = asyncio.Event()
        tasks: list[asyncio.Task[None]] = []

        class Policy:
            async def intercept(self, context: AgentContext) -> Verdict:
                started = asyncio.Event()

                async def child() -> None:
                    try:
                        await asyncio.Event().wait()
                    finally:
                        loop.call_soon_threadsafe(child_closed.set)

                async def parent() -> None:
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        tasks.append(asyncio.create_task(child()))

                tasks.append(asyncio.create_task(parent()))
                await started.wait()
                return ALLOW

        worker = make_worker(Policy())
        await async_emit(worker, "spawn")
        await worker.aclose()
        await asyncio.wait_for(child_closed.wait(), 1)
        assert len(tasks) == 2 and all(task.done() for task in tasks)
        assert not worker._thread.is_alive()

    asyncio.run(scenario())


def test_cancellation_resistant_background_retains_owned_loop_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_module, "_CLEANUP_TIMEOUT", 0.05)

    async def scenario() -> None:
        caller_loop = asyncio.get_running_loop()
        cancelled, exited, retained = asyncio.Event(), asyncio.Event(), asyncio.Event()
        sdk_loop: asyncio.AbstractEventLoop | None = None
        gate: asyncio.Event | None = None
        tasks: list[asyncio.Task[None]] = []

        class RetainedNotification(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if record.msg == "adapter_background_cleanup_retained":
                    caller_loop.call_soon_threadsafe(retained.set)

        handler = RetainedNotification()
        worker_module.logger.addHandler(handler)

        class Policy:
            async def intercept(self, context: AgentContext) -> Verdict:
                nonlocal sdk_loop, gate
                sdk_loop = asyncio.get_running_loop()
                gate = asyncio.Event()
                started = asyncio.Event()

                async def background() -> None:
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        caller_loop.call_soon_threadsafe(cancelled.set)
                        assert gate is not None
                        await gate.wait()
                    finally:
                        caller_loop.call_soon_threadsafe(exited.set)

                tasks.append(asyncio.create_task(background()))
                await started.wait()
                return ALLOW

        worker = make_worker(Policy())
        await async_emit(worker, "background")
        try:
            with pytest.raises(AdapterCleanupError):
                await worker.aclose()
            await asyncio.wait_for(cancelled.wait(), 1)
            await asyncio.wait_for(retained.wait(), 1)
            assert worker._thread.is_alive()
            assert worker._job is None and not worker._callers
            assert sdk_loop is not None and not sdk_loop.is_closed()
            await checkpoint()
            with pytest.raises(AdapterBlocked):
                await async_emit(worker, "after-close")
        finally:
            assert sdk_loop is not None and gate is not None
            sdk_loop.call_soon_threadsafe(gate.set)
            await asyncio.wait_for(exited.wait(), 1)
            with pytest.raises(AdapterCleanupError):
                await worker.aclose()
            worker_module.logger.removeHandler(handler)
        assert not worker._thread.is_alive() and sdk_loop.is_closed()
        assert all(task.done() for task in tasks)

    asyncio.run(scenario())


def test_second_close_cancellation_retains_bounded_monitor_and_primary_cancel(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(worker_module, "_CLEANUP_TIMEOUT", 0.2)
    caplog.set_level(logging.WARNING)

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        entered, exited = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        worker = make_worker(AllowPolicy())

        def factory(context: AgentContextBuilder) -> AgentContext:
            loop.call_soon_threadsafe(entered.set)
            try:
                assert release.wait(3)
                return context.input(content="cleanup")
            finally:
                loop.call_soon_threadsafe(exited.set)

        emission = asyncio.create_task(worker.aemit(0, factory))
        await asyncio.wait_for(entered.wait(), 1)
        close = asyncio.create_task(worker.aclose())
        try:
            await checkpoint()
            close.cancel("primary cancellation")
            await checkpoint()
            close.cancel("second cancellation")
            with pytest.raises(asyncio.CancelledError, match="primary cancellation"):
                await close
            with worker._condition:
                assert len(worker._close_tasks) == 1 and worker._job is not None
                cleanup = next(iter(worker._close_tasks))
            with pytest.raises(AdapterCleanupError):
                await cleanup
            await checkpoint()
            assert not worker._close_tasks and not worker._closers
            with pytest.raises(AdapterBlocked):
                await emission
        finally:
            release.set()
            await asyncio.wait_for(exited.wait(), 1)
            await worker.aclose()
        assert any(record.message == "adapter_close_cancelled_again" for record in caplog.records)
        assert any(record.message == "adapter_close_monitor_failed" for record in caplog.records)

    asyncio.run(scenario())


def test_external_builder_actual_outcomes_records_and_detached_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        policy = GatePolicy()
        worker = make_worker(policy, max_records=2)
        external = builder()
        actual: list[EmitOutcome] = []
        original = InterceptionEmitter.emit

        async def capture(emitter: InterceptionEmitter, context: AgentContext) -> EmitOutcome:
            outcome = await original(emitter, context)
            actual.append(outcome)
            return outcome

        monkeypatch.setattr(InterceptionEmitter, "emit", capture)
        first = asyncio.create_task(worker.aemit_outcome(0, external.input(content="first")))
        await asyncio.wait_for(policy.entered.wait(), 1)
        context = external.input(content="detached")
        second = asyncio.create_task(worker.aemit_outcome(0, context))
        try:
            await checkpoint()
            context["target"] = {"content": "changed"}
            context["agent"] = {"id": "forged", "framework": "other"}
            policy.release()
            first_outcome, second_outcome = await asyncio.gather(first, second)
            assert first_outcome is actual[0] and second_outcome is actual[1]
            assert object_value(second_outcome.target) == {"content": "detached", "role": "user"}
            with worker._condition:
                assert second_outcome.record is worker._records[-1]
            shutdown = await worker.aemit_unchecked(0, external.agent_shutdown(reason="complete"))
            assert shutdown.sequence == 2 and shutdown.session_id == _SESSION
            assert [record.sequence for record in worker.records] == [1, 2]
            assert worker.records_dropped == 1
            assert [context["sequence"] for context in policy.contexts] == [0, 1, 2]
        finally:
            policy.release()
            await worker.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("agent", {"id": "foreign", "framework": "test"}),
        ("agent", {"id": _IDENTITY, "framework": "foreign"}),
        ("session", {"id": "foreign"}),
        ("sequence", True),
        ("sequence", -1),
        ("sequence", 1.0),
        ("interception_point", "unknown"),
    ],
)
@pytest.mark.parametrize("unchecked", [False, True])
def test_external_metadata_mismatch_denies_before_sdk_and_preserves_audit(
    field: str, value: JSON, unchecked: bool
) -> None:
    async def scenario() -> None:
        policy = AllowPolicy()
        worker = make_worker(policy)
        external = builder()
        context = external.input(content="mismatch")
        context[field] = value
        try:
            with pytest.raises(AdapterBlocked):
                if unchecked:
                    await worker.aemit_unchecked(0, context)
                else:
                    await worker.aemit_outcome(0, context)
            assert worker.denied and worker.available
            assert isinstance(worker.last_error, ValueError)
            assert not policy.contexts and not worker.records
            shutdown = await worker.aemit_unchecked(0, external.agent_shutdown(reason="failed"))
            assert shutdown.sequence == 1 and shutdown.proceeds
            assert len(policy.contexts) == 1 and len(worker.records) == 1
        finally:
            await worker.aclose()

    asyncio.run(scenario())


def test_unchecked_deny_returns_actual_record_and_sets_sticky_before_notification() -> None:
    async def scenario() -> None:
        class DenyPolicy:
            def intercept(self, context: AgentContext) -> Verdict:
                return Verdict.deny(reason="test_denial")

        worker = make_worker(DenyPolicy())
        external = builder()
        try:
            record = await worker.aemit_unchecked(0, external.input(content="denied"))
            assert isinstance(record, InterceptionRecord) and not record.proceeds
            assert worker.denied and worker.last_error is not None
            with worker._condition:
                assert record is worker._records[-1]
            with pytest.raises(AdapterBlocked):
                await worker.aemit_outcome(0, external.input(content="later"))
            shutdown = await worker.aemit_unchecked(0, external.agent_shutdown(reason="denied"))
            assert not shutdown.proceeds and len(worker.records) == 2
            assert worker.available
        finally:
            await worker.aclose()

    asyncio.run(scenario())


def test_sdk_transform_and_run_all_composition_are_preserved() -> None:
    async def scenario() -> None:
        calls: list[str] = []

        class TransformPolicy:
            def intercept(self, context: AgentContext) -> Verdict:
                calls.append("transform")
                return Verdict(
                    Decision.TRANSFORM, transform=Transform("$target.content", "replacement")
                )

        class ObservePolicy:
            def intercept(self, context: AgentContext) -> Verdict:
                calls.append("observe")
                return ALLOW

        bindings = {0: Binding(_IDENTITY, (TransformPolicy(), ObservePolicy()), _SESSION)}
        worker = Worker(bindings, None, 1, 2, 1, framework="test")
        bindings[1] = Binding("caller-owned", ())
        try:
            outcome = await worker.aemit_outcome(0, builder().input(content="original"))
            assert object_value(outcome.target)["content"] == "replacement"
            assert calls == ["transform", "observe"]
            assert len(outcome.record.verdicts) == 2
        finally:
            await worker.aclose()
        assert set(bindings) == {0, 1} and bindings[0].session_id == _SESSION
        assert not worker._bindings

    asyncio.run(scenario())


def test_async_reentry_from_sdk_thread_poison_is_not_a_deadlock() -> None:
    async def scenario() -> None:
        class Policy:
            async def intercept(self, context: AgentContext) -> Verdict:
                await async_emit(worker, "reentry")
                return ALLOW

        worker = make_worker(Policy())
        try:
            with pytest.raises(AdapterBlocked):
                await async_emit(worker, "outer")
            assert worker.denied and not worker.available
        finally:
            await worker.aclose()
        assert not worker._thread.is_alive() and not worker._callers
        assert len(worker.records) == 1 and not worker.records[0].proceeds

    asyncio.run(scenario())


def test_validator_original_error_is_private_and_logs_do_not_contain_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        worker = make_worker(AllowPolicy())
        error = ValueError("sensitive-policy-and-prompt-content")

        def validate(value: object) -> object:
            raise error

        try:
            with pytest.raises(AdapterBlocked) as caught:
                await worker.aemit(0, lambda context: context.input(content="secret"), validate)
            assert caught.value.__cause__ is error and worker.last_error is error
            assert worker.denied and worker.available
            assert len(worker.records) == 1 and worker.records[0].proceeds
            await worker.aemit(
                0, lambda context: context.agent_shutdown(reason="failed"), audit=True
            )
            assert len(worker.records) == 2
        finally:
            await worker.aclose()
        assert error.args[0] not in caplog.text and "secret" not in caplog.text

    asyncio.run(scenario())
