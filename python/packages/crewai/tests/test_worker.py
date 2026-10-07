"""Deterministic bounded-admission, timeout, and drain tests against the real SDK."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import uuid4

import pytest
from agent_control_crewai import AdapterBlocked, AdapterCleanupError
from agent_control_crewai._worker import Binding, Worker
from agent_hooks import ALLOW, AgentContext, Verdict


def worker_for(policy: Any, *, timeout: float = 1, emit_timeout: float = 2) -> Worker:
    return Worker({0: Binding(str(uuid4()), (policy,))}, None, timeout, emit_timeout, 10)


def emit(worker: Worker) -> object:
    return worker.emit(0, lambda builder: builder.input(content={"hello": "world"}))


def test_one_persistent_event_loop_and_unique_sequences() -> None:
    loops: list[asyncio.AbstractEventLoop] = []
    threads: list[int] = []
    contexts: list[AgentContext] = []

    class AsyncPolicy:
        async def intercept(self, context: AgentContext) -> Verdict:
            loops.append(asyncio.get_running_loop())
            threads.append(threading.get_ident())
            contexts.append(context)
            return ALLOW

    worker = worker_for(AsyncPolicy())
    try:
        emit(worker)
        emit(worker)
    finally:
        worker.close()
    assert loops[0] is loops[1] and loops[0].is_closed()
    assert threads[0] == threads[1] != threading.get_ident()
    assert [ctx["sequence"] for ctx in contexts] == [0, 1]
    assert contexts[0]["session"] == contexts[1]["session"]
    assert not worker._bindings and worker._loop is None


def test_sdk_async_timeout_cancels_and_drains() -> None:
    cancelled = threading.Event()

    class AsyncPolicy:
        async def intercept(self, context: AgentContext) -> Verdict:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return ALLOW

    worker = worker_for(AsyncPolicy(), timeout=0.05)
    try:
        with pytest.raises(AdapterBlocked):
            emit(worker)
        assert cancelled.wait(1)
        assert worker.records[0].verdict.reason == "host_error:interceptor_timeout"
    finally:
        worker.close()
    assert not worker._thread.is_alive()


@pytest.mark.parametrize("_attempt", range(3))
def test_nonpreemptible_callback_keeps_capacity_until_actual_exit(_attempt: int) -> None:
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    calls: list[int] = []

    class BlockingPolicy:
        def intercept(self, context: AgentContext) -> Verdict:
            calls.append(1)
            entered.set()
            try:
                assert release.wait(3)
                return ALLOW
            finally:
                exited.set()

    worker = worker_for(BlockingPolicy(), emit_timeout=0.05)
    try:
        started = time.monotonic()
        with pytest.raises(AdapterBlocked):
            emit(worker)
        assert time.monotonic() - started < 1
        assert entered.is_set() and not exited.is_set()
        assert worker._job is not None
        with pytest.raises(AdapterCleanupError):
            worker.close()
        assert worker._thread.is_alive()
        with pytest.raises(AdapterBlocked):
            emit(worker)
        assert calls == [1]
    finally:
        release.set()
        worker._thread.join(timeout=2)
        worker.close()
    assert exited.is_set()
    assert worker._job is None and not worker._bindings


def test_outer_timeout_cancels_cooperative_callback() -> None:
    cancelled = threading.Event()

    class AsyncPolicy:
        async def intercept(self, context: AgentContext) -> Verdict:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return ALLOW

    worker = worker_for(AsyncPolicy(), timeout=3, emit_timeout=0.1)
    try:
        with pytest.raises(AdapterBlocked):
            emit(worker)
        assert cancelled.wait(1)
    finally:
        worker.close()
    assert not worker._thread.is_alive()


def test_one_running_and_eight_waiters_saturation_is_atomic() -> None:
    entered, release = threading.Event(), threading.Event()
    calls: list[int] = []

    class BlockingPolicy:
        def intercept(self, context: AgentContext) -> Verdict:
            calls.append(1)
            entered.set()
            assert release.wait(5)
            return ALLOW

    worker = worker_for(BlockingPolicy(), emit_timeout=3)
    try:
        with ThreadPoolExecutor(max_workers=9) as pool:
            running = pool.submit(emit, worker)
            assert entered.wait(1)
            waiting = [pool.submit(emit, worker) for _ in range(8)]
            with worker._condition:
                assert worker._condition.wait_for(lambda: len(worker._callers) == 9, timeout=2)
            with pytest.raises(AdapterBlocked):
                emit(worker)
            assert calls == [1] and worker._job is not None
            release.set()
            for future in [running, *waiting]:
                with pytest.raises(AdapterBlocked):
                    future.result(timeout=2)
    finally:
        release.set()
        worker.close()
    assert not worker._callers and not worker._thread.is_alive()


def test_same_worker_reentry_fails_without_deadlock() -> None:
    class ReentrantPolicy:
        def intercept(self, context: AgentContext) -> Verdict:
            emit(worker)
            return ALLOW

    worker = worker_for(ReentrantPolicy(), emit_timeout=0.2)
    try:
        with pytest.raises(AdapterBlocked):
            emit(worker)
    finally:
        worker.close()
    assert not worker._thread.is_alive()


def test_cleanup_error_from_background_task_is_not_silent() -> None:
    tasks: list[asyncio.Task[None]] = []

    class AsyncPolicy:
        async def intercept(self, context: AgentContext) -> Verdict:
            started = asyncio.Event()

            async def background() -> None:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise RuntimeError("background cleanup failed") from None

            tasks.append(asyncio.create_task(background()))
            await started.wait()
            return ALLOW

    worker = worker_for(AsyncPolicy())
    emit(worker)
    with pytest.raises(AdapterCleanupError):
        worker.close()
    assert not worker._thread.is_alive()


def test_invalid_target_is_denied_before_next_emission() -> None:
    class AllowPolicy:
        def intercept(self, context: AgentContext) -> Verdict:
            return ALLOW

    worker = worker_for(AllowPolicy())

    def invalid(value: object) -> object:
        raise ValueError("invalid host transform")

    try:
        with pytest.raises(AdapterBlocked):
            worker.emit(0, lambda builder: builder.input(content={}), invalid)
        with pytest.raises(AdapterBlocked):
            emit(worker)
        assert len(worker.records) == 1
        # Denial does not poison a healthy lane: shutdown can still be audited.
        worker.emit(0, lambda builder: builder.agent_shutdown(reason="failed"), audit=True)
        assert len(worker.records) == 2
    finally:
        worker.close()
