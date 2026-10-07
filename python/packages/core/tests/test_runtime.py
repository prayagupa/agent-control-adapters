"""Exact execution limits, atomic correlation and owned native-task lifecycle."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import pytest
from agent_hooks_adapters import AdapterBlocked, RunRequest
from agent_hooks_adapters import runtime as runtime_module
from agent_hooks_adapters.runtime import (
    NativeExecutionLimits,
    NativeTaskOwner,
    RunAdmission,
    RunBudget,
    prompt_value,
    validate_topology,
)

_T = TypeVar("_T")


def run(coroutine: Coroutine[object, object, _T]) -> _T:
    return asyncio.run(coroutine)


@pytest.mark.parametrize(
    "run_request",
    [
        None,
        RunRequest(),
        RunRequest({}),
        RunRequest({"prompt": ""}),
        RunRequest({"prompt": " \n\t"}),
        RunRequest({"prompt": 1}),
        RunRequest({"prompt": "x" * 65_537}),
        RunRequest({"prompt": "safe", "session_id": "caller"}),
        RunRequest({"prompt": "safe", "messages": []}),
        RunRequest({"prompt": "safe", "callbacks": []}),
    ],
)
def test_prompt_admission_rejects_extra_fields_and_invalid_values(
    run_request: RunRequest | None,
) -> None:
    admission = RunAdmission()
    with pytest.raises(ValueError):
        admission.admit(run_request)
    with pytest.raises(RuntimeError, match="single-use"):
        admission.admit(RunRequest({"prompt": "valid"}))


@pytest.mark.parametrize("count", [1, 65_536])
def test_prompt_length_boundaries(count: int) -> None:
    assert prompt_value(RunRequest({"prompt": "x" * count})) == "x" * count


def test_single_use_admission_is_atomic() -> None:
    admission = RunAdmission()
    ready = threading.Barrier(2)

    def invoke() -> str:
        ready.wait(timeout=2)
        try:
            return admission.admit(RunRequest({"prompt": "accepted"}))
        except RuntimeError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        calls = [executor.submit(invoke) for _ in range(2)]
        assert sorted(call.result(timeout=2) for call in calls) == ["accepted", "rejected"]


def test_actor_node_edge_caps() -> None:
    validate_topology(actors=1, nodes=1, edges=0)
    validate_topology(actors=16, nodes=16, edges=64)
    for actors, nodes, edges in [(17, 1, 0), (1, 17, 0), (1, 1, 65), (0, 1, 0), (1, 0, 0)]:
        with pytest.raises(ValueError):
            validate_topology(actors=actors, nodes=nodes, edges=edges)
    with pytest.raises(TypeError):
        validate_topology(actors=True, nodes=1, edges=0)


def test_model_call_budget_exhausted() -> None:
    budget = RunBudget("trace")
    for index in range(10):
        budget.before_model(0, str(index))
        budget.after_model(0, str(index))
    assert budget.snapshot.model_calls == 10
    with pytest.raises(AdapterBlocked, match="Model call limit") as blocked:
        budget.before_model(0, "eleventh")
    assert budget.snapshot.model_calls == 10
    with pytest.raises(AdapterBlocked) as later:
        budget.event()
    assert later.value is blocked.value


def test_one_tool_call_per_model_output() -> None:
    budget = RunBudget("trace")
    budget.before_model(0, "model")
    budget.after_model(0, "model", ["tool"])
    budget.before_tool(0, "tool")
    budget.after_tool(0, "tool")
    assert budget.snapshot.pending == 0
    budget.before_model(0, "model2")
    with pytest.raises(AdapterBlocked, match="one tool"):
        budget.after_model(0, "model2", ["first", "second"])


def test_pending_correlation_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate the pending bound from the independently tested, lower model-call cap.
    monkeypatch.setattr(runtime_module, "_MAX_MODEL_CALLS", 65)
    budget = RunBudget("trace")
    for index in range(64):
        budget.before_model(0, f"model_{index}")
    assert budget.snapshot.pending == 64
    with pytest.raises(AdapterBlocked, match="Pending correlation"):
        budget.before_model(0, "overflow")
    assert budget.snapshot.pending == 64
    budget.close()
    assert budget.snapshot.pending == 0
    with pytest.raises(AdapterBlocked):
        budget.event()


def test_native_event_cap() -> None:
    budget = RunBudget("trace")
    for _ in range(256):
        budget.event()
    assert budget.snapshot.native_events == 256
    with pytest.raises(AdapterBlocked, match="Native event limit"):
        budget.event()
    assert budget.snapshot.native_events == 256


def test_fixed_recursion_and_concurrency() -> None:
    limits = NativeExecutionLimits()
    assert limits.recursion == 30
    assert limits.tool_concurrency == 1


def test_unknown_duplicate_and_cross_actor_correlation() -> None:
    budget = RunBudget("trace")
    budget.before_model(0, "model")
    budget.after_model(0, "model", ["tool"])
    with pytest.raises(AdapterBlocked, match="Unknown"):
        budget.before_tool(1, "tool")
    other = RunBudget("other")
    other.before_model(0, "model")
    other.after_model(0, "model")
    with pytest.raises(AdapterBlocked, match="duplicate"):
        other.after_model(0, "model")


def test_atomic_model_budget_under_concurrent_callers() -> None:
    budget = RunBudget("trace")
    barrier = threading.Barrier(11)

    def invoke(index: int) -> bool:
        barrier.wait(timeout=2)
        try:
            budget.before_model(0, str(index))
            return True
        except AdapterBlocked:
            return False

    with ThreadPoolExecutor(max_workers=11) as executor:
        futures = [executor.submit(invoke, index) for index in range(11)]
        assert sum(future.result(timeout=2) for future in futures) == 10
    assert budget.snapshot.model_calls == 10
    assert budget.snapshot.pending == 10


def test_native_run_success_preserves_caller_loop() -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        failures: list[BaseException] = []
        owner = NativeTaskOwner(failures.append)

        async def execute() -> str:
            assert asyncio.get_running_loop() is loop
            return "native"

        assert await owner.run(execute) == "native"
        assert not owner.pending
        assert owner.cleanup_error is None
        assert not failures
        assert not loop.is_closed()
        assert await execute() == "native"
        with pytest.raises(RuntimeError, match="single-use"):
            await owner.run(execute)

    run(scenario())


def test_native_run_deadline_and_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module, "_NATIVE_RUN_TIMEOUT", 0.02)
    monkeypatch.setattr(runtime_module, "_NATIVE_CLEANUP_TIMEOUT", 0.1)

    async def scenario() -> None:
        entered, stopped = asyncio.Event(), asyncio.Event()
        failures: list[BaseException] = []
        owner = NativeTaskOwner(failures.append)

        async def execute() -> str:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            return "unreachable"

        with pytest.raises(AdapterBlocked, match="deadline") as blocked:
            await owner.run(execute)
        assert entered.is_set()
        assert stopped.is_set()
        assert not owner.pending
        assert owner.cleanup_error is None
        assert failures == [blocked.value]

    run(scenario())


def test_native_run_cancellation_retains_primary_during_second_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module, "_NATIVE_CLEANUP_TIMEOUT", 0.2)

    async def scenario() -> None:
        entered, draining, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        failures: list[BaseException] = []
        owner = NativeTaskOwner(failures.append)

        async def execute() -> str:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                draining.set()
                await release.wait()
            return "drained"

        caller = asyncio.create_task(owner.run(execute))
        await entered.wait()
        caller.cancel("primary")
        await draining.wait()
        caller.cancel("secondary")
        release.set()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await caller
        assert cancelled.value.args == ("primary",)
        assert len(failures) == 1
        assert not owner.pending
        assert owner.cleanup_error is None
        assert not any(
            task.get_name().startswith("agent-controls-") and not task.done()
            for task in asyncio.all_tasks()
        )

    run(scenario())


def test_noncooperative_native_task_remains_owned_until_real_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module, "_NATIVE_RUN_TIMEOUT", 0.01)
    monkeypatch.setattr(runtime_module, "_NATIVE_CLEANUP_TIMEOUT", 0.02)

    async def scenario() -> None:
        release, stopped = asyncio.Event(), asyncio.Event()
        failures: list[BaseException] = []
        owner = NativeTaskOwner(failures.append)

        async def execute() -> str:
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            finally:
                stopped.set()
            return "late"

        try:
            with pytest.raises(AdapterBlocked, match="deadline"):
                await owner.run(execute)
            assert owner.pending
            assert owner.cleanup_error is not None
            assert len(failures) == 1
            with pytest.raises(RuntimeError, match="single-use"):
                await owner.run(execute)
        finally:
            release.set()
            await asyncio.wait_for(stopped.wait(), timeout=1)
        assert not owner.pending

    run(scenario())


def test_original_native_exception_preserved_and_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        original = ValueError("private prompt/tool/policy payload")
        failures: list[BaseException] = []
        owner = NativeTaskOwner(failures.append)

        async def execute() -> str:
            raise original

        with pytest.raises(ValueError) as caught:
            await owner.run(execute)
        assert caught.value is original
        assert failures == [original]
        assert owner.cleanup_error is None
        assert not owner.pending

    run(scenario())
    assert "private prompt/tool/policy payload" not in caplog.text
