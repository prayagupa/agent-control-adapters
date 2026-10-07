"""P3 real-Rego consumers and cancellation-resistant actual-native runner contracts."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Literal

import pytest
from agent_adapter_conformance import assert_governed_success, make_rego_manifest
from agent_control_adapters import AdapterBlocked, RunRequest
from agent_control_adapters import runtime as shared_runtime
from agent_control_adapters.codecs import wire_equal
from agent_control_adapters.json import JSON
from agent_control_adapters.runtime import NativeTaskOwner
from agent_hooks import Decision
from graph_adapter_contract_vectors import FamilyCase, Mode, execute
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph_adapter_test_helpers import POINTS, NativeTransport

_CONSUMED: list[tuple[str, str, JSON]] = [
    ("input", "$target.content", "rego input"),
    ("pre_model_call", "$target[0].content", "rego request"),
    ("post_model_call", "$target.content", "rego model"),
    ("pre_tool_call", "$target.value", "rego arguments"),
    ("post_tool_call", "$target", "rego result"),
    ("post_tool_call", "$target", {"echo": "rego", "valid": True, "count": 0, "none": None}),
    ("output", "$target.content", "rego output"),
]
_REJECTED: list[tuple[str, str, JSON, int, int]] = [
    ("input", "$target.role", "system", 0, 0),
    ("pre_model_call", "$target[0].role", "assistant", 0, 0),
    ("pre_model_call", "$target[0].id", "spoofed-message", 0, 0),
    ("post_model_call", "$target.tool_calls[0].id", "stolen-call", 1, 0),
    ("post_model_call", "$target.tool_calls[0].name", "foreign-tool", 1, 0),
    ("post_model_call", "$target.tool_calls[0].args.value", "rewritten-route", 1, 0),
    ("post_model_call", "$target.finish_reason", "stop", 1, 0),
    ("post_model_call", "$target.usage", {"total_tokens": 0}, 1, 0),
    ("pre_tool_call", "$target.actor_id", "foreign-actor", 1, 0),
    ("pre_tool_call", "$target.value", 17, 1, 0),
    ("post_tool_call", "$metadata.agent_id", "foreign-actor", 1, 1),
    ("output", "$target.tool_calls", [], 2, 1),
]


def _transform_manifest(root: Path, point: str, path: str, value: JSON) -> Path:
    verdict = json.dumps({"decision": "transform", "transform": {"path": path, "value": value}})
    return make_rego_manifest(
        root,
        f"verdict := {verdict} if {{ input.snapshot.interception_point == {json.dumps(point)} }}",
    )


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(("point", "path", "value"), _CONSUMED)
def test_real_rego_transforms_consumed(
    family_case: FamilyCase, mode: Mode, point: str, path: str, value: JSON, tmp_path: Path
) -> None:
    """Run native graphs with real ACS and assert the downstream consumer, not just verdict."""
    adapter = family_case.adapter(path=_transform_manifest(tmp_path, point, path, value))
    result = execute(adapter, mode)
    assert_governed_success(result, points=POINTS)
    transformed = [
        record
        for record in result.records
        if record.interception_point.value == point
        and record.verdict.decision is Decision.TRANSFORM
    ]
    assert len(transformed) == (2 if point in ("pre_model_call", "post_model_call") else 1)
    assert family_case.model.calls == 2
    assert family_case.effects == [value if point == "pre_tool_call" else "original"]
    if point in ("input", "pre_model_call"):
        first = family_case.model.seen[0]
        message = first[0] if point == "pre_model_call" else first[-1]
        assert message.content == value
        assert message.type in ("system", "human")
    elif point == "post_model_call":
        prior = next(item for item in family_case.model.seen[1] if isinstance(item, AIMessage))
        assert prior.content == value
        assert prior.id == "response-tool"
        assert prior.tool_calls == [
            {"id": "call-1", "name": "ping", "args": {"value": "original"}, "type": "tool_call"}
        ]
        assert prior.usage_metadata == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        assert prior.response_metadata == {"finish_reason": "tool_calls"}
        assert result.output == value
    elif point in ("pre_tool_call", "post_tool_call"):
        actual = next(item for item in family_case.model.seen[1] if isinstance(item, ToolMessage))
        expected = value if point == "post_tool_call" else "rego arguments"
        assert isinstance(actual.content, str)
        if isinstance(expected, str):
            assert actual.content == expected
        else:
            assert wire_equal(json.loads(actual.content), expected)
        assert actual.tool_call_id == "call-1"
        assert actual.name == "ping" and actual.status == "success"
    else:
        assert result.output == value
    scope = family_case.scope()
    assert not scope.active
    assert scope.ledger.pending == scope.session.budget.snapshot.pending == 0
    assert adapter.cleanup_error is None


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(("point", "path", "value", "calls", "effects"), _REJECTED)
def test_real_rego_unsupported_metadata_and_routing_fail_closed(
    family_case: FamilyCase,
    mode: Mode,
    point: str,
    path: str,
    value: JSON,
    calls: int,
    effects: int,
    tmp_path: Path,
) -> None:
    """Reject unsupported changes without dispatching the original/forged route or success."""
    adapter = family_case.adapter(path=_transform_manifest(tmp_path, point, path, value))
    with pytest.raises(AdapterBlocked) as raised:
        execute(adapter, mode)
    assert raised.value.__cause__ is not None
    actual = [record for record in adapter.records if record.interception_point.value == point]
    assert actual and actual[-1].verdict.decision in (Decision.TRANSFORM, Decision.DENY)
    assert family_case.model.calls == calls
    assert family_case.effects == (["original"] if effects else [])
    assert family_case.scope().ledger.pending == 0
    if effects == 0:
        assert not any(
            record.interception_point.value == "post_tool_call" for record in adapter.records
        )
    if point != "output":
        assert not any(record.interception_point.value == "output" for record in adapter.records)


@pytest.mark.parametrize("failure", ["cancel", "deadline"])
@pytest.mark.parametrize("iteration", [0, 1])
def test_actual_native_resistant_retention_and_repeated_cleanup(
    family_case: FamilyCase,
    failure: Literal["cancel", "deadline"],
    iteration: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hold actual native transport after cancellation until an explicit barrier release."""
    monkeypatch.setattr(shared_runtime, "_NATIVE_RUN_TIMEOUT", 1.0)
    monkeypatch.setattr(shared_runtime, "_NATIVE_CLEANUP_TIMEOUT", 0.05)

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        baseline_tasks = asyncio.all_tasks()
        baseline_threads = {thread.ident for thread in threading.enumerate()}
        entered, cancellation, release, exited = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )
        transports: list[asyncio.Task[object]] = []
        cancellations: list[asyncio.CancelledError] = []

        async def resistant(
            self: NativeTransport,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: object,
        ) -> ChatResult:
            assert asyncio.get_running_loop() is loop
            current = asyncio.current_task()
            assert current is not None
            transports.append(current)
            entered.set()
            try:
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError as error:
                        cancellations.append(error)
                        cancellation.set()
                self.seen.append(list(messages))
                self.calls += 1
                return ChatResult(generations=[ChatGeneration(message=self.responses[0])])
            finally:
                exited.set()

        monkeypatch.setattr(NativeTransport, "_agenerate", resistant)
        adapter = family_case.adapter(max_records=8)
        from agent_control_langgraph._inspection import attribute

        owner = attribute(adapter, "_owner")
        assert isinstance(owner, NativeTaskOwner)
        caller = asyncio.create_task(
            adapter.arun(RunRequest({"prompt": f"resistant native iteration {iteration}"})),
            name="p3-actual-native-caller",
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            scope = family_case.scope()
            assert scope.session.budget.snapshot.model_calls == 1
            assert scope.session.budget.snapshot.pending == 1
            if failure == "cancel":
                caller.cancel()
            await asyncio.wait_for(cancellation.wait(), 3)
            assert owner.pending and len(transports) == 1 and not transports[0].done()
            if failure == "cancel":
                caller.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(caller), 3)
            else:
                with pytest.raises(AdapterBlocked, match="deadline"):
                    await asyncio.wait_for(asyncio.shield(caller), 3)
            assert owner.pending
            assert adapter.cleanup_error is owner.cleanup_error
            assert adapter.cleanup_error is not None
            assert not release.is_set() and not exited.is_set()
            assert family_case.effects == [] and family_case.model.calls == 0
            assert scope.session.last_error is not None
            primary = scope.session.last_error
            assert isinstance(
                primary, asyncio.CancelledError if failure == "cancel" else AdapterBlocked
            )
            assert cancellations
            assert (
                len(
                    [
                        task
                        for task in asyncio.all_tasks()
                        if task.get_name() == "agent-controls-native-run" and not task.done()
                    ]
                )
                == 1
            )
            retained = adapter.records
            assert len(retained) <= 8
            assert any(record.interception_point.value == "pre_model_call" for record in retained)
            assert not any(
                record.interception_point.value in ("post_model_call", "pre_tool_call", "output")
                for record in retained
            )
            for _ in range(2):
                await scope.session.aclose(primary)
                assert scope.session.last_error is primary
                assert owner.pending and not transports[0].done()
                assert adapter.records == retained
                assert scope.session.budget.snapshot.pending == 0
                with pytest.raises(RuntimeError, match="single-use"):
                    await adapter.arun(RunRequest({"prompt": "never a second task"}))
                with pytest.raises(RuntimeError, match="single-use"):
                    adapter.run(RunRequest({"prompt": "never a second sync effect"}))
                assert scope.session.budget.snapshot.model_calls == 1
        finally:
            release.set()
            live = asyncio.all_tasks() - baseline_tasks
            if live:
                await asyncio.wait_for(asyncio.gather(*live, caller, return_exceptions=True), 3)
        assert exited.is_set() and not owner.pending
        assert transports[0].done()
        assert family_case.effects == [] and family_case.model.calls == 1
        scope = family_case.scope()
        assert not scope.active
        assert scope.ledger.pending == scope.session.budget.snapshot.pending == 0
        await scope.session.aclose(scope.session.last_error)
        assert scope.session.last_error is primary
        assert adapter.cleanup_error is not None
        assert not (asyncio.all_tasks() - baseline_tasks)
        assert not any(
            thread.ident not in baseline_threads and thread.name.startswith("agent-hooks-")
            for thread in threading.enumerate()
        )
        assert not loop.is_closed()

    asyncio.run(scenario())
