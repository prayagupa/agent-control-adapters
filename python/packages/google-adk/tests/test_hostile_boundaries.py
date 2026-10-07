"""Strong actor ownership, routing immutability and bounded resistant native work."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import cast

import pytest
from adk_adapter_test_helpers import (
    BoundaryModel,
    Control,
    controlled,
    native_case,
    text_response,
)
from agent_control_adapters import AdapterBlocked, AdapterCleanupError, RunRequest
from agent_control_adapters.runtime import NativeTaskOwner, RunBudget
from agent_control_google_adk._profile import native_field, profile
from google.adk.agents import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService


@pytest.mark.parametrize(
    "key,value",
    [
        ("id", "spoofed"),
        ("name", "foreign_tool"),
        ("args", {"value": "unapproved"}),
        ("id", True),
        ("args", {"value": "native", "session_id": "spoof"}),
    ],
)
def test_model_transform_cannot_change_native_tool_routing(key: str, value: object) -> None:
    async def scenario() -> None:
        case = native_case(use_tool=True)
        call: dict[str, object] = {"id": "call-0", "name": "ping", "args": {"value": "native"}}
        call[key] = value
        adapter = controlled(
            case,
            interceptors=[
                Control(
                    "post_model_call",
                    replacement={
                        "content": None,
                        "tool_calls": [call],
                        "finish_reason": "stop",
                    },
                )
            ],
        )
        with pytest.raises(AdapterBlocked):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert case.effects == []
        assert len(case.model.seen) == 1
        assert not any(
            record.interception_point.value == "pre_tool_call" for record in adapter.records
        )

    asyncio.run(scenario())


def test_custom_native_agent_and_delegation_subtree_rejected_before_effects() -> None:
    class CustomAgent(LlmAgent):
        pass

    case = native_case()
    case.runner.agent = CustomAgent(name="custom", model=case.model, mode="chat")
    case.runner.app.root_agent = case.runner.agent
    with pytest.raises(TypeError):
        profile(case.runner)
    case = native_case()
    child = native_case().agent
    child.parent_agent = case.agent
    case.agent.sub_agents.append(child)
    with pytest.raises(ValueError):
        profile(case.runner)


@pytest.mark.parametrize("operation", ["model", "tool", "post"])
def test_unknown_duplicate_and_mismatched_correlations_are_sticky(operation: str) -> None:
    budget = RunBudget("adk-correlation-test")
    budget.before_model(0, "known-model")
    if operation == "model":
        with pytest.raises(AdapterBlocked, match="Duplicate"):
            budget.before_model(0, "known-model")
    elif operation == "tool":
        with pytest.raises(AdapterBlocked, match="Unknown"):
            budget.before_tool(0, "foreign")
    else:
        with pytest.raises(AdapterBlocked, match="Unknown"):
            budget.after_model(1, "known-model")
    with pytest.raises(AdapterBlocked):
        budget.event()
    budget.close()
    assert budget.snapshot.pending == 0


@pytest.mark.parametrize("second_cancel", [False, True])
def test_resistant_actual_native_model_remains_owned_through_bounded_drain(
    monkeypatch: pytest.MonkeyPatch,
    second_cancel: bool,
) -> None:
    monkeypatch.setattr("agent_control_adapters.runtime._NATIVE_CLEANUP_TIMEOUT", 0.04)

    async def scenario() -> None:
        entered, cancelled, release, exited = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )

        async def resistant(
            self: BoundaryModel,
            llm_request: LlmRequest,
            stream: bool = False,
        ) -> AsyncGenerator[LlmResponse, None]:
            entered.set()
            try:
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                yield text_response()
            finally:
                exited.set()

        monkeypatch.setattr(BoundaryModel, "generate_content_async", resistant)
        adapter = controlled(native_case())
        task = asyncio.create_task(adapter.arun(RunRequest({"prompt": "safe"})))
        await entered.wait()
        task.cancel()
        await cancelled.wait()
        if second_cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        owner = native_field(adapter, "_owner")
        assert isinstance(owner, NativeTaskOwner)
        assert owner.pending and adapter.cleanup_error is not None and not exited.is_set()
        owned = native_field(owner, "_native_task")
        assert isinstance(owned, asyncio.Task)
        owned_task = cast("asyncio.Task[object]", owned)
        release.set()
        done, pending = await asyncio.wait((owned_task,), timeout=1)
        assert done and not pending
        assert exited.is_set() and not owner.pending
        assert not asyncio.get_running_loop().is_closed()

    asyncio.run(scenario())


@pytest.mark.parametrize("deny", [False, True])
def test_owned_session_cleanup_failure_surfaces_without_replacing_primary(
    monkeypatch: pytest.MonkeyPatch,
    deny: bool,
) -> None:
    async def failed_delete(
        self: InMemorySessionService,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> None:
        raise OSError("Actual owned session deletion test failure")

    monkeypatch.setattr(InMemorySessionService, "delete_session", failed_delete)

    async def scenario() -> None:
        adapter = controlled(
            native_case(),
            interceptors=[
                Control("pre_model_call", deny=True) if deny else Control("never", deny=True),
            ],
        )
        expected = AdapterBlocked if deny else AdapterCleanupError
        with pytest.raises(expected):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert adapter.cleanup_error is not None
        assert isinstance(adapter.cleanup_error.__cause__, OSError)

    asyncio.run(scenario())
