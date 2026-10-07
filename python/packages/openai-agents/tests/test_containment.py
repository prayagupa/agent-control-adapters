"""Cancellation-resistant native model/tool work remains bounded and explicitly retained."""

from __future__ import annotations

import asyncio

import pytest
from agent_control_adapters import RunRequest
from agent_control_adapters import runtime as shared_runtime
from agent_control_openai_agents import adapter as native_adapter
from agent_control_openai_agents._codec import raw_arguments
from agents.tool_context import ToolContext
from openai_adapter_test_helpers import Policy, controlled, native_case, scope_of


@pytest.mark.parametrize("during", ["model", "tool"])
def test_cancel_resistant_native_work_retained_until_real_exit(
    during: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shared_runtime, "_NATIVE_CLEANUP_TIMEOUT", 0.03)
    monkeypatch.setattr(native_adapter, "_TOOL_DRAIN_TIMEOUT", 0.02)

    async def scenario() -> None:
        before = asyncio.all_tasks()
        entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        case = native_case(use_tool=during == "tool")
        if during == "model":
            case.model.entered, case.model.release, case.model.cancelled = (
                entered,
                release,
                cancelled,
            )
            case.model.suppress_cancel = True
        else:

            async def resistant(context: ToolContext[object], raw: str) -> str:
                entered.set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        cancelled.set()
                return str(raw_arguments(raw)["value"])

            case.tool.on_invoke_tool = resistant
        adapter = controlled(case, policies=[Policy()])
        task = asyncio.create_task(adapter.arun(RunRequest({"prompt": "safe"})))
        await asyncio.wait_for(entered.wait(), 2)
        try:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), 1)
            assert cancelled.is_set()
            assert adapter.cleanup_error is not None
            retained = asyncio.all_tasks() - before
            assert retained, "Cancellation must not falsely claim native work has stopped"
            assert not case.model.closed
        finally:
            release.set()
            pending = asyncio.all_tasks() - before
            if pending:
                _, remaining = await asyncio.wait(pending, timeout=1)
                assert not remaining, "Retained native work failed to exit after release"
        assert not scope_of(adapter).tool_tasks
        assert asyncio.all_tasks() == before

    asyncio.run(scenario())
