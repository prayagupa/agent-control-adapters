"""Native event-counter threshold and complete scalar default fingerprint."""

from __future__ import annotations

import asyncio

import pytest
from adk_adapter_test_helpers import controlled, native_case
from agent_hooks_adapters import AdapterBlocked, RunRequest
from agent_hooks_google_adk._plugin import ControlsPlugin
from agent_hooks_google_adk._profile import native_field
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event


def test_actual_plugin_native_event_257_denies_at_exact_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ControlsPlugin.on_event_callback
    observed: list[object] = []

    async def exercise(
        self: ControlsPlugin,
        *,
        invocation_context: InvocationContext,
        event: Event,
    ) -> None:
        for _ in range(256):
            await original(self, invocation_context=invocation_context, event=event)
        observed.append(native_field(self, "_native_events"))
        try:
            await original(self, invocation_context=invocation_context, event=event)
        finally:
            observed.append(native_field(self, "_native_events"))

    monkeypatch.setattr(ControlsPlugin, "on_event_callback", exercise)

    async def scenario() -> None:
        case = native_case()
        adapter = controlled(case)
        with pytest.raises(AdapterBlocked):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert observed == [256, 257]
        assert len(case.model.seen) == 1
        assert adapter.cleanup_error is None

    asyncio.run(scenario())


def test_actual_tool_default_mutation_rejected_before_admitted_effects() -> None:
    async def scenario() -> None:
        case = native_case()

        async def optional(value: str = "captured-default") -> dict[str, str]:
            """A native optional argument with a scalar default."""
            return {"result": value}

        case.agent.tools = [optional]
        adapter = controlled(case)
        optional.__defaults__ = ("changed-default",)
        with pytest.raises(ValueError, match="changed"):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert case.model.seen == []
        assert adapter.records == ()

    asyncio.run(scenario())
