"""Registered public selector and exact aliases through the actual native runner."""

from __future__ import annotations

import asyncio

from adk_adapter_test_helpers import Allow, native_case
from agent_adapter_conformance import assert_governed_success
from agent_control_adapters import (
    Backend,
    RunRequest,
    create_adapter,
    with_agent_controls,
    with_agent_hooks,
)
from agent_control_google_adk.adapter import GoogleADKAdapter


def test_registered_public_selector_returns_native_async_adapter() -> None:
    async def scenario() -> None:
        case = native_case()
        adapter = with_agent_controls(
            backend=Backend.GOOGLE_ADK,
            runtime=case.runner,
            interceptors=[Allow()],
        )
        assert isinstance(adapter, GoogleADKAdapter)
        result = await adapter.arun(RunRequest({"prompt": "public factory test"}))
        assert_governed_success(
            result,
            points=(
                "agent_startup",
                "input",
                "pre_model_call",
                "post_model_call",
                "output",
                "agent_shutdown",
            ),
        )
        assert len(case.model.seen) == 1

    asyncio.run(scenario())


def test_public_aliases_remain_exact_function_identities() -> None:
    assert with_agent_hooks is with_agent_controls
    assert create_adapter is with_agent_controls
    assert Backend.GOOGLE_ADK.value == "google-adk"
