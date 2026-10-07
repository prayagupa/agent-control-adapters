"""Owning-project strict admission, actual actor membership and profile rejection."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest
from adk_adapter_test_helpers import Allow, controlled, native_case
from agent_adapter_conformance import INVALID_PROMPTS
from agent_hooks_adapters import AgentPolicy, RunRequest
from agent_hooks_google_adk._profile import profile
from agent_hooks_google_adk.portable import create_backend
from google.adk.agents import (
    BaseAgent,
    LlmAgent,
    SequentialAgent,  # pyright: ignore[reportDeprecated]
)
from google.adk.apps import App
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types


@pytest.mark.parametrize("run_request", INVALID_PROMPTS)
def test_invalid_admitted_prompt_consumes_instance_without_native_work(
    run_request: RunRequest | None,
) -> None:
    async def scenario() -> None:
        case = native_case()
        adapter = controlled(case)
        with pytest.raises(ValueError):
            await adapter.arun(run_request)
        with pytest.raises(RuntimeError, match="single-use"):
            await adapter.arun(RunRequest({"prompt": "corrected"}))
        assert case.model.seen == []
        assert adapter.records == ()
        assert adapter.cleanup_error is None

    asyncio.run(scenario())


def test_sync_refusal_before_admission_then_arun_success_and_single_use() -> None:
    async def scenario() -> None:
        case = native_case()
        adapter = controlled(case)
        with pytest.raises(RuntimeError, match="requires await"):
            adapter.run()
        assert (await adapter.arun(RunRequest({"prompt": "safe"}))).output == "native answer"
        with pytest.raises(RuntimeError, match="single-use"):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        with pytest.raises(RuntimeError, match="requires await"):
            adapter.run()
        assert len(case.model.seen) == 1

    asyncio.run(scenario())


def test_native_profile_mutation_consumes_admission_before_startup() -> None:
    async def scenario() -> None:
        case = native_case()
        adapter = controlled(case)
        case.agent.instruction = "changed after binding"
        with pytest.raises(ValueError, match="changed"):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        with pytest.raises(RuntimeError, match="single-use"):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert case.model.seen == []
        assert adapter.records == ()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("retry_config", {"max_attempts": 2}),
        ("timeout", 1),
        ("input_schema", str),
        ("output_schema", str),
        ("state_schema", str),
        ("before_agent_callback", lambda: None),
        ("after_agent_callback", lambda: None),
        ("before_model_callback", lambda: None),
        ("after_model_callback", lambda: None),
        ("on_model_error_callback", lambda: None),
        ("before_tool_callback", lambda: None),
        ("after_tool_callback", lambda: None),
        ("on_tool_error_callback", lambda: None),
        ("static_instruction", "uncovered"),
        ("planner", object()),
        ("code_executor", object()),
        ("parallel_worker", True),
        ("output_key", "state"),
        ("include_contents", "none"),
        ("mode", "task"),
        ("mode", None),
        ("generate_content_config", types.GenerateContentConfig(response_modalities=["AUDIO"])),
        (
            "generate_content_config",
            types.GenerateContentConfig(response_mime_type="application/json"),
        ),
    ],
)
def test_unsupported_profile_rejects_before_manifest_io(field: str, value: object) -> None:
    case = native_case()
    setattr(case.agent, field, value)
    with pytest.raises((TypeError, ValueError)):
        controlled(case, policy_file=Path("/missing-adk-manifest.yaml"))
    assert case.model.seen == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("is_long_running", True),
        ("custom_metadata", {"unguarded": True}),
        ("_require_confirmation", True),
        ("_defers_response", True),
        ("_context_param_name", "value"),
        ("_ignore_params", []),
    ],
)
def test_unguarded_native_tool_behaviors_reject(field: str, value: object) -> None:
    case = native_case(use_tool=True)
    function = profile(case.runner).tools[0].function
    assert inspect.isfunction(function)
    tool = FunctionTool(function)
    setattr(tool, field, value)
    case.agent.tools = [tool]
    with pytest.raises(ValueError):
        controlled(case, policy_file=Path("/missing-adk-manifest.yaml"))


def test_sync_injected_and_untyped_tools_reject_before_manifest_io() -> None:
    def sync_tool(value: str) -> str:
        return value

    async def injected(tool_context: object) -> str:
        return "uncovered"

    async def untyped(value: object) -> str:
        return str(value)

    for tool in (sync_tool, injected, untyped):
        case = native_case()
        case.agent.tools = [tool]
        with pytest.raises((TypeError, ValueError)):
            controlled(case, policy_file=Path("/missing-adk-manifest.yaml"))


def test_plugins_and_persistence_reject_before_policy_io() -> None:
    case = native_case()
    plugin = BasePlugin(name="application_plugin")
    runner = Runner(
        app=App(name="caller", root_agent=case.agent, plugins=[plugin]),
        session_service=InMemorySessionService(),
    )
    with pytest.raises(ValueError, match="Plugins"):
        profile(runner)
    with pytest.raises(TypeError):
        create_backend(
            runtime=object(),
            interceptors=None,
            policy_file=Path("/missing.yaml"),
            agents=None,
            agent_policies=(),
            resolver=None,
            timeout=5,
            emit_timeout=30,
            max_records=1000,
        )


def test_foreign_duplicate_and_missing_actor_bindings_reject_before_policy_io() -> None:
    case = native_case()
    for bindings in (
        {"foreign": native_case().agent},
        {"first": case.agent, "duplicate": case.agent},
    ):
        with pytest.raises(ValueError):
            controlled(case, policy_file=Path("/missing.yaml"), agents=bindings)
    with pytest.raises(ValueError):
        controlled(
            case,
            policy_file=Path("/missing.yaml"),
            agent_policies=[
                AgentPolicy("missing", interceptors=(Allow(),)),
            ],
        )


def test_sequential_actual_actor_ownership_and_independent_sdk_sequences() -> None:
    async def scenario() -> None:
        first, second = native_case(), native_case()
        second.agent.name = "second_actor"
        root = SequentialAgent(  # pyright: ignore[reportDeprecated]
            name="trusted_sequence",
            sub_agents=[first.agent, second.agent],
        )
        runner = Runner(
            agent=root,
            app_name="caller",
            session_service=InMemorySessionService(),
        )
        adapter = create_backend(
            runtime=runner,
            interceptors=[Allow()],
            policy_file=None,
            agents={"first": first.agent, "second": second.agent},
            agent_policies=(),
            resolver=None,
            timeout=5,
            emit_timeout=30,
            max_records=1000,
        )
        result = await adapter.arun(RunRequest({"prompt": "safe"}))
        assert result.output == "native answer"
        assert len(first.model.seen) == len(second.model.seen) == 1
        groups: dict[str, list[int]] = {}
        for record in result.records:
            groups.setdefault(record.session_id, []).append(record.sequence)
        assert len(groups) == 3
        assert all(sequence == list(range(len(sequence))) for sequence in groups.values())
        assert root.sub_agents == [first.agent, second.agent]
        assert first.agent.model is first.model and second.agent.model is second.model

    asyncio.run(scenario())


def test_actor_node_cap_actual_native_tree_at_threshold() -> None:
    actors: list[BaseAgent] = [
        LlmAgent(
            name=f"actor_{index}",
            mode="chat",
            model=native_case().model,
        )
        for index in range(15)
    ]
    root = SequentialAgent(name="sequence", sub_agents=actors)  # pyright: ignore[reportDeprecated]
    runner = Runner(agent=root, app_name="caller", session_service=InMemorySessionService())
    assert len(profile(runner).actors) == 16
    extra = LlmAgent(name="actor_16", mode="chat", model=native_case().model)
    extra.parent_agent = root
    root.sub_agents.append(extra)
    with pytest.raises(ValueError, match="16"):
        profile(runner)
