"""ACS-only admission, real actor membership and exact pinned native profiles."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import cast

import pytest
from agent_control_adapters import AgentPolicy, Backend, RunRequest, with_agent_controls
from agent_control_adapters import bindings as bindings_module
from agent_control_adapters.acs import load_acs_policy
from agent_framework import Agent, ChatOptions, FunctionTool
from agent_hooks import ALLOW, AgentContext, Interceptor, Verdict
from maf_adapter_test_helpers import controlled, native_case, native_tool_pair


class PythonPolicy:
    """An ordinary Python interceptor, deliberately invalid for ACS-only MAF."""

    def intercept(self, context: AgentContext) -> Verdict:
        return ALLOW


def _forbidden_load(path: str | Path) -> Interceptor:
    pytest.fail("Invalid native admission must precede manifest I/O")


@pytest.mark.parametrize("per_agent", [False, True])
def test_acs_only_rejects_python_before_manifest_io(
    per_agent: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bindings_module, "load_acs_policy", _forbidden_load)

    async def scenario() -> None:
        case = native_case(monkeypatch)
        try:
            with pytest.raises(TypeError, match="native ACS"):
                with_agent_controls(
                    backend=Backend.MAF,
                    runtime=case.agent,
                    interceptors=None if per_agent else [PythonPolicy()],
                    policy_file="unopened.yaml" if per_agent else None,
                    agent_policies=[AgentPolicy("agent", (PythonPolicy(),))] if per_agent else (),
                )
            assert case.model.calls == 0
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("both", [False, True])
def test_source_xor_before_manifest_io(both: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bindings_module, "load_acs_policy", _forbidden_load)

    async def scenario() -> None:
        case = native_case(monkeypatch)
        try:
            with pytest.raises(ValueError, match="exactly one"):
                with_agent_controls(
                    backend=Backend.MAF,
                    runtime=case.agent,
                    interceptors=[PythonPolicy()] if both else None,
                    policy_file="unopened.yaml" if both else None,
                )
            assert case.model.calls == 0
        finally:
            await case.transport.close()

    asyncio.run(scenario())


def test_prebuilt_native_acs_and_trusted_binding(
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch)
        native = load_acs_policy(allow_manifest)
        try:
            adapter = with_agent_controls(
                backend=Backend.MAF,
                runtime=case.agent,
                interceptors=[native],
                agents={"trusted": case.agent},
                agent_policies=[AgentPolicy("trusted", (native,))],
            )
            result = await adapter.arun(RunRequest({"prompt": "Hello"}))
            assert result.output == "native response"
            assert all(record.interceptors_registered == 2 for record in result.records)
            assert len({record.session_id for record in result.records}) == 1
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("binding", ["foreign", "duplicate", "unknown_policy", "duplicate_policy"])
def test_foreign_duplicate_actual_bindings_before_io(
    binding: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bindings_module, "load_acs_policy", _forbidden_load)

    async def scenario() -> None:
        case = native_case(monkeypatch)
        agents: dict[str, object] = {"agent": case.agent}
        policies: tuple[AgentPolicy, ...] = ()
        if binding == "foreign":
            agents = {"agent": Agent[ChatOptions[None]](case.client)}
        elif binding == "duplicate":
            agents["alias"] = case.agent
        elif binding == "unknown_policy":
            policies = (AgentPolicy("unknown", policy_file="unopened.yaml"),)
        else:
            policy = AgentPolicy("agent", policy_file="unopened.yaml")
            policies = (policy, policy)
        try:
            with pytest.raises(ValueError, match=r"unique|actual"):
                with_agent_controls(
                    backend=Backend.MAF,
                    runtime=case.agent,
                    policy_file="unopened.yaml",
                    agents=agents,
                    agent_policies=policies,
                )
            assert case.model.calls == 0
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("middleware", [object()]),
        ("context_providers", [object()]),
        ("require_per_service_call_history_persistence", True),
        ("additional_properties", {"unsafe": True}),
        ("compaction_strategy", object()),
        ("tokenizer", object()),
    ],
)
def test_unsupported_native_profile_before_io(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bindings_module, "load_acs_policy", _forbidden_load)

    async def scenario() -> None:
        case = native_case(monkeypatch)
        setattr(case.agent, field, value)
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                controlled(case, Path("unopened.yaml"))
            assert case.model.calls == 0
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "option",
    [
        "response_format",
        "store",
        "previous_response_id",
        "conversation_id",
        "session_id",
        "messages",
        "stream",
        "prompt",
        "mcp_tools",
    ],
)
def test_stored_resume_stream_and_unsafe_options_rejected(
    option: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bindings_module, "load_acs_policy", _forbidden_load)

    async def scenario() -> None:
        case = native_case(monkeypatch)
        options = cast("dict[str, object]", case.agent.default_options)
        options[option] = "unsafe"
        try:
            with pytest.raises(ValueError, match="Unsupported MAF options"):
                controlled(case, Path("unopened.yaml"))
            assert case.model.calls == 0
        finally:
            await case.transport.close()

    asyncio.run(scenario())


def test_serial_invocation_is_explicit_not_native_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch, serial=False)
        try:
            with pytest.raises(ValueError, match="serial tools"):
                controlled(case, Path("unopened.yaml"))
            assert case.model.calls == 0
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("chat_middleware", [object()]),
        ("function_middleware", [object()]),
        ("additional_properties", {"unsafe": True}),
        ("compaction_strategy", object()),
        ("tokenizer", object()),
    ],
)
def test_unsafe_native_client_configuration_before_io(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bindings_module, "load_acs_policy", _forbidden_load)

    async def scenario() -> None:
        case = native_case(monkeypatch)
        setattr(case.agent.client, field, value)
        try:
            with pytest.raises(ValueError, match="Unsupported MAF client"):
                controlled(case, Path("unopened.yaml"))
            assert case.model.calls == 0
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("layer", ["agent", "client"])
def test_custom_native_pipeline_rejected_before_io(
    layer: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bindings_module, "load_acs_policy", _forbidden_load)

    async def scenario() -> None:
        case = native_case(monkeypatch)
        target = case.agent if layer == "agent" else case.agent.client
        name = "run" if layer == "agent" else "get_response"
        monkeypatch.setattr(target, name, object())
        try:
            with pytest.raises(ValueError, match="Custom native"):
                controlled(case, Path("unopened.yaml"))
            assert case.model.calls == 0
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", ""),
        ("func", None),
        ("approval_mode", "always_require"),
        ("kind", "hosted"),
        ("additional_properties", {"unsafe": True}),
        ("max_invocations", 1),
        ("max_invocation_exceptions", 1),
        ("result_parser", object()),
        ("_context_parameter_name", "context"),
        ("_schema_supplied", True),
    ],
)
def test_unsafe_tool_features_before_io(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bindings_module, "load_acs_policy", _forbidden_load)

    async def scenario() -> None:
        case = native_case(monkeypatch, use_tool=True)
        tools = case.agent.default_options["tools"]
        assert isinstance(tools, list) and isinstance(tools[0], FunctionTool)
        setattr(tools[0], field, value)
        try:
            with pytest.raises(ValueError, match="Unsupported native FunctionTool"):
                controlled(case, Path("unopened.yaml"))
            assert case.model.calls == 0
            assert case.effects == []
        finally:
            await case.transport.close()

    asyncio.run(scenario())


def test_duplicate_native_tools_before_io(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bindings_module, "load_acs_policy", _forbidden_load)

    async def scenario() -> None:
        case = native_case(monkeypatch, use_tool=True)
        tools = case.agent.default_options["tools"]
        assert isinstance(tools, list) and isinstance(tools[0], FunctionTool)
        case.agent.default_options["tools"] = [tools[0], tools[0]]
        try:
            with pytest.raises(ValueError, match="Unsupported native FunctionTool"):
                controlled(case, Path("unopened.yaml"))
            assert case.model.calls == 0
            assert case.effects == []
        finally:
            await case.transport.close()

    asyncio.run(scenario())


def test_actual_native_agent_as_tool_rejected_before_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bindings_module, "load_acs_policy", _forbidden_load)

    async def scenario() -> None:
        case = native_case(monkeypatch)
        case.agent.default_options["tools"] = [case.agent.as_tool(name="delegate")]
        try:
            with pytest.raises(ValueError, match="Unsupported native FunctionTool"):
                controlled(case, Path("unopened.yaml"))
            assert case.model.calls == 0
            assert case.effects == []
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("mutation", ["instructions", "serial", "tool_function", "tool_name_swap"])
def test_configuration_mutation_revalidated_before_effects(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch, use_tool=True)
        pair = native_tool_pair(case) if mutation == "tool_name_swap" else None
        adapter = controlled(case, allow_manifest)
        if mutation == "instructions":
            case.agent.default_options["instructions"] = "changed trusted instructions"
        elif mutation == "serial":
            case.client.function_invocation_configuration["allow_concurrent_invocation"] = True
        elif mutation == "tool_function":
            tools = case.agent.default_options["tools"]
            assert isinstance(tools, list) and isinstance(tools[0], FunctionTool)

            def replacement(value: str) -> str:
                return value

            tools[0].func = replacement
        else:
            assert pair is not None
            first, second = pair
            first.name, second.name = second.name, first.name
        try:
            with pytest.raises(ValueError):
                await adapter.arun(RunRequest({"prompt": "Call ping"}))
            assert case.effects == []
            assert case.model.calls == 0
            assert adapter.records == ()
            with pytest.raises(RuntimeError, match="single-use"):
                await adapter.arun(RunRequest({"prompt": "retry"}))
        finally:
            await case.transport.close()

    asyncio.run(scenario())


def test_unchanged_tool_names_keep_their_native_effects(
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch, use_tool=True)
        first, second = native_tool_pair(case)
        adapter = controlled(case, allow_manifest)
        try:
            result = await adapter.arun(RunRequest({"prompt": "Call ping"}))
            assert result.output == "native response"
            assert case.model.calls == 2
            assert case.effects == ["native"]
            assert first.name == "ping" and second.name == "other"
            assert adapter.cleanup_error is None
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("count", [64, 65])
def test_native_tool_inventory_exact_bound(
    count: int,
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch, use_tool=True)
        original = case.agent.default_options["tools"]
        assert isinstance(original, list) and isinstance(original[0], FunctionTool)
        tools: list[FunctionTool] = []
        for index in range(count):
            candidate = copy.copy(original[0])
            candidate.name = f"tool_{index}"
            tools.append(candidate)
        case.agent.default_options["tools"] = tools
        try:
            if count == 65:
                with pytest.raises(ValueError, match="bounded"):
                    controlled(case, allow_manifest)
                assert case.model.calls == 0
            else:
                controlled(case, allow_manifest)
        finally:
            await case.transport.close()

    asyncio.run(scenario())
