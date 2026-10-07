"""Closed native profile, every FunctionTool field, stored prompt/resume, and admission."""

from __future__ import annotations

import asyncio
from dataclasses import fields
from pathlib import Path

import pytest
from agent_adapter_conformance import INVALID_PROMPTS
from agent_control_adapters import AdapterBlocked, AgentPolicy, RunRequest
from agent_control_adapters.json import JSON
from agent_control_openai_agents._model import GovernedModel
from agent_control_openai_agents._profile import (
    FUNCTION_TOOL_FIELDS,
    profile,
    validate_arguments,
    validate_settings,
)
from agents import ModelSettings, function_tool
from agents.models.interface import ModelTracing
from jsonschema.exceptions import ValidationError
from openai_adapter_test_helpers import Policy, controlled, native_case, scope_of


def test_complete_pinned_function_tool_inventory() -> None:
    case = native_case(use_tool=True)
    assert frozenset(field.name for field in fields(case.tool)) == FUNCTION_TOOL_FIELDS
    assert len(FUNCTION_TOOL_FIELDS) == 29
    assert frozenset(vars(case.tool)) == FUNCTION_TOOL_FIELDS


@pytest.mark.parametrize(
    "name,value",
    [
        ("allowed_callers", ["direct"]),
        ("allowed_callers", ["programmatic"]),
        ("_is_agent_tool", True),
        ("_is_codex_tool", True),
        ("_agent_instance", object()),
        ("_agent_tool_default_identity", ("agent", "tool")),
        ("_tool_namespace", "remote"),
        ("_tool_namespace_description", "remote"),
        ("_mcp_title", "remote"),
        ("_tool_origin", {"kind": "mcp"}),
        ("_mcp_tool_binding", ("server", "tool", 0)),
        ("needs_approval", True),
        ("defer_loading", True),
        ("is_enabled", False),
        ("tool_input_guardrails", [object()]),
        ("tool_output_guardrails", [object()]),
        ("custom_data_extractor", object()),
        ("output_json_schema", {"type": "string"}),
        ("_output_type_adapter", object()),
        ("_failure_error_function", object()),
        ("strict_json_schema", False),
        ("timeout_seconds", 1.0),
        ("timeout_error_function", object()),
        ("timeout_behavior", "unsupported"),
        ("_emit_tool_origin", 1),
        ("_use_default_failure_error_function", 1),
    ],
)
def test_every_unsupported_function_tool_mode_before_policy_io(
    name: str, value: object, tmp_path: Path
) -> None:
    case = native_case(use_tool=True)
    setattr(case.tool, name, value)
    with pytest.raises((TypeError, ValueError)):
        controlled(case, path=tmp_path / "not-present.yaml")
    assert case.model.seen == [] and case.effects == []


@pytest.mark.parametrize(
    "name,value",
    [
        ("model", None),
        ("model", "remote-default"),
        ("instructions", lambda: "dynamic"),
        ("prompt", {"id": "stored"}),
        ("hooks", object()),
        ("handoffs", [object()]),
        ("mcp_servers", [object()]),
        ("mcp_config", {"convert_schemas_to_strict": True}),
        ("handoff_description", "delegation"),
        ("input_guardrails", [object()]),
        ("output_guardrails", [object()]),
        ("output_type", str),
        ("tool_use_behavior", "stop_on_first_tool"),
        ("reset_tool_choice", False),
        ("native_extension", object()),
    ],
)
def test_unsupported_agent_profile_zero_effects(name: str, value: object, tmp_path: Path) -> None:
    case = native_case(use_tool=True)
    setattr(case.agent, name, value)
    with pytest.raises((TypeError, ValueError)):
        controlled(case, path=tmp_path / "not-present.yaml")
    assert case.model.seen == [] and case.effects == []


@pytest.mark.parametrize(
    "name,value",
    [
        ("parallel_tool_calls", True),
        ("parallel_tool_calls", 0),
        ("retry", object()),
        ("context_management", [{"type": "compaction"}]),
        ("extra_body", {"prompt": "stored"}),
        ("extra_args", {"previous_response_id": "stored"}),
        ("store", True),
        ("reasoning", {"effort": "low"}),
        ("metadata", {"session_id": "caller"}),
        ("max_tokens", True),
        ("max_tokens", 4097),
    ],
)
def test_unsupported_settings_zero_effects(name: str, value: object, tmp_path: Path) -> None:
    case = native_case()
    setattr(case.agent.model_settings, name, value)
    with pytest.raises((TypeError, ValueError)):
        controlled(case, path=tmp_path / "not-present.yaml")
    assert case.model.seen == []


@pytest.mark.parametrize("model_attempt_timeout", [None, 60.0, 120.0])
def test_approved_model_settings_keep_native_timeout_default(
    model_attempt_timeout: float | None,
) -> None:
    settings = ModelSettings(temperature=0, max_tokens=64, parallel_tool_calls=False)
    assert settings.timeout is None and settings.retry is None
    settings.timeout = model_attempt_timeout
    validated = validate_settings(settings)
    assert validated["timeout"] == model_attempt_timeout
    assert validated["temperature"] == 0 and validated["max_tokens"] == 64


@pytest.mark.parametrize("model_attempt_timeout", [0.0, -1.0, 120.001])
def test_model_attempt_timeout_profile_bounds_preserved(model_attempt_timeout: float) -> None:
    settings = ModelSettings(parallel_tool_calls=False)
    settings.timeout = model_attempt_timeout
    with pytest.raises(ValueError, match=r"Native model-attempt timeout must be in \(0, 120\]"):
        validate_settings(settings)


def test_stored_prompt_and_resume_rejected(tmp_path: Path) -> None:
    case = native_case()
    case.agent.prompt = {"id": "stored"}
    with pytest.raises(ValueError, match="prompt"):
        controlled(case, path=tmp_path / "not-present.yaml")

    async def scenario(field: str) -> None:
        fresh = native_case()
        adapter = controlled(fresh, policies=[Policy()])
        scope = scope_of(adapter)
        scope.session.begin()
        agent = scope.clone()
        assert isinstance(agent.model, GovernedModel)
        primary: BaseException | None = None
        try:
            with pytest.raises(AdapterBlocked) as blocked:
                await agent.model.get_response(
                    "system",
                    "user",
                    ModelSettings(parallel_tool_calls=False),
                    [],
                    None,
                    [],
                    ModelTracing.DISABLED,
                    previous_response_id="stored" if field == "previous_response_id" else None,
                    conversation_id="stored" if field == "conversation_id" else None,
                    prompt={"id": "stored"} if field == "prompt" else None,
                )
            primary = blocked.value
            assert blocked.value.__cause__ is None
            assert fresh.model.seen == []
            assert isinstance(scope.session.last_error, ValueError)
        finally:
            await scope.session.aclose(primary)

    for field in ("previous_response_id", "conversation_id", "prompt"):
        asyncio.run(scenario(field))


@pytest.mark.parametrize("run_request", INVALID_PROMPTS)
def test_invalid_admitted_prompt_consumes_instance_zero_native_work(
    run_request: RunRequest | None,
) -> None:
    async def scenario() -> None:
        case = native_case()
        adapter = controlled(case, policies=[Policy()])
        with pytest.raises(ValueError):
            await adapter.arun(run_request)
        with pytest.raises(RuntimeError, match="single-use"):
            await adapter.arun(RunRequest({"prompt": "corrected"}))
        assert adapter.records == () and case.model.seen == []

    asyncio.run(scenario())


@pytest.mark.parametrize("length", [1, 65536])
def test_exact_prompt_length_boundaries(length: int) -> None:
    async def scenario() -> None:
        case = native_case()
        result = await controlled(case, policies=[Policy()]).arun(
            RunRequest({"prompt": "p" * length})
        )
        assert result.output == "native response"

    asyncio.run(scenario())


def test_run_refuses_before_admission_then_same_adapter_arun() -> None:
    case = native_case()
    adapter = controlled(case, policies=[Policy()])
    with pytest.raises(RuntimeError, match=r"await adapter.arun"):
        adapter.run(None)
    assert adapter.records == () and case.model.seen == []
    result = asyncio.run(adapter.arun(RunRequest({"prompt": "corrected"})))
    assert result.output == "native response"


def test_foreign_duplicate_missing_binding_before_policy_io(tmp_path: Path) -> None:
    case = native_case()
    for agents in ({"foreign": object()}, {"one": case.agent, "two": case.agent}):
        with pytest.raises(ValueError):
            controlled(case, path=tmp_path / "missing.yaml", agents=agents)
    with pytest.raises(ValueError):
        controlled(
            case,
            path=tmp_path / "missing.yaml",
            agent_policies=[AgentPolicy("missing", interceptors=(Policy(),))],
        )
    assert case.model.seen == []


def test_caller_mutation_rejected_at_admission_before_policy_lane() -> None:
    async def scenario() -> None:
        case = native_case()
        adapter = controlled(case, policies=[Policy()])
        case.agent.instructions = "mutated after construction"
        with pytest.raises(AdapterBlocked) as blocked:
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert isinstance(blocked.value.__cause__, ValueError)
        assert case.model.seen == [] and adapter.records == ()

    asyncio.run(scenario())


def test_decorated_async_primitive_function_native_execution() -> None:
    effects: list[str] = []

    @function_tool(failure_error_function=None)
    async def ping(value: str) -> str:
        """Echo a value."""
        effects.append(value)
        return value

    async def scenario() -> None:
        case = native_case(use_tool=True)
        case.agent.tools = [ping]
        result = await controlled(case, policies=[Policy()]).arun(RunRequest({"prompt": "safe"}))
        assert result.output == "native response" and effects == ["native"]

    asyncio.run(scenario())


def test_decorated_sync_tool_rejected_no_default_executor() -> None:
    @function_tool(failure_error_function=None)
    def ping(value: str) -> str:
        """Echo a value."""
        return value

    case = native_case(use_tool=True)
    case.agent.tools = [ping]
    with pytest.raises(ValueError, match="Synchronous"):
        profile(case.agent)


def test_schema_primitive_number_types_no_coercion_and_property_named_type() -> None:
    schema: dict[str, JSON] = {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "integer": {"type": "integer"},
            "number": {"type": "number"},
        },
        "required": ["type", "integer", "number"],
        "additionalProperties": False,
    }
    assert validate_arguments(schema, {"type": "plain", "integer": 1, "number": 1.0})
    for value in (
        {"type": "plain", "integer": 1.0, "number": 1.0},
        {"type": "plain", "integer": True, "number": 1.0},
        {"type": "plain", "integer": 1, "number": 1},
    ):
        with pytest.raises((TypeError, ValueError, ValidationError)):
            validate_arguments(schema, value)
