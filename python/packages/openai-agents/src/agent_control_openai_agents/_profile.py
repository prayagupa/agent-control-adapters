"""Exact-version, closed native profile validated before policy I/O and effects."""

from __future__ import annotations

import inspect
import json
import math
import weakref
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import cast, get_origin

from agent_control_adapters.contracts import validate_agent_id
from agent_control_adapters.json import JSON, normalize, object_value
from agent_control_adapters.runtime import validate_topology
from agents import Agent, FunctionTool, ModelSettings
from agents.models.interface import Model
from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator

FUNCTION_TOOL_FIELDS = frozenset(
    {
        "name",
        "description",
        "params_json_schema",
        "on_invoke_tool",
        "strict_json_schema",
        "is_enabled",
        "tool_input_guardrails",
        "tool_output_guardrails",
        "needs_approval",
        "timeout_seconds",
        "timeout_behavior",
        "timeout_error_function",
        "defer_loading",
        "custom_data_extractor",
        "allowed_callers",
        "output_json_schema",
        "_output_type_adapter",
        "_failure_error_function",
        "_use_default_failure_error_function",
        "_is_agent_tool",
        "_agent_tool_default_identity",
        "_is_codex_tool",
        "_agent_instance",
        "_tool_namespace",
        "_tool_namespace_description",
        "_mcp_title",
        "_tool_origin",
        "_emit_tool_origin",
        "_mcp_tool_binding",
    }
)
AGENT_FIELDS = frozenset(
    {
        "name",
        "handoff_description",
        "tools",
        "mcp_servers",
        "mcp_config",
        "instructions",
        "prompt",
        "handoffs",
        "model",
        "model_settings",
        "input_guardrails",
        "output_guardrails",
        "output_type",
        "hooks",
        "tool_use_behavior",
        "reset_tool_choice",
    }
)
MODEL_SETTINGS_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "tool_choice",
        "parallel_tool_calls",
        "truncation",
        "max_tokens",
        "reasoning",
        "verbosity",
        "metadata",
        "store",
        "prompt_cache_retention",
        "include_usage",
        "response_include",
        "top_logprobs",
        "extra_query",
        "extra_body",
        "extra_headers",
        "extra_args",
        "retry",
        "context_management",
        "prompt_cache_options",
        "preserve_raw_usage",
        "timeout",
    }
)
_SETTINGS_ALLOWED = frozenset(
    {
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "tool_choice",
        "parallel_tool_calls",
        "max_tokens",
        "include_usage",
        "timeout",
    }
)
_TOOL_NONE = frozenset(
    {
        "timeout_seconds",
        "timeout_error_function",
        "custom_data_extractor",
        "allowed_callers",
        "output_json_schema",
        "_output_type_adapter",
        "_failure_error_function",
        "_agent_tool_default_identity",
        "_agent_instance",
        "_tool_namespace",
        "_tool_namespace_description",
        "_mcp_title",
        "_tool_origin",
        "_mcp_tool_binding",
    }
)


def _exact_fields(value: object, expected: frozenset[str]) -> None:
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError("An ordinary native dataclass instance is required")
    instance_fields = frozenset(vars(value))
    if type(value) is Agent and "__orig_class__" in instance_fields:
        if get_origin(value.__orig_class__) is not Agent:
            raise ValueError("Invalid native generic type metadata")
        instance_fields -= {"__orig_class__"}
    if type(value) is Agent and "_function_approval_identity" in instance_fields:
        identity: object = value._function_approval_identity
        if not isinstance(identity, tuple) or len(identity) != 2:
            raise ValueError("Invalid SDK-owned function approval identity")
        witness: object = identity[0]
        token: object = identity[1]
        if (
            not isinstance(witness, weakref.ReferenceType)
            or cast("weakref.ReferenceType[object]", witness)() is not value
            or not isinstance(token, str)
            or len(token) != 32
            or any(character not in "0123456789abcdef" for character in token)
        ):
            raise ValueError("Invalid SDK-owned function approval identity")
        instance_fields -= {"_function_approval_identity"}
    if frozenset(field.name for field in fields(value)) != expected or instance_fields != expected:
        raise ValueError("Unsupported pinned native fields or instance extensions")


def validate_settings(settings: ModelSettings) -> dict[str, JSON]:
    """Reject retries, storage, remote configuration, rich output and parallel calls."""
    if type(settings) is not ModelSettings:
        raise TypeError("An ordinary native ModelSettings is required")
    _exact_fields(settings, MODEL_SETTINGS_FIELDS)
    data = object_value(asdict(settings))
    for name, value in data.items():
        if name not in _SETTINGS_ALLOWED and value is not None:
            raise ValueError(f"Unsupported model setting: {name}")
    if settings.parallel_tool_calls is not None and settings.parallel_tool_calls is not False:
        raise ValueError("Parallel native tool calls are unsupported")
    if settings.tool_choice not in (None, "auto", "none", "required"):
        raise ValueError("Unsupported native tool choice")
    if settings.include_usage is not None and type(settings.include_usage) is not bool:
        raise TypeError("include_usage must be boolean")
    if settings.max_tokens is not None and (
        type(settings.max_tokens) is not int or not 1 <= settings.max_tokens <= 4096
    ):
        raise ValueError("max_tokens must be an integer in 1..4096")
    for name in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "timeout"):
        number = data[name]
        if number is not None and (
            type(number) not in (int, float)
            or not isinstance(number, int | float)
            or not math.isfinite(number)
        ):
            raise ValueError("Native numeric settings must be finite numbers")
    if settings.timeout is not None and not 0 < settings.timeout <= 120:
        raise ValueError("Native model-attempt timeout must be in (0, 120]")
    return data


def _schema(value: object) -> dict[str, JSON]:
    schema = object_value(value)
    Draft202012Validator.check_schema(schema)

    allowed = {
        "type",
        "title",
        "description",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
    }

    def visit(node: dict[str, JSON]) -> None:
        if set(node) - allowed:
            raise ValueError("Referenced, union, defaulted and executable-pattern schemas reject")
        kind = node.get("type")
        if kind not in ("object", "array", "string", "integer", "number", "boolean", "null"):
            raise ValueError("Every tool schema node requires a supported explicit type")
        if kind == "object":
            properties = object_value(node.get("properties", {}))
            required = node.get("required", [])
            if (
                node.get("additionalProperties") is not False
                or not isinstance(required, list)
                or any(not isinstance(key, str) for key in required)
                or len(required) != len(properties)
                or set(required) != set(properties)
            ):
                raise ValueError("Every object field must be required; extra fields must reject")
            for child in properties.values():
                visit(object_value(child))
        elif kind == "array":
            visit(object_value(node.get("items")))

    visit(schema)
    if schema.get("type") != "object":
        raise ValueError("Tool arguments require an object schema")
    return schema


def _validator(schema: dict[str, JSON]) -> Validator:
    return Draft202012Validator(schema)


def validate_arguments(schema: dict[str, JSON], value: object) -> dict[str, JSON]:
    """Validate original and changed JSON without dropping fields or coercing numbers."""
    arguments = object_value(value)
    error = next(_validator(schema).iter_errors(arguments), None)
    if error is not None:
        raise error

    def numeric(node: dict[str, JSON], item: JSON) -> None:
        kind = node.get("type")
        if kind == "integer" and type(item) is not int:
            raise ValueError("Integer arguments cannot be coerced")
        if kind == "number" and type(item) is not float:
            raise ValueError("Floating-point arguments cannot be coerced")
        if isinstance(item, dict):
            properties = object_value(node.get("properties", {}))
            for key, child in item.items():
                numeric(object_value(properties[key]), child)
        elif isinstance(item, list):
            child_schema = object_value(node.get("items"))
            for child in item:
                numeric(child_schema, child)

    numeric(schema, arguments)
    return arguments


@dataclass(frozen=True)
class ToolSpec:
    """Strong ownership of a validated callback and an independent strict JSON schema."""

    tool: FunctionTool
    schema: dict[str, JSON]
    stamp: str


def tool_spec(value: object) -> ToolSpec:
    """Inventory every public/private field; reject all nonlocal execution modes."""
    if type(value) is not FunctionTool:
        raise TypeError("Only ordinary local FunctionTool instances are supported")
    tool = value
    _exact_fields(tool, FUNCTION_TOOL_FIELDS)
    validate_agent_id(tool.name)
    if not isinstance(cast("object", tool.description), str):
        raise ValueError("Tool descriptions must be plain text")
    normalize(tool.description)
    if not (
        inspect.iscoroutinefunction(tool.on_invoke_tool)
        or inspect.iscoroutinefunction(
            inspect.getattr_static(type(tool.on_invoke_tool), "__call__", None)
        )
    ):
        raise ValueError("Local tool callbacks must be asynchronous")
    if tool.strict_json_schema is not True or tool.is_enabled is not True:
        raise ValueError("Tools must have strict schemas and be statically enabled")
    if tool.needs_approval is not False or tool.defer_loading is not False:
        raise ValueError("Native approval and deferred loading are unsupported")
    for guardrails in (tool.tool_input_guardrails, tool.tool_output_guardrails):
        if guardrails is not None and (type(guardrails) is not list or guardrails):
            raise ValueError("Native tool guardrails are unsupported")
    for name in _TOOL_NONE:
        if getattr(tool, name) is not None:
            raise ValueError(f"Unsupported FunctionTool mode: {name}")
    for name in ("_is_agent_tool", "_is_codex_tool"):
        if getattr(tool, name) is not False:
            raise ValueError(f"Unsupported FunctionTool mode: {name}")
    for name in ("_use_default_failure_error_function", "_emit_tool_origin"):
        if type(getattr(tool, name)) is not bool:
            raise ValueError(f"Invalid FunctionTool flag: {name}")
    if tool.timeout_behavior not in ("error_as_result", "raise_exception"):
        raise ValueError("Unsupported native tool timeout behavior")
    schema = _schema(tool.params_json_schema)
    wrapped: object = getattr(tool, "__wrapped__", None)
    if wrapped is not None:
        if not callable(wrapped) or not inspect.iscoroutinefunction(wrapped):
            raise ValueError("Synchronous/delegating decorated tools are unsupported")
        signature = inspect.signature(cast(Callable[..., object], wrapped))
        for parameter in signature.parameters.values():
            if (
                parameter.kind
                not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                or parameter.default is not inspect.Parameter.empty
                or parameter.annotation
                not in (str, int, float, bool, "str", "int", "float", "bool")
            ):
                raise ValueError("Decorated tools require required primitive parameters")
        if set(signature.parameters) != set(object_value(schema.get("properties", {}))):
            raise ValueError("Decorated tool schema must match its actual parameters")
        properties = object_value(schema.get("properties", {}))
        primitive_types = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
        }
        for name, parameter in signature.parameters.items():
            if object_value(properties[name]).get("type") != primitive_types[parameter.annotation]:
                raise ValueError(
                    "Decorated tool schema cannot contradict executable parameter types"
                )
        if vars(tool)["_use_default_failure_error_function"]:
            raise ValueError("Decorated tools must use failure_error_function=None")
    data = {
        name: normalize(getattr(tool, name))
        for name in FUNCTION_TOOL_FIELDS
        if name != "on_invoke_tool"
    }
    data["on_invoke_tool"] = id(tool.on_invoke_tool)
    return ToolSpec(tool, schema, json.dumps(data, sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class Profile:
    """Caller-owned objects and a captured profile fingerprint, never copied clients."""

    agent: Agent[object]
    model: Model
    tools: tuple[ToolSpec, ...]
    stamp: str


def profile(runtime: object) -> Profile:
    """Validate all native modes and ownership before loading policy manifests."""
    if type(runtime) is not Agent:
        raise TypeError("runtime must be an existing native agents.Agent")
    agent = cast("Agent[object]", runtime)
    _exact_fields(agent, AGENT_FIELDS)
    validate_agent_id(agent.name)
    if not isinstance(agent.model, Model):
        raise TypeError("An explicit public Model instance is required; strings/defaults reject")
    if agent.instructions is not None and not isinstance(agent.instructions, str):
        raise ValueError("Only str/None instructions are supported")
    normalize(agent.instructions)
    if agent.prompt is not None:
        raise ValueError("Stored or dynamic Agent.prompt is unsupported")
    if (
        any(
            type(value) is not list
            for value in (
                agent.handoffs,
                agent.mcp_servers,
                agent.input_guardrails,
                agent.output_guardrails,
            )
        )
        or type(cast("object", agent.mcp_config)) is not dict
    ):
        raise ValueError("Native lists/config must have their ordinary pinned types")
    if (
        agent.handoff_description is not None
        or agent.handoffs
        or agent.mcp_servers
        or agent.mcp_config
        or agent.input_guardrails
        or agent.output_guardrails
        or agent.output_type is not None
        or agent.hooks is not None
        or agent.tool_use_behavior != "run_llm_again"
        or agent.reset_tool_choice is not True
    ):
        raise ValueError("Native hooks, guardrails, handoffs, MCP and rich output are unsupported")
    if type(agent.tools) is not list or len(agent.tools) > 16:
        raise ValueError("The local tool list must contain at most 16 tools")
    specs = tuple(tool_spec(tool) for tool in agent.tools)
    if len({spec.tool.name for spec in specs}) != len(specs):
        raise ValueError("Local tool names must be unique")
    validate_topology(actors=1, nodes=1, edges=0)
    data = {
        "name": agent.name,
        "instructions": agent.instructions,
        "model": id(agent.model),
        "settings": validate_settings(agent.model_settings),
        "tools": [{"identity": id(spec.tool), "stamp": spec.stamp} for spec in specs],
    }
    return Profile(agent, agent.model, specs, json.dumps(data, sort_keys=True, allow_nan=False))


def revalidate(native: Profile) -> None:
    """Reject caller/native mutation before any later governed effect."""
    if profile(native.agent).stamp != native.stamp:
        raise ValueError("The captured native Agent/Model/FunctionTool profile changed")
