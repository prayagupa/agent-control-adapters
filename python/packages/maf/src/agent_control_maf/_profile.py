"""Pinned native pipeline admission and strong configuration fingerprints."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import cast

from agent_control_adapters.json import JSON, normalize, object_value
from agent_control_adapters.runtime import validate_topology
from agent_framework import Agent, BaseChatClient, ChatOptions, FunctionTool
from agent_framework._tools import FunctionInvocationLayer

NativeAgent = Agent[ChatOptions[None]]

_OPTION_FIELDS = {
    "instructions",
    "model",
    "temperature",
    "max_tokens",
    "seed",
    "top_p",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "tool_choice",
    "tools",
    "allow_multiple_tool_calls",
}


@dataclass(frozen=True)
class Profile:
    """A native agent, strongly retained tools and immutable configuration proof."""

    agent: NativeAgent
    client: object
    tools: tuple[FunctionTool, ...]
    functions: tuple[object, ...]
    fingerprint: str


def profile(runtime: object) -> Profile:
    """Reject unsupported persistence, delegation, middleware and output modes."""
    if type(runtime) is not Agent:
        raise TypeError("runtime must be a native Microsoft Agent Framework Agent")
    agent: NativeAgent = runtime
    if "run" in vars(agent):
        raise ValueError("Custom native run overrides are unsupported")
    for field in (
        "context_providers",
        "middleware",
        "agent_middleware",
        "mcp_tools",
        "compaction_strategy",
        "tokenizer",
        "additional_properties",
        "require_per_service_call_history_persistence",
    ):
        if getattr(agent, field, None):
            raise ValueError(f"Unsupported MAF agent configuration: {field}")
    client: object = agent.client
    if not isinstance(client, BaseChatClient | FunctionInvocationLayer) or not isinstance(
        client, FunctionInvocationLayer
    ):
        raise TypeError("MAF requires a native function-invocation chat client")
    if "get_response" in vars(client) or inspect.unwrap(
        type(client).get_response
    ) is not inspect.unwrap(FunctionInvocationLayer.get_response):
        raise ValueError("Custom native client pipeline overrides are unsupported")
    for field in (
        "chat_middleware",
        "function_middleware",
        "compaction_strategy",
        "tokenizer",
        "additional_properties",
    ):
        if getattr(client, field, None):
            raise ValueError(f"Unsupported MAF client configuration: {field}")
    config = object_value(cast("object", client.function_invocation_configuration))
    if (
        config.get("allow_concurrent_invocation") is not False
        or config.get("include_detailed_errors") is not False
        or config.get("additional_tools") not in (None, [])
        or config.get("disable_approval_response_binding") is True
        or config.get("enabled") is not True
    ):
        raise ValueError("MAF requires serial tools, redacted errors and no additional tools")
    raw_options = dict(agent.default_options)
    if set(raw_options) - _OPTION_FIELDS:
        raise ValueError("Unsupported MAF options, persistence or structured output")
    raw_tools: object = raw_options.pop("tools", [])
    if not isinstance(raw_tools, list):
        raise ValueError("MAF requires a bounded native FunctionTool list")
    captured_tools = cast("list[object]", raw_tools)
    if len(captured_tools) > 64:
        raise ValueError("MAF requires a bounded native FunctionTool list")
    tools: list[FunctionTool] = []
    names: set[str] = set()
    schemas: list[JSON] = []
    functions: list[object] = []
    for candidate in captured_tools:
        if type(candidate) is not FunctionTool:
            raise ValueError("Hosted, remote and agent-as-tool integrations are unsupported")
        tool = candidate
        if (
            not tool.name
            or tool.name in names
            or tool.func is None
            or tool.approval_mode != "never_require"
            or tool.kind not in (None, "function")
            or tool.additional_properties
            or tool.max_invocations is not None
            or tool.max_invocation_exceptions is not None
            or tool.result_parser is not None
            or getattr(tool, "_context_parameter_name", None) is not None
            or getattr(tool, "_schema_supplied", False)
        ):
            raise ValueError("Unsupported native FunctionTool configuration")
        names.add(tool.name)
        tools.append(tool)
        functions.append(tool.func)
        schemas.append(normalize(tool.input_model.model_json_schema() if tool.input_model else {}))
    options = object_value(raw_options)
    if not isinstance(options.get("model"), str) or not options["model"]:
        raise ValueError("An explicit native model is required")
    validate_topology(actors=1, nodes=1, edges=0)
    fingerprint = json.dumps(
        normalize(
            {
                "options": options,
                "config": config,
                "names": [tool.name for tool in tools],
                "schemas": schemas,
            }
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    return Profile(agent, client, tuple(tools), tuple(functions), fingerprint)


def revalidate(original: Profile) -> None:
    """Deny trusted application mutation before native work or later callbacks."""
    current = profile(original.agent)
    if (
        current.client is not original.client
        or len(current.tools) != len(original.tools)
        or any(
            first is not second for first, second in zip(current.tools, original.tools, strict=True)
        )
        or any(
            first is not second
            for first, second in zip(current.functions, original.functions, strict=True)
        )
        or current.fingerprint != original.fingerprint
    ):
        raise ValueError("Native MAF configuration changed after adapter construction")
