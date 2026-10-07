"""Lossless codecs and explicit native transport/tool profile shared by graph adapters."""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from agent_hooks_adapters.codecs import model_text, text_messages, text_value, wire_equal
from agent_hooks_adapters.json import JSON, normalize, object_value
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from ._inspection import attribute


@dataclass(frozen=True)
class ToolProfile:
    """Retain a local tool, its exact schema and explicit sync/async implementations."""

    tool: StructuredTool
    schema: type[BaseModel]
    schema_json: dict[str, JSON]
    func: object
    coroutine: object
    name: str
    description: str

    def revalidate(self) -> None:
        """Reject application mutation before any governed effect."""
        if (
            self.tool.name != self.name
            or self.tool.args_schema is not self.schema
            or self.tool.func is not self.func
            or self.tool.coroutine is not self.coroutine
            or self.tool.callbacks is not None
            or self.tool.return_direct
            or self.tool.handle_tool_error
            or self.tool.handle_validation_error
            or self.tool.response_format != "content"
            or self.tool.description != self.description
            or self.tool.verbose
            or self.tool.tags
            or self.tool.metadata
            or not wire_equal(self.schema.model_json_schema(), self.schema_json)
        ):
            raise ValueError("Native tool configuration changed")

    def arguments(self, value: object) -> dict[str, JSON]:
        """Validate raw arguments before a native parser can drop or coerce fields."""
        self.revalidate()
        arguments = object_value(value)
        parsed = self.schema.model_validate(arguments, strict=True)
        if not wire_equal(parsed.model_dump(mode="json", exclude_unset=True), arguments):
            raise ValueError("Native tool arguments cannot be dropped or coerced")
        return arguments


def tools_profile(tools: Sequence[BaseTool]) -> tuple[ToolProfile, ...]:
    """Accept only local StructuredTools with both explicit implementations."""
    if (
        isinstance(tools, str | bytes)
        or not isinstance(cast("object", tools), Sequence)
        or len(tools) > 64
    ):
        raise ValueError("Provide a bounded native tool sequence")
    result: list[ToolProfile] = []
    for tool in tools:
        if type(tool) is not StructuredTool:
            raise ValueError("Only concrete local StructuredTool objects are supported")
        schema = tool.args_schema
        if (
            not isinstance(schema, type)
            or not issubclass(schema, BaseModel)
            or not callable(tool.func)
            or inspect.iscoroutinefunction(tool.func)
            or not inspect.iscoroutinefunction(tool.coroutine)
            or tool.callbacks is not None
            or tool.return_direct
            or tool.handle_tool_error
            or tool.handle_validation_error
            or tool.response_format != "content"
            or tool.verbose
            or tool.tags
            or tool.metadata
        ):
            raise ValueError("Tools require explicit sync/async bodies and a plain schema")
        if any(item.name == tool.name or item.tool is tool for item in result):
            raise ValueError("Native tools must have unique names and identities")
        hints = inspect.signature(tool.func).parameters
        if any(
            parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
            or "Injected" in str(parameter.annotation)
            or "Runtime" in str(parameter.annotation)
            for parameter in hints.values()
        ):
            raise ValueError("Injected, variadic and runtime-dependent tools are unsupported")
        if set(hints) != set(schema.model_fields):
            raise ValueError("Every native tool parameter must be an explicit schema field")
        result.append(
            ToolProfile(
                tool,
                schema,
                object_value(schema.model_json_schema()),
                tool.func,
                tool.coroutine,
                tool.name,
                tool.description,
            )
        )
    return tuple(result)


def model_profile(model: object) -> BaseChatModel:
    """Require an explicit model with native async I/O, not an executor fallback."""
    if not isinstance(model, BaseChatModel):
        raise ValueError("An explicit BaseChatModel is required")
    if (
        attribute(type(model), "_generate") is attribute(BaseChatModel, "_generate")
        or attribute(type(model), "_agenerate") is attribute(BaseChatModel, "_agenerate")
        or not inspect.iscoroutinefunction(attribute(type(model), "_agenerate"))
        or model.callbacks is not None
        or model.verbose
        or model.tags
        or (model.metadata is not None and set(model.metadata) != {"lc_versions"})
        or model.cache not in (None, False)
        or model.rate_limiter is not None
        or attribute(type(model), "invoke") is not attribute(BaseChatModel, "invoke")
        or attribute(type(model), "ainvoke") is not attribute(BaseChatModel, "ainvoke")
        or getattr(model, "streaming", False)
        or getattr(model, "model_kwargs", {})
        or getattr(model, "extra_body", None) is not None
    ):
        raise ValueError("Models require explicit sync/async transport and no callbacks/cache")
    return model


_MODEL_SETTINGS = (
    "name",
    "metadata",
    "disable_streaming",
    "output_version",
    "model_name",
    "model",
    "temperature",
    "openai_api_base",
    "base_url",
    "request_timeout",
    "timeout",
    "max_retries",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "streaming",
    "stream_usage",
    "model_kwargs",
    "extra_body",
    "default_headers",
    "default_query",
    "reasoning_effort",
    "reasoning",
    "verbosity",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "top_p",
    "logprobs",
)
_MODEL_OWNERS = (
    "client",
    "async_client",
    "root_client",
    "root_async_client",
    "http_client",
    "http_async_client",
    "openai_api_key",
    "api_key",
    "custom_get_token_ids",
)
_MODEL_METHODS = ("_generate", "_agenerate", "bind_tools", "invoke", "ainvoke")


@dataclass(frozen=True)
class ModelProof:
    """Capture executable SDK settings without including mutable transport observations."""

    model: BaseChatModel
    settings: dict[str, JSON]
    owners: tuple[tuple[str, object], ...]
    methods: tuple[tuple[str, object], ...]

    @classmethod
    def capture(cls, model: BaseChatModel) -> ModelProof:
        """Own a detached non-secret settings snapshot and strong client/method identities."""
        model_profile(model)
        settings = object_value(
            {
                name: attribute(model, name)
                for name in _MODEL_SETTINGS
                if name in type(model).model_fields
            }
        )
        owners = tuple(
            (name, attribute(model, name))
            for name in _MODEL_OWNERS
            if name in type(model).model_fields
        )
        methods = tuple((name, attribute(type(model), name)) for name in _MODEL_METHODS)
        return cls(model, settings, owners, methods)

    def revalidate(self) -> None:
        """Reject settings, client ownership or executable transport mutation before effects."""
        model_profile(self.model)
        settings = {
            name: attribute(self.model, name)
            for name in _MODEL_SETTINGS
            if name in type(self.model).model_fields
        }
        if (
            not wire_equal(settings, self.settings)
            or any(attribute(self.model, name) is not owner for name, owner in self.owners)
            or any(attribute(type(self.model), name) is not method for name, method in self.methods)
            or any(name in vars(self.model) for name in _MODEL_METHODS)
        ):
            raise ValueError("Native model configuration changed")


def model_identity(model: BaseChatModel) -> str:
    """Use only application-owned model metadata, never model-produced authority."""
    value: object = getattr(model, "model_name", None) or getattr(model, "model", None)
    return text_value(value if isinstance(value, str) else type(model).__name__)


def messages_value(value: object) -> list[AnyMessage]:
    """Require native text messages, rejecting custom/rich/remove-message surfaces."""
    if not isinstance(value, list) or not value:
        raise ValueError("Native messages must be a nonempty list")
    messages: list[AnyMessage] = []
    for item in cast("list[object]", value):
        if type(item) not in (HumanMessage, SystemMessage, AIMessage, ToolMessage):
            raise ValueError("Unsupported native message class")
        if not isinstance(item, HumanMessage | SystemMessage | AIMessage | ToolMessage):
            raise ValueError("Unsupported native message")
        if not isinstance(item.content, str):
            raise ValueError("Only plain text native messages are supported")
        messages.append(item)
    return messages


def project_messages(messages: list[AnyMessage]) -> list[dict[str, JSON]]:
    """Project text while preserving roles, native IDs and tool routing."""
    result: list[dict[str, JSON]] = []
    for message in messages_value(messages):
        role = {"human": "user", "ai": "assistant"}.get(message.type, message.type)
        item: dict[str, JSON] = {"role": role, "content": text_value(message.content)}
        if message.id is not None:
            item["id"] = message.id
        if message.name is not None:
            item["name"] = message.name
        if isinstance(message, AIMessage):
            if message.invalid_tool_calls:
                raise ValueError("Invalid native tool calls are unsupported")
            item["tool_calls"] = normalize(message.tool_calls)
        if isinstance(message, ToolMessage):
            item["tool_call_id"] = message.tool_call_id
            item["status"] = message.status
        result.append(item)
    return result


def replace_messages(value: object, messages: list[AnyMessage]) -> list[AnyMessage]:
    """Write text-only transforms back into detached native message copies."""
    target = text_messages(value, project_messages(messages))
    return [
        message.model_copy(update={"content": replacement["content"]})
        for message, replacement in zip(messages, target, strict=True)
    ]


def model_value(message: object, tools: tuple[ToolProfile, ...]) -> dict[str, JSON]:
    """Validate the actual AIMessage before any native tool parser runs."""
    if type(message) is not AIMessage:
        raise ValueError("Exactly one concrete AIMessage is required")
    if not isinstance(message.content, str) or message.invalid_tool_calls:
        raise ValueError("Rich, null or invalid native model output is unsupported")
    calls: list[JSON] = []
    for call in message.tool_calls:
        if set(call) != {"id", "name", "args", "type"} or call.get("type") != "tool_call":
            raise ValueError("Unsupported native tool-call fields")
        identifier = call["id"]
        if not isinstance(identifier, str) or not 1 <= len(identifier) <= 128:
            raise ValueError("A bounded native tool-call ID is required")
        tool = next((item for item in tools if item.name == call["name"]), None)
        if tool is None:
            raise ValueError("Model requested an unregistered tool")
        arguments = tool.arguments(call["args"])
        calls.append({"id": identifier, "name": tool.name, "args": arguments})
    if len(calls) > 1:
        raise ValueError("Only one tool call per model output is supported")
    finish: object = message.response_metadata.get("finish_reason")
    if finish is None:
        finish = "tool_calls" if calls else "stop"
    if not isinstance(finish, str):
        raise ValueError("Unsupported native finish reason")
    if message.usage_metadata is not None:
        normalize(message.usage_metadata)
    return {"content": message.content, "tool_calls": calls, "finish_reason": finish}


def replace_model(value: object, message: AIMessage, tools: tuple[ToolProfile, ...]) -> AIMessage:
    """Change content only; preserve native usage, call IDs and response metadata."""
    target = model_text(value, model_value(message, tools))
    return message.model_copy(update={"content": target["content"]})


def call_value(value: object, tools: tuple[ToolProfile, ...]) -> tuple[ToolCall, ToolProfile]:
    """Validate raw native tool arguments before invoking the actual tool."""
    call = object_value(value)
    if set(call) != {"id", "name", "args", "type"} or call["type"] != "tool_call":
        raise ValueError("Unsupported executable tool call")
    identifier, name = call["id"], call["name"]
    if not isinstance(identifier, str) or not isinstance(name, str):
        raise ValueError("Tool identity must be native text")
    tool = next((item for item in tools if item.name == name), None)
    if tool is None:
        raise ValueError("Unknown executable native tool")
    arguments = tool.arguments(call["args"])
    return {"id": identifier, "name": name, "args": arguments, "type": "tool_call"}, tool


def tool_content(value: object) -> str:
    """Render a strict JSON tool result without accepting Command or rich artifacts."""
    result = normalize(value)
    return result if isinstance(result, str) else json.dumps(result, allow_nan=False)


def final_message(value: object) -> AIMessage:
    """Require a completed text assistant response with no pending tool calls."""
    state = value
    if not isinstance(state, dict) or "messages" not in state:
        raise ValueError("Native graph returned an unsupported state")
    messages = messages_value(cast("dict[str, object]", state)["messages"])
    message = messages[-1]
    if not isinstance(message, AIMessage) or message.tool_calls or message.invalid_tool_calls:
        raise ValueError("Native graph did not finish with a plain assistant message")
    return message
