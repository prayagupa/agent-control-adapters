"""Lossless text-only executable projection for the pinned public Model interface."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from typing import cast

from agent_hooks_adapters.codecs import model_text, text_messages, text_value, wire_equal
from agent_hooks_adapters.json import JSON, normalize, object_value
from agents.items import ModelResponse, TResponseInputItem, TResponseOutputItem
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from pydantic import TypeAdapter

_INPUT_ADAPTER = TypeAdapter(list[TResponseInputItem])


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate raw JSON argument key")
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise ValueError("Non-finite raw JSON arguments are unsupported")


def raw_arguments(value: str) -> dict[str, JSON]:
    """Reject oversized, duplicate, malformed and nonportable raw JSON before SDK parsing."""
    if len(value) > 1_048_576 or len(value.encode("utf-8")) > 1_048_576:
        raise ValueError("Raw JSON argument byte limit exceeded")
    parsed: object = json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant)
    return object_value(parsed)


@dataclass(frozen=True)
class InputProjection:
    """System instructions remain a separate actual native parameter."""

    instructions: str | None
    items: list[dict[str, JSON]]
    messages: list[dict[str, JSON]]

    def rebuild(self, value: object) -> tuple[str | None, list[TResponseInputItem]]:
        """Apply text only and reconstruct real typed input without routing loss."""
        changed = text_messages(value, self.messages)
        offset = int(self.instructions is not None)
        instructions = text_value(changed[0]["content"]) if offset else None
        items: list[dict[str, JSON]] = []
        for item, message in zip(self.items, changed[offset:], strict=True):
            replacement = object_value(item)
            kind = item.get("type", "message")
            if kind == "message":
                original = item["content"]
                if isinstance(original, str):
                    replacement["content"] = text_value(message["content"])
                else:
                    if not isinstance(original, list) or len(original) != 1:
                        raise ValueError("Only one plain text message part is supported")
                    part = object_value(original[0])
                    part["text"] = text_value(message["content"])
                    replacement["content"] = [part]
            elif kind == "function_call_output":
                replacement["output"] = text_value(message["content"])
            items.append(replacement)
        typed = _INPUT_ADAPTER.validate_python(items, strict=True, extra="allow")
        if not wire_equal(typed, items):
            raise ValueError("Native input reconstruction cannot drop or coerce fields")
        return instructions, typed


def project_input(
    instructions: str | None, value: str | list[TResponseInputItem]
) -> InputProjection:
    """Project only supported native messages and preserve all non-text identity fields."""
    items: list[dict[str, JSON]] = (
        [{"role": "user", "content": text_value(value)}]
        if isinstance(value, str)
        else [object_value(item) for item in value]
    )
    messages: list[dict[str, JSON]] = []
    if instructions is not None:
        messages.append(
            {"role": "system", "content": text_value(instructions), "source": "system_instructions"}
        )
    for item in items:
        kind = item.get("type", "message")
        if kind == "message":
            if set(item) - {"role", "content", "type", "id", "status", "phase", "provider_data"}:
                raise ValueError("Extended native message fields are unsupported")
            _provider(item)
            if item.get("role") not in ("user", "assistant"):
                raise ValueError("Only native user/assistant history is supported")
            content = item.get("content")
            if isinstance(content, str):
                text = text_value(content)
                part_metadata: JSON = None
            elif isinstance(content, list) and len(content) == 1:
                part = object_value(content[0])
                if set(part) - {"type", "text", "annotations", "logprobs"}:
                    raise ValueError("Extended native text fields are unsupported")
                if part.get("type") not in ("input_text", "output_text"):
                    raise ValueError("Rich native message parts are unsupported")
                if part.get("annotations") or part.get("logprobs"):
                    raise ValueError("Annotated/logprob native text is unsupported")
                text = text_value(part.get("text"))
                part_metadata = {key: child for key, child in part.items() if key != "text"}
            else:
                raise ValueError("Only one plain text native message part is supported")
            messages.append(
                {
                    "role": item["role"],
                    "content": text,
                    "native": {key: child for key, child in item.items() if key != "content"},
                    "part": part_metadata,
                }
            )
        elif kind == "function_call":
            if set(item) - {
                "type",
                "call_id",
                "arguments",
                "name",
                "id",
                "status",
                "namespace",
                "caller",
                "async",
                "provider_data",
            }:
                raise ValueError("Extended native function call fields are unsupported")
            _provider(item)
            if (
                item.get("namespace") is not None
                or item.get("caller") is not None
                or item.get("async") is not None
            ):
                raise ValueError("Programmatic, namespaced and remote calls are unsupported")
            messages.append({"role": "assistant", "content": None, "native": item})
        elif kind == "function_call_output":
            if set(item) - {"type", "call_id", "output"}:
                raise ValueError("Extended native tool output fields are unsupported")
            if not isinstance(item.get("call_id"), str):
                raise ValueError("Invalid native tool output correlation")
            messages.append(
                {
                    "role": "tool",
                    "content": text_value(item.get("output")),
                    "tool_call_id": item["call_id"],
                    "native": {key: child for key, child in item.items() if key != "output"},
                }
            )
        else:
            raise ValueError("Unsupported native model input item")
    return InputProjection(instructions, items, messages)


def _provider(value: dict[str, JSON]) -> dict[str, JSON]:
    metadata = value.get("provider_data")
    if metadata is None:
        return {}
    provider = object_value(metadata)
    if set(provider) - {"model", "response_id", "finish_reason"} or any(
        not isinstance(child, str) for child in provider.values()
    ):
        raise ValueError("Unsupported provider metadata")
    return provider


@dataclass(frozen=True)
class OutputProjection:
    """Canonical target plus the actual typed response whose metadata must survive."""

    response: ModelResponse
    target: dict[str, JSON]
    calls: tuple[ResponseFunctionToolCall, ...]

    def rebuild(self, value: object) -> ModelResponse:
        """Return a real copied ModelResponse consumed by Runner, not a notification."""
        changed = model_text(value, self.target)
        output: list[TResponseOutputItem] = []
        for item in self.response.output:
            if isinstance(item, ResponseOutputMessage):
                part = cast(ResponseOutputText, item.content[0])
                text = text_value(changed["content"])
                if not text.strip():
                    raise ValueError("Empty native model text is unsupported")
                output.append(
                    item.model_copy(
                        deep=True,
                        update={"content": [part.model_copy(deep=True, update={"text": text})]},
                    )
                )
            else:
                output.append(item.model_copy(deep=True))
        return replace(
            self.response,
            output=output,
            usage=copy.deepcopy(self.response.usage),
            raw_usage=(
                object_value(self.response.raw_usage)
                if self.response.raw_usage is not None
                else None
            ),
        )


def project_output(response: ModelResponse) -> OutputProjection:
    """Reject empty/rich outputs; a single proven local function call may be textless."""
    if type(response) is not ModelResponse or not 1 <= len(response.output) <= 2:
        raise ValueError("Unsupported native ModelResponse")
    text: str | None = None
    finish_reason: str | None = None
    calls: list[ResponseFunctionToolCall] = []
    targets: list[JSON] = []
    for item in response.output:
        extras = object_value(item.model_extra or {})
        if set(extras) - {"provider_data"}:
            raise ValueError("Extended native model output fields are unsupported")
        provider = _provider(extras)
        native_finish = provider.get("finish_reason")
        if isinstance(native_finish, str):
            if finish_reason is not None and finish_reason != native_finish:
                raise ValueError("Conflicting native finish metadata")
            finish_reason = native_finish
        if type(item) is ResponseOutputMessage:
            if text is not None or item.status != "completed" or len(item.content) != 1:
                raise ValueError("Only one completed plain text model message is supported")
            part = item.content[0]
            if type(part) is not ResponseOutputText or part.annotations or part.logprobs:
                raise ValueError("Rich native model output is unsupported")
            if part.model_extra:
                raise ValueError("Extended native model output is unsupported")
            text = text_value(part.text)
            if not text.strip():
                raise ValueError("Empty native model text is unsupported")
        elif type(item) is ResponseFunctionToolCall:
            if (
                calls
                or item.namespace is not None
                or item.caller is not None
                or item.async_ is not None
                or item.status not in (None, "completed")
                or not 1 <= len(item.call_id) <= 128
            ):
                raise ValueError("Only one direct local function call is supported")
            calls.append(item)
            targets.append(
                {"id": item.call_id, "name": item.name, "args": raw_arguments(item.arguments)}
            )
        else:
            raise ValueError("Hosted, reasoning, multimodal and remote model outputs reject")
    if text is None and not calls:
        raise ValueError("Native output requires text or a supported function call")
    normalize(response.response_id)
    normalize(response.request_id)
    if response.raw_usage is not None:
        object_value(response.raw_usage)
    return OutputProjection(
        response,
        {
            "content": text,
            "tool_calls": targets,
            "finish_reason": finish_reason or ("tool_calls" if calls else "stop"),
        },
        tuple(calls),
    )
