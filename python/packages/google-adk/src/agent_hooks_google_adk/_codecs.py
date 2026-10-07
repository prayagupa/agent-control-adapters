"""Lossless native text projection with strict, executable callback write-back."""

from __future__ import annotations

from dataclasses import dataclass

from agent_hooks_adapters.codecs import model_text, text_messages, text_value
from agent_hooks_adapters.json import JSON, normalize, object_value
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from ._profile import native_field


def plain(content: types.Content) -> str:
    """Require native plain text; never stringify null, media or function calls."""
    if not content.parts:
        raise ValueError("Native content requires nonempty text parts")
    texts: list[str] = []
    for part in content.parts:
        projected = object_value(part.model_dump(mode="json", exclude_none=True))
        if set(projected) != {"text"} or not isinstance(part.text, str):
            raise ValueError("Only plain native text is supported")
        texts.append(text_value(part.text))
    return text_value("".join(texts))


def _message(content: types.Content) -> dict[str, JSON]:
    if content.role not in ("user", "model", "system"):
        raise ValueError("Unsupported native message role")
    parts = content.parts
    if not parts:
        raise ValueError("Empty native model message")
    if all(set(part.model_dump(exclude_none=True)) == {"text"} for part in parts):
        return {"role": content.role, "content": plain(content)}
    projected: list[JSON] = []
    for part in parts:
        value = object_value(part.model_dump(mode="json", exclude_none=True))
        if set(value) not in ({"function_call"}, {"function_response"}):
            raise ValueError("Rich or mixed native model history is unsupported")
        projected.append(value)
    return {"role": content.role, "content": projected}


@dataclass(frozen=True)
class RequestProjection:
    """Capture typed native messages separately from the policy's detached JSON."""

    messages: list[dict[str, JSON]]
    contents: tuple[types.Content, ...]
    has_system: bool

    def apply(self, request: LlmRequest, value: object) -> object:
        """Replace actual request text/config while preserving native routing values."""
        messages = text_messages(value, self.messages)
        offset = 1 if self.has_system else 0
        config = request.config.model_copy(deep=True)
        if self.has_system:
            config.system_instruction = text_value(messages[0]["content"])
        contents: list[types.Content] = []
        for content, message in zip(self.contents, messages[offset:], strict=True):
            replacement = content.model_copy(deep=True)
            if isinstance(message["content"], str):
                replacement.parts = [types.Part(text=message["content"])]
            contents.append(replacement)
        request.config = config
        request.contents = contents
        return messages


def request_projection(request: LlmRequest) -> RequestProjection:
    """Project system placement and all native message IDs without inventing history."""
    if (
        request.previous_interaction_id is not None
        or request.cache_config is not None
        or request.cache_metadata is not None
        or request.service_tier is not None
    ):
        raise ValueError("Stateful/cached model requests are unsupported")
    messages: list[dict[str, JSON]] = []
    system = native_field(request.config, "system_instruction")
    if system is not None:
        if isinstance(system, str):
            text = text_value(system)
        elif isinstance(system, types.Content):
            text = plain(system)
        else:
            raise ValueError("Unsupported native system instruction")
        messages.append({"role": "system", "content": text})
    messages.extend(_message(content) for content in request.contents)
    normalize(messages)
    return RequestProjection(messages, tuple(request.contents), system is not None)


@dataclass(frozen=True)
class ResponseProjection:
    """Keep actual native response metadata and call IDs outside transform authority."""

    target: dict[str, JSON]
    response: LlmResponse

    def apply(self, value: object) -> LlmResponse:
        """Reconstruct native text only; preserve calls, usage, IDs and finish metadata."""
        target = model_text(value, self.target)
        response = self.response.model_copy(deep=True)
        if isinstance(target["content"], str):
            if response.content is None:
                raise ValueError("Missing native model content")
            response.content.parts = [types.Part(text=target["content"])]
        return response


def response_projection(response: LlmResponse) -> ResponseProjection:
    """Accept honest plain text or native tool-call-only responses, not null/rich output."""
    content = response.content
    allowed = {
        "model_version",
        "content",
        "usage_metadata",
        "finish_reason",
        "partial",
        "turn_complete",
        "interrupted",
    }
    if any(
        native_field(response, field) is not None
        for field in type(response).model_fields
        if field not in allowed
    ):
        raise ValueError("Rich, stateful and live model response metadata is unsupported")
    if (
        content is None
        or content.role != "model"
        or not content.parts
        or response.partial
        or response.interrupted
        or response.error_code is not None
        or response.error_message is not None
    ):
        raise ValueError("Incomplete, errored or empty native model output")
    parts = content.parts
    calls: list[JSON] = []
    if all(set(part.model_dump(exclude_none=True)) == {"text"} for part in parts):
        text: JSON = plain(content)
        if not text.strip():
            raise ValueError("Native model output requires nonempty text")
    else:
        text = None
        for part in parts:
            value = object_value(part.model_dump(mode="json", exclude_none=True))
            call = part.function_call
            if set(value) != {"function_call"} or call is None:
                raise ValueError("Only plain text or tool-call-only model output is supported")
            if not call.name:
                raise ValueError("Native tool call requires a name")
            if call.id is None:
                raise ValueError("Native tool call requires an explicit correlation ID")
            calls.append({"id": call.id, "name": call.name, "args": object_value(call.args)})
    finish = response.finish_reason.value if response.finish_reason is not None else "stop"
    target: dict[str, JSON] = {"content": text, "tool_calls": calls, "finish_reason": finish}
    normalize(target)
    return ResponseProjection(target, response)
