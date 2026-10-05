"""Bounded, detached JSON values at the host/SDK boundary."""

from __future__ import annotations

from agent_hooks_adapters.json import (
    JSON as JSON,
)
from agent_hooks_adapters.json import (
    normalize as normalize,
)
from agent_hooks_adapters.json import (
    object_value as object_value,
)


def envelope(value: object, keys: set[str]) -> dict[str, JSON]:
    """Require exactly the supported envelope fields."""
    result = object_value(value)
    if set(result) != keys:
        raise ValueError("Unsupported envelope fields")
    return result


def text_value(value: object) -> str:
    """Require bounded plain text."""
    result = normalize(value)
    if not isinstance(result, str):
        raise ValueError("Expected text")
    return result


def messages_value(value: object) -> list[dict[str, JSON]]:
    """Validate text messages and native function-call history, not multimodal input."""
    result = normalize(value)
    if not isinstance(result, list) or not result:
        raise ValueError("Expected nonempty messages")
    messages: list[dict[str, JSON]] = []
    for message in result:
        if not isinstance(message, dict) or not set(message) <= {
            "role",
            "content",
            "name",
            "tool_calls",
            "tool_call_id",
            "cache_breakpoint",
        }:
            raise ValueError("Unsupported message fields")
        if message.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError("Unsupported message role")
        if "content" not in message:
            raise ValueError("Message content is required")
        if not isinstance(message.get("content"), (str, type(None))):
            raise ValueError("Only text messages are supported")
        for name in ("name", "tool_call_id"):
            if name in message and not isinstance(message[name], str):
                raise ValueError("Invalid message identifier")
        if "cache_breakpoint" in message and type(message["cache_breakpoint"]) is not bool:
            raise ValueError("Invalid cache breakpoint")
        if "tool_calls" in message:
            calls = message["tool_calls"]
            if not isinstance(calls, list):
                raise ValueError("Invalid tool calls")
            for call in calls:
                entry = envelope(call, {"id", "type", "function"})
                if not isinstance(entry["id"], str) or entry["type"] != "function":
                    raise ValueError("Invalid tool call")
                function = envelope(entry["function"], {"name", "arguments"})
                if not isinstance(function["name"], str) or not isinstance(
                    function["arguments"], (str, dict)
                ):
                    raise ValueError("Invalid function call")
        messages.append(message)
    return messages
