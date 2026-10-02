"""Bounded, detached JSON values at the host/SDK boundary."""

from __future__ import annotations

import math
from typing import TypeAlias, cast

JSON: TypeAlias = None | bool | int | float | str | list["JSON"] | dict[str, "JSON"]


def normalize(value: object) -> JSON:
    """Copy strict JSON; reject cycles, excessive size, and nonportable numbers."""
    active: set[int] = set()
    nodes = 0
    size = 0

    def visit(item: object, depth: int) -> JSON:
        nonlocal nodes, size
        nodes += 1
        if depth > 32 or nodes > 10_000:
            raise ValueError("JSON complexity limit exceeded")
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            integer = item
            if abs(integer) > 2**53 - 1:
                raise ValueError("JSON integer outside interoperable range")
            return integer
        if type(item) is float:
            number = item
            if not math.isfinite(number):
                raise ValueError("JSON number must be finite")
            return number
        if type(item) is str:
            text = item
            if len(text) > 1_048_576 - size:
                raise ValueError("JSON byte limit exceeded")
            size += len(text.encode("utf-8"))
            if size > 1_048_576:
                raise ValueError("JSON byte limit exceeded")
            return text
        if type(item) not in (list, dict):
            raise ValueError("Unsupported JSON value")
        if id(item) in active:
            raise ValueError("Cyclic JSON value")
        identity = id(item)
        active.add(identity)
        try:
            if isinstance(item, list):
                return [visit(child, depth + 1) for child in cast("list[object]", item)]
            result: dict[str, JSON] = {}
            for key, child in cast("dict[object, object]", item).items():
                if type(key) is not str:
                    raise ValueError("JSON keys must be strings")
                visit(key, depth + 1)
                result[key] = visit(child, depth + 1)
            return result
        finally:
            active.remove(identity)

    return visit(value, 0)


def object_value(value: object) -> dict[str, JSON]:
    """Require a detached JSON object."""
    result = normalize(value)
    if not isinstance(result, dict):
        raise ValueError("Expected JSON object")
    return result


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
