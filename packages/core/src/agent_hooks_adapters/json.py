"""Bounded, detached JSON values shared by adapter requests and host boundaries."""

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
    """Require a detached, bounded JSON object."""
    result = normalize(value)
    if not isinstance(result, dict):
        raise ValueError("Expected JSON object")
    return result
