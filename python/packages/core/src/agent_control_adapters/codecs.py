"""Bounded write-back codecs that preserve native routing and trusted field shape."""

from __future__ import annotations

from .json import JSON, normalize, object_value


def envelope(value: object, keys: set[str]) -> dict[str, JSON]:
    """Require exactly the fields the receiving native mapper can consume."""
    result = object_value(value)
    if set(result) != keys:
        raise ValueError("Unsupported envelope fields")
    return result


def text_value(value: object) -> str:
    """Require bounded plain text instead of a success-shaped string conversion."""
    result = normalize(value)
    if not isinstance(result, str):
        raise ValueError("Expected text")
    return result


def wire_equal(left: object, right: object) -> bool:
    """Compare strict JSON without treating booleans as numeric routing metadata."""
    first, second = normalize(left), normalize(right)
    return _equal(first, second)


def _equal(left: JSON, right: JSON) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_equal(item, right[key]) for key, item in left.items())
        )
    if isinstance(left, list):
        return (
            isinstance(right, list)
            and len(left) == len(right)
            and all(_equal(first, second) for first, second in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def unchanged(value: object, original: object) -> JSON:
    """Reject transforms at a notification-only or unsupported native surface."""
    result = normalize(value)
    if not wire_equal(result, original):
        raise ValueError("Transforms are unsupported at this native surface")
    return result


def input_text(value: object) -> str:
    """Apply only prompt content, preserving the host-created user role."""
    target = envelope(value, {"content", "role"})
    if target["role"] != "user":
        raise ValueError("Input role cannot change")
    return text_value(target["content"])


def output_text(value: object) -> str:
    """Consume the canonical SDK output content object."""
    return text_value(envelope(value, {"content"})["content"])


def text_messages(value: object, original: list[dict[str, JSON]]) -> list[dict[str, JSON]]:
    """Permit text-only replacement without changes to roles, call IDs or routing."""
    target = normalize(value)
    if not isinstance(target, list) or len(target) != len(original):
        raise ValueError("Model message count cannot change")
    result: list[dict[str, JSON]] = []
    for replacement, source in zip(target, original, strict=True):
        message = envelope(replacement, set(source))
        for key, item in source.items():
            if key == "content" and isinstance(item, str):
                text_value(message[key])
            elif not wire_equal(message[key], item):
                raise ValueError("Model routing and non-text content cannot change")
        result.append(message)
    return result


def model_text(value: object, original: dict[str, JSON]) -> dict[str, JSON]:
    """Replace plain model text, preserving function calls and finish metadata."""
    target = envelope(value, set(original))
    for key, item in original.items():
        if key == "content" and isinstance(item, str):
            text_value(target[key])
        elif not wire_equal(target[key], item):
            raise ValueError("Model routing and non-text content cannot change")
    return target
