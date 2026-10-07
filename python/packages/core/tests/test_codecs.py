"""Applied content-only transforms cannot smuggle role, call or metadata changes."""

from __future__ import annotations

import pytest
from agent_control_adapters.codecs import (
    input_text,
    model_text,
    output_text,
    text_messages,
    unchanged,
    wire_equal,
)
from agent_control_adapters.json import JSON


def test_wire_equality_distinguishes_bool_number_nested_and_key_order() -> None:
    assert not wire_equal(True, 1)
    assert not wire_equal({"id": [False]}, {"id": [0]})
    assert wire_equal({"a": [1], "b": "x"}, {"b": "x", "a": [1]})
    assert not wire_equal({"a": [1]}, {"a": [1, 2]})
    assert not wire_equal(1, 1.0)


def test_prompt_and_output_transforms_consume_canonical_content() -> None:
    assert input_text({"content": "changed", "role": "user"}) == "changed"
    assert output_text({"content": "changed"}) == "changed"
    with pytest.raises(ValueError):
        input_text({"content": "changed", "role": "system"})
    with pytest.raises(ValueError):
        output_text({"content": "changed", "session_id": "spoof"})
    with pytest.raises(ValueError):
        input_text({"content": False, "role": "user"})


def test_text_message_transform_preserves_system_and_function_routing() -> None:
    original: list[dict[str, JSON]] = [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "prompt"},
        {"role": "tool", "content": "result", "tool_call_id": "trusted"},
    ]
    transformed: list[dict[str, JSON]] = [
        {"role": "system", "content": "changed system"},
        {"role": "user", "content": "changed prompt"},
        {"role": "tool", "content": "changed result", "tool_call_id": "trusted"},
    ]
    assert text_messages(transformed, original) == transformed
    transformed[0]["role"] = "user"
    with pytest.raises(ValueError, match="routing"):
        text_messages(transformed, original)
    transformed[0]["role"] = "system"
    transformed[2]["tool_call_id"] = "spoof"
    with pytest.raises(ValueError, match="routing"):
        text_messages(transformed, original)


def test_model_transform_cannot_change_calls_or_finish_reason() -> None:
    original: dict[str, JSON] = {
        "content": "native",
        "tool_calls": [{"id": "trusted", "name": "ping", "arguments": {}}],
        "finish_reason": "tool_calls",
    }
    transformed = {**original, "content": "changed"}
    assert model_text(transformed, original)["content"] == "changed"
    with pytest.raises(ValueError, match="routing"):
        model_text({**transformed, "tool_calls": []}, original)
    with pytest.raises(ValueError, match="routing"):
        model_text({**transformed, "finish_reason": "stop"}, original)


def test_notification_transform_rejected_and_return_is_detached() -> None:
    original: dict[str, JSON] = {"tools_registered": ["ping"]}
    result = unchanged(original, original)
    assert result == original
    assert result is not original
    with pytest.raises(ValueError, match="unsupported"):
        unchanged({"tools_registered": []}, original)
