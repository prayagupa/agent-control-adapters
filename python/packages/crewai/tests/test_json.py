"""Strict content boundaries never stringify arbitrary Python objects."""

from __future__ import annotations

import pytest
from agent_hooks_crewai._json import envelope, messages_value, normalize, object_value


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
        2**53,
        -(2**53),
        object(),
        {1: "not-string-key"},
        ("tuple",),
        {"set"},
        b"bytes",
        "\ud800",
    ],
)
def test_non_json_values_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        normalize(value)


def test_cycles_depth_size_and_nodes_bounded() -> None:
    cycle: list[object] = []
    cycle.append(cycle)
    mapping: dict[str, object] = {}
    mapping["self"] = mapping
    deep: object = None
    for _ in range(34):
        deep = [deep]
    for value in [cycle, mapping, deep, "x" * 1_048_577, [None] * 10_001]:
        with pytest.raises(ValueError):
            normalize(value)


def test_shared_noncyclic_references_are_copied() -> None:
    shared = {"value": "text"}
    copied = normalize([shared, shared])
    assert copied == [shared, shared]
    assert isinstance(copied, list) and copied[0] is not copied[1]
    shared["value"] = "changed"
    assert copied == [{"value": "text"}, {"value": "text"}]


@pytest.mark.parametrize(
    "value",
    [
        [{"role": "root", "content": "x"}],
        [{"role": "user", "content": "x", "session": {}}],
        [{"role": "user", "content": ["multimodal"]}],
        [{"role": "user", "content": "x", "cache_breakpoint": "yes"}],
        [{"role": "assistant", "tool_calls": [{"id": "x", "type": "unknown", "function": {}}]}],
        [],
        {},
        None,
    ],
)
def test_invalid_message_shapes(value: object) -> None:
    with pytest.raises(ValueError):
        messages_value(value)


def test_unknown_envelope_fields_rejected() -> None:
    with pytest.raises(ValueError):
        envelope({"content": "x", "session": "spoof"}, {"content"})
    with pytest.raises(ValueError):
        object_value([])
