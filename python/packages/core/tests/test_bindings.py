"""Policy source and actual-object validation happens before policy file I/O."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import pytest
from agent_control_adapters import AgentPolicy, NativeRuntime
from agent_control_adapters import bindings as bindings_module
from agent_control_adapters.bindings import prepare_controls
from agent_hooks import ALLOW, AgentContext, Interceptor, Verdict


class Allow:
    def intercept(self, context: AgentContext) -> Verdict:
        return ALLOW


def test_native_runtime_detaches_mapping_and_retains_actual_objects() -> None:
    runtime, actor = object(), object()
    original = {"actor": actor}
    wrapped = NativeRuntime(runtime, original)
    original["actor"] = object()
    assert wrapped.runtime is runtime
    assert wrapped.agents["actor"] is actor
    with pytest.raises(TypeError):
        wrapped.agents["actor"] = object()  # pyright: ignore[reportIndexIssue]
    with pytest.raises(FrozenInstanceError):
        wrapped.runtime = object()  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("count", [0, 17])
def test_actor_cap_rejects_empty_and_oversized_native_runtime(count: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 16"):
        NativeRuntime(object(), {f"actor_{index}": object() for index in range(count)})


def test_native_runtime_rejects_duplicate_objects_and_bad_labels() -> None:
    actor = object()
    with pytest.raises(ValueError, match="distinct"):
        NativeRuntime(object(), {"first": actor, "second": actor})
    with pytest.raises(ValueError):
        NativeRuntime(object(), {"request/identity": actor})
    with pytest.raises(ValueError):
        NativeRuntime(object(), {"actor": None})
    with pytest.raises(TypeError):
        NativeRuntime(object(), cast("Mapping[str, object]", []))


def test_controls_own_configuration_and_run_shared_policies_first() -> None:
    actor = object()
    shared, extra = Allow(), Allow()
    native = {"native_label": actor}
    agents = {"trusted_label": actor}
    policies = [AgentPolicy("trusted_label", (extra,))]
    controls = prepare_controls(
        native, interceptors=[shared], agents=agents, agent_policies=policies
    )
    native.clear()
    agents.clear()
    policies.clear()
    binding = controls.actors[0]
    assert binding.label == "trusted_label"
    assert binding.native is actor
    assert binding.interceptors == (shared, extra)
    assert binding.identity != "trusted_label"
    assert binding.session_id != binding.identity
    second = prepare_controls({"agent": actor}, interceptors=[shared]).actors[0]
    assert second.identity != binding.identity
    assert second.session_id != binding.session_id


@pytest.mark.parametrize(
    "case",
    ["foreign", "duplicate", "unknown_policy", "label_collision", "timeout", "record_cap"],
)
def test_validation_precedes_policy_io(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_load(path: str | Path) -> Interceptor:
        pytest.fail("Invalid configuration must not open a policy file")

    monkeypatch.setattr(bindings_module, "load_acs_policy", forbidden_load)
    first, second = object(), object()
    agents = {"first": first}
    assignments: tuple[AgentPolicy, ...] = ()
    timeout, records = 5.0, 1000
    if case == "foreign":
        agents = {"first": object()}
    elif case == "duplicate":
        agents = {"first": first, "alias": first}
    elif case == "unknown_policy":
        assignments = (AgentPolicy("missing", policy_file="unopened.yaml"),)
    elif case == "label_collision":
        agents = {"second": first}
    elif case == "timeout":
        timeout = float("nan")
    elif case == "record_cap":
        records = 100_001
    with pytest.raises((TypeError, ValueError)):
        prepare_controls(
            {"first": first, "second": second},
            policy_file="unopened.yaml",
            agents=agents,
            agent_policies=assignments,
            timeout=timeout,
            max_records=records,
        )


def test_duplicate_policy_rejected_before_policy_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_load(path: str | Path) -> Interceptor:
        pytest.fail("Duplicate policy must not open a manifest")

    monkeypatch.setattr(bindings_module, "load_acs_policy", forbidden_load)
    actor = object()
    policy = AgentPolicy("agent", policy_file="unopened.yaml")
    with pytest.raises(ValueError, match="unique"):
        prepare_controls(
            {"agent": actor}, policy_file="unopened.yaml", agent_policies=[policy, policy]
        )


@pytest.mark.parametrize("count", [0, 65])
def test_interceptor_count_threshold(count: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 64"):
        prepare_controls({"agent": object()}, interceptors=[Allow() for _ in range(count)])


def test_acs_loader_shared_then_agent_order(monkeypatch: pytest.MonkeyPatch) -> None:
    paths: list[Path] = []
    loaded: list[Interceptor] = []

    def load(path: str | Path) -> Interceptor:
        paths.append(Path(path))
        interceptor = Allow()
        loaded.append(interceptor)
        return interceptor

    monkeypatch.setattr(bindings_module, "load_acs_policy", load)
    controls = prepare_controls(
        {"agent": object()},
        policy_file="shared.yaml",
        agent_policies=[AgentPolicy("agent", policy_file="actor.yaml")],
    )
    assert [path.name for path in paths] == ["shared.yaml", "actor.yaml"]
    assert controls.actors[0].interceptors == tuple(loaded)


def test_acs_only_rejects_python_interceptor_before_file_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(interceptor: Interceptor) -> None:
        raise TypeError("native ACS required")

    def forbidden_load(path: str | Path) -> Interceptor:
        pytest.fail("Invalid ACS-only source must not open a manifest")

    monkeypatch.setattr(bindings_module, "validate_acs_interceptor", reject)
    monkeypatch.setattr(bindings_module, "load_acs_policy", forbidden_load)
    with pytest.raises(TypeError, match="ACS"):
        prepare_controls(
            {"agent": object()},
            policy_file="shared.yaml",
            agent_policies=[AgentPolicy("agent", (Allow(),))],
            acs_only=True,
        )
