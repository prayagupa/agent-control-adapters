"""Real CrewAI and ACS/Rego integration; only model I/O uses deterministic fixtures."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from typing import cast

import pytest
from agent_control_adapters import (
    Adapter,
    AdapterBlocked,
    AgentPolicy,
    Backend,
    RunRequest,
    with_agent_controls,
)
from agent_control_adapters.acs import load_acs_policy
from agent_control_crewai import CrewAIAdapter, portable
from agent_hooks import (
    ALLOW,
    AgentContext,
    AgentContextBuilder,
    ApprovalOutcome,
    ApprovalRequest,
    ApprovalResolution,
    Decision,
    InterceptionPoint,
    Interceptor,
    Transform,
    Verdict,
)
from crewai import Agent, Crew, Task
from crewai.tools import tool
from crewai_adapter_test_helpers import NetworkStub, Policy, make_crew, tool_calls

pytestmark = pytest.mark.acs
ALL_POINTS: tuple[str, ...] = tuple(point.value for point in InterceptionPoint)
AGENT_POINTS = ("pre_model_call", "post_model_call", "pre_tool_call", "post_tool_call")


@pytest.fixture(autouse=True)
def deterministic_acs_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construct test policies without ambient deadline or language overrides."""
    monkeypatch.delenv("ACS_OPA_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("ACS_REGO_V0", raising=False)


def manifest(
    root: Path,
    rule: str = "",
    *,
    points: tuple[str, ...] = ALL_POINTS,
    bundle: str = "./policy",
    create_bundle: bool = True,
) -> Path:
    """Write an actual native manifest and Rego bundle, not an evaluator stub."""
    root.mkdir(parents=True, exist_ok=True)
    if create_bundle:
        directory = root / "policy"
        directory.mkdir()
        (directory / "guard.rego").write_text(
            'package controls\nimport rego.v1\ndefault verdict := {"decision": "allow"}\n' + rule
        )
    path = root / "controls.yaml"
    path.write_text(
        json.dumps(
            {
                "agent_control_specification_version": "0.5.0-alpha.1",
                "policies": {
                    "guard": {"type": "rego", "bundle": bundle, "query": "data.controls.verdict"}
                },
                "intervention_points": {
                    point: {"policy_target": "$snap.target", "policy": {"id": "guard"}}
                    for point in points
                },
            }
        )
    )
    return path


def native(adapter: Adapter) -> CrewAIAdapter:
    """Access the actual backend to assert its owned-resource postconditions."""
    assert isinstance(adapter, CrewAIAdapter)
    return adapter


def assert_cleaned(adapter: Adapter) -> None:
    backend = native(adapter)
    worker = backend._worker
    assert worker is not None
    assert not worker._thread.is_alive()
    assert worker._loop is None and worker._job is None and worker._task is None
    assert not worker._bindings and not worker._callers
    assert not backend._pending and not backend._shared and not backend._policies


def second_agent(crew: Crew, first: Agent, model: NetworkStub) -> Agent:
    """Keep the same role text to prove bindings use real object identities."""
    second = Agent(
        role=first.role,
        goal=first.goal,
        backstory=first.backstory,
        llm=model,
        max_iter=3,
        max_retry_limit=0,
        respect_context_window=False,
    )
    crew.agents.append(second)
    crew.tasks.append(
        Task(description="Second answer {question}", expected_output="Short answer", agent=second)
    )
    return second


@pytest.mark.parametrize("attempt", range(3))
def test_yaml_allow_uses_real_runner_bounded_audit_and_single_use(
    tmp_path: Path, attempt: int
) -> None:
    crew, _, model = make_crew()
    path = manifest(tmp_path / str(attempt))
    adapter = with_agent_controls(
        backend=Backend.CREWAI, runtime=crew, policy_file=path, max_records=3
    )
    result = adapter.run(RunRequest({"question": "hello"}))
    assert result.output == "completed" and len(model.seen) == 1
    assert len(result.records) == 3 and result.records_dropped == 3
    assert result.records == adapter.records
    assert all(summary.name is None for record in result.records for summary in record.verdicts)
    with pytest.raises(RuntimeError, match="single-use"):
        adapter.run()
    assert_cleaned(adapter)


def test_yaml_blocks_trusted_tool_name_not_forged_argument(tmp_path: Path) -> None:
    effects: list[str] = []

    @tool
    def echo(value: str, name: str = "safe") -> str:
        """Preview text without external effects."""
        effects.append(value)
        return value + name

    response = tool_calls("test")
    function = response[0]["function"]
    assert isinstance(function, dict)
    function["arguments"] = {"value": "test", "name": "search"}
    model = NetworkStub(native=True, answers=[response, "Final Answer: complete"])
    crew, _, _ = make_crew(model, tools=[echo], cache=False)
    path = manifest(
        tmp_path,
        """verdict := {"decision": "deny", "reason": "test:acs_tool"} if {
    input.snapshot.interception_point == "pre_tool_call"
    input.snapshot.tool_call.name == "echo"
}
""",
    )
    adapter = with_agent_controls(backend=Backend.CREWAI, runtime=crew, policy_file=path)
    with pytest.raises(AdapterBlocked):
        adapter.run(RunRequest({"question": "hello"}))
    assert not effects and len(model.seen) == 1
    assert any(record.verdict.reason == "test:acs_tool" for record in adapter.records)
    assert not any(
        record.interception_point.value == "post_tool_call" for record in adapter.records
    )
    assert_cleaned(adapter)


@pytest.mark.parametrize("shared_yaml", [False, True], ids=["native-shared", "yaml-shared"])
@pytest.mark.parametrize("agent_yaml", [False, True], ids=["native-agent", "yaml-agent"])
def test_all_source_combinations_preserve_agent_scope_and_shared_first_order(
    shared_yaml: bool, agent_yaml: bool, tmp_path: Path
) -> None:
    crew, first, model = make_crew()
    second = second_agent(crew, first, model)
    shared = Policy()
    shared_path = manifest(tmp_path / "shared")

    def assignment(agent: Agent, label: str) -> AgentPolicy:
        def transform(context: AgentContext) -> Verdict:
            assert context["agent"]["id"] == str(agent.id)
            if context["interception_point"] == "post_model_call":
                return Verdict(
                    Decision.TRANSFORM,
                    transform=Transform("$target.content", f"Final Answer: {label}"),
                )
            return ALLOW

        if not agent_yaml:
            return AgentPolicy(label, (Policy(transform),))
        path = manifest(
            tmp_path / label,
            f"""verdict := {{
    "decision": "transform",
    "transform": {{"path": "$target.content", "value": "Final Answer: {label}"}}
}} if {{
    input.snapshot.interception_point == "post_model_call"
    input.snapshot.agent.id == {json.dumps(str(agent.id))}
}}
verdict := {{"decision": "deny", "reason": "test:foreign_agent"}} if {{
    input.snapshot.agent.id != {json.dumps(str(agent.id))}
}}
""",
            points=AGENT_POINTS,
        )
        return AgentPolicy(label, policy_file=path)

    adapter = with_agent_controls(
        backend=Backend.CREWAI,
        runtime=crew,
        interceptors=None if shared_yaml else [shared],
        policy_file=shared_path if shared_yaml else None,
        agents={"first": first, "second": second},
        agent_policies=[assignment(first, "first"), assignment(second, "second")],
    )
    result = adapter.run(RunRequest({"question": "hello"}))
    assert len(model.seen) == 2
    outputs: list[str] = []
    for task in crew.tasks:
        assert task.output is not None
        outputs.append(task.output.raw)
    assert outputs == ["first", "second"]
    records = [
        record for record in result.records if record.interception_point.value == "post_model_call"
    ]
    assert len(records) == 2 and len({record.session_id for record in records}) == 2
    assert all(record.interceptors_registered == 2 for record in records)
    assert all(record.verdict.decision == Decision.TRANSFORM for record in records)
    assert all([summary.name for summary in record.verdicts] == [None, None] for record in records)
    assert_cleaned(adapter)


@pytest.mark.parametrize("case", ["foreign", "unknown", "duplicate", "duplicate-policy"])
def test_all_bindings_are_validated_before_any_yaml_loading(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crew, agent, model = make_crew()
    _, foreign, _ = make_crew()
    agents: dict[str, object] = {"one": foreign if case == "foreign" else agent}
    policies = [AgentPolicy("unknown" if case == "unknown" else "one", policy_file=tmp_path / "a")]
    if case == "duplicate":
        agents["two"] = agent
    if case == "duplicate-policy":
        policies *= 2

    def no_load(path: str | Path) -> Interceptor:
        raise AssertionError("Invalid bindings performed policy I/O")

    monkeypatch.setattr(portable, "load_acs_policy", no_load)
    with pytest.raises(ValueError):
        with_agent_controls(
            backend=Backend.CREWAI,
            runtime=crew,
            policy_file=tmp_path / "shared",
            agents=agents,
            agent_policies=policies,
        )
    assert not model.seen


@pytest.mark.parametrize("limit", [False, True, 0, -1, 100_001, 1.5, "1", None])
def test_invalid_record_limits_fail_before_yaml_loading(
    limit: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crew, _, model = make_crew()

    def no_load(path: str | Path) -> Interceptor:
        raise AssertionError("Invalid record limit performed policy I/O")

    monkeypatch.setattr(portable, "load_acs_policy", no_load)
    with pytest.raises(ValueError):
        with_agent_controls(
            backend=Backend.CREWAI,
            runtime=crew,
            policy_file=tmp_path / "controls.yaml",
            max_records=cast(int, limit),
        )
    assert not model.seen


@pytest.mark.parametrize("runtime", [None, object(), [], "not-a-crew"])
def test_invalid_runtime_fails_before_yaml_loading(
    runtime: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_load(path: str | Path) -> Interceptor:
        raise AssertionError("Invalid runtime performed policy I/O")

    monkeypatch.setattr(portable, "load_acs_policy", no_load)
    with pytest.raises(TypeError, match="Crew runtime"):
        with_agent_controls(
            backend=Backend.CREWAI, runtime=runtime, policy_file=tmp_path / "controls.yaml"
        )


def test_native_runtime_builder_exception_propagates_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = import_module("agent_control_spec")
    error = RuntimeError("Native runtime builder failed")
    crew, _, model = make_crew()

    def fail_builder(manifest_path: str) -> Interceptor:
        raise error

    monkeypatch.setattr(module, "AcsInterceptor", fail_builder)
    with pytest.raises(RuntimeError) as raised:
        with_agent_controls(
            backend=Backend.CREWAI, runtime=crew, policy_file=tmp_path / "controls.yaml"
        )
    assert raised.value is error
    assert not model.seen


def test_yaml_agent_bindings_snapshot_mutable_customer_collections(tmp_path: Path) -> None:
    crew, agent, model = make_crew()
    _, foreign, _ = make_crew()
    shared = manifest(tmp_path / "shared")
    scoped = manifest(
        tmp_path / "agent",
        f"""verdict := {{"decision": "deny", "reason": "test:wrong_agent"}} if {{
    input.snapshot.agent.id != {json.dumps(str(agent.id))}
}}""",
        points=AGENT_POINTS,
    )
    agents: dict[str, object] = {"one": agent}
    policies = [AgentPolicy("one", policy_file=scoped)]
    adapter = with_agent_controls(
        backend=Backend.CREWAI,
        runtime=crew,
        policy_file=shared,
        agents=agents,
        agent_policies=policies,
    )
    agents["one"] = foreign
    policies[:] = [AgentPolicy("one", policy_file=tmp_path / "missing.yaml")]
    result = adapter.run(RunRequest({"question": "test"}))
    assert result.output == "completed" and len(model.seen) == 1
    assert all(
        record.interceptors_registered == 2
        for record in result.records
        if record.interception_point.value in AGENT_POINTS
    )
    assert_cleaned(adapter)


def test_missing_tool_bundle_can_fail_after_a_model_request(tmp_path: Path) -> None:
    effects: list[str] = []

    @tool
    def echo(value: str) -> str:
        """Capture a tool body that must not execute after a native engine failure."""
        effects.append(value)
        return value

    model = NetworkStub(native=True, answers=[tool_calls("echo")])
    crew, _, _ = make_crew(model, tools=[echo], cache=False)
    path = manifest(tmp_path)
    path.write_text(
        json.dumps(
            {
                "agent_control_specification_version": "0.5.0-alpha.1",
                "policies": {
                    "healthy": {
                        "type": "rego",
                        "bundle": "./policy",
                        "query": "data.controls.verdict",
                    },
                    "broken": {
                        "type": "rego",
                        "bundle": "./missing",
                        "query": "data.controls.verdict",
                    },
                },
                "intervention_points": {
                    point: {
                        "policy_target": "$snap.target",
                        "policy": {"id": "broken" if point == "pre_tool_call" else "healthy"},
                    }
                    for point in ALL_POINTS
                },
            }
        )
    )
    adapter = with_agent_controls(backend=Backend.CREWAI, runtime=crew, policy_file=path)
    with pytest.raises(AdapterBlocked):
        adapter.run(RunRequest({"question": "test"}))
    assert len(model.seen) == 1 and not effects
    assert any(
        record.interception_point.value == "pre_tool_call"
        and record.verdict.reason == "runtime_error:policy_invocation_failed"
        for record in adapter.records
    )
    assert not any(
        record.interception_point.value == "post_tool_call" for record in adapter.records
    )
    assert_cleaned(adapter)


@pytest.mark.parametrize("extra", [0, 1])
def test_native_snapshot_limit_is_measured_on_the_full_serialization(
    tmp_path: Path, extra: int
) -> None:
    limit = 1_048_576
    context = AgentContextBuilder(
        agent_id="size-probe", framework="crewai", session_id="size-probe"
    ).input(content="")
    fixed_bytes = len(json.dumps(context, separators=(",", ":"), ensure_ascii=False).encode())
    content = "x" * ((limit - fixed_bytes) // 2 + extra)
    context["input"]["content"] = content
    context["target"]["content"] = content
    serialized_bytes = len(json.dumps(context, separators=(",", ":"), ensure_ascii=False).encode())
    assert limit - 1 <= serialized_bytes <= limit + 2
    policy = load_acs_policy(manifest(tmp_path))
    verdict = policy.intercept(context)
    assert isinstance(verdict, Verdict)
    assert (verdict.decision == Decision.DENY) is (serialized_bytes > limit)
    assert verdict.reason == (
        "runtime_error:resource_limit_exceeded" if serialized_bytes > limit else None
    )


def test_valid_maximum_request_cannot_bypass_the_full_context_budget(tmp_path: Path) -> None:
    crew, _, model = make_crew()
    content = "x" * (1_048_576 - len("question"))
    request = RunRequest({"question": content})
    assert len(content.encode()) + len(b"question") == 1_048_576
    adapter = with_agent_controls(
        backend=Backend.CREWAI, runtime=crew, policy_file=manifest(tmp_path)
    )
    with pytest.raises(AdapterBlocked):
        adapter.run(request)
    assert not model.seen
    assert_cleaned(adapter)


def test_real_rego_input_and_response_transforms_use_existing_write_back(tmp_path: Path) -> None:
    crew, _, model = make_crew()
    path = manifest(
        tmp_path,
        """verdict := {"decision": "transform", "transform": {
    "path": "$target.content.question", "value": "input changed by policy"
}} if {
    input.snapshot.interception_point == "input"
}
verdict := {"decision": "transform", "transform": {
    "path": "$target.content", "value": "Final Answer: response changed by policy"
}} if {
    input.snapshot.interception_point == "post_model_call"
}""",
    )
    adapter = with_agent_controls(backend=Backend.CREWAI, runtime=crew, policy_file=path)
    result = adapter.run(RunRequest({"question": "original input"}))
    assert len(model.seen) == 1 and "input changed by policy" in json.dumps(model.seen)
    assert "original input" not in json.dumps(model.seen)
    assert result.output == "response changed by policy"
    assert {
        record.interception_point.value
        for record in result.records
        if record.verdict.decision == Decision.TRANSFORM
    } == {"input", "post_model_call"}
    assert_cleaned(adapter)


@pytest.mark.parametrize("duration", [False, 0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["timeout", "emit_timeout"])
def test_invalid_deadlines_fail_before_yaml_loading(
    field: str, duration: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crew, _, model = make_crew()

    def no_load(path: str | Path) -> Interceptor:
        raise AssertionError("Invalid deadline performed policy I/O")

    monkeypatch.setattr(portable, "load_acs_policy", no_load)
    with pytest.raises(ValueError):
        with_agent_controls(
            backend=Backend.CREWAI,
            runtime=crew,
            policy_file=tmp_path / "controls.yaml",
            timeout=duration if field == "timeout" else 5,
            emit_timeout=duration if field == "emit_timeout" else 30,
        )
    assert not model.seen


@pytest.mark.parametrize("source", ["missing", "malformed", "directory"])
def test_native_manifest_loader_errors_surface_before_running(source: str, tmp_path: Path) -> None:
    crew, _, model = make_crew()
    path = tmp_path / "controls.yaml"
    if source == "malformed":
        path.write_text("agent_control_specification_version: [")
    elif source == "directory":
        path = tmp_path
    with pytest.raises(ValueError):
        with_agent_controls(backend=Backend.CREWAI, runtime=crew, policy_file=path)
    assert not model.seen


@pytest.mark.parametrize("per_agent", [False, True])
def test_missing_bundle_fails_at_the_first_point_bound_to_it(
    per_agent: bool, tmp_path: Path
) -> None:
    crew, first, model = make_crew()
    broken = manifest(tmp_path / "broken", bundle="./missing", create_bundle=False)
    if per_agent:
        second = second_agent(crew, first, model)
        adapter = with_agent_controls(
            backend=Backend.CREWAI,
            runtime=crew,
            interceptors=[Policy()],
            agents={"second": second},
            agent_policies=[AgentPolicy("second", policy_file=broken)],
        )
    else:
        adapter = with_agent_controls(backend=Backend.CREWAI, runtime=crew, policy_file=broken)
    with pytest.raises(AdapterBlocked):
        adapter.run(RunRequest({"question": "hello"}))
    assert len(model.seen) == (1 if per_agent else 0)
    failure = next(record for record in adapter.records if record.verdict.decision == Decision.DENY)
    assert failure.interception_point.value == ("pre_model_call" if per_agent else "agent_startup")
    assert failure.verdict.reason == "runtime_error:policy_invocation_failed"
    assert_cleaned(adapter)


def test_real_rego_engine_error_is_not_a_success_result(tmp_path: Path) -> None:
    crew, _, model = make_crew()
    path = manifest(tmp_path, "this is not valid Rego")
    adapter = with_agent_controls(backend=Backend.CREWAI, runtime=crew, policy_file=path)
    with pytest.raises(AdapterBlocked):
        adapter.run()
    assert not model.seen
    assert adapter.records[0].verdict.reason == "runtime_error:policy_invocation_failed"
    assert_cleaned(adapter)


@pytest.mark.parametrize("point", ["agent_startup", "agent_shutdown"])
def test_missing_shared_point_retains_enforcement_vs_shutdown_audit_semantics(
    point: str, tmp_path: Path
) -> None:
    crew, _, model = make_crew()
    path = manifest(tmp_path, points=tuple(item for item in ALL_POINTS if item != point))
    adapter = with_agent_controls(backend=Backend.CREWAI, runtime=crew, policy_file=path)
    if point == "agent_startup":
        with pytest.raises(AdapterBlocked):
            adapter.run()
        assert not model.seen
    else:
        assert adapter.run(RunRequest({"question": "hello"})).output == "completed"
        assert len(model.seen) == 1
    assert any(
        record.interception_point.value == point
        and record.verdict.decision == Decision.DENY
        and record.verdict.reason is not None
        and record.verdict.reason.startswith("runtime_error:")
        for record in adapter.records
    )
    assert_cleaned(adapter)


class Approve:
    """Resolve actual SDK approval requests by their trusted context identity."""

    def resolve(self, request: ApprovalRequest) -> ApprovalResolution:
        return ApprovalResolution(ApprovalOutcome.APPROVE, request.context_identity, ALLOW)


def test_yaml_approval_timeout_allow_does_not_create_host_authority(tmp_path: Path) -> None:
    crew, _, model = make_crew()
    path = manifest(
        tmp_path,
        """verdict := {"decision": "deny", "reason": "test:approval", "approval": {}} if {
    input.snapshot.interception_point == "pre_model_call"
}""",
    )
    document = cast(dict[str, object], json.loads(path.read_text()))
    document["approval"] = {"timeout_seconds": 1, "on_timeout": "allow"}
    path.write_text(json.dumps(document))
    adapter = with_agent_controls(backend=Backend.CREWAI, runtime=crew, policy_file=path)
    with pytest.raises(AdapterBlocked):
        adapter.run(RunRequest({"question": "test"}))
    assert not model.seen
    assert any(record.verdict.decision == Decision.DENY for record in adapter.records)
    assert_cleaned(adapter)


@pytest.mark.parametrize("agent_yaml", [False, True])
def test_shared_yaml_mandatory_deny_cannot_be_lifted_by_agent_approval(
    tmp_path: Path, agent_yaml: bool
) -> None:
    crew, agent, model = make_crew()
    shared = manifest(
        tmp_path / "shared",
        """verdict := {"decision": "deny", "reason": "test:shared_deny"} if {
    input.snapshot.interception_point == "pre_model_call"
}""",
    )
    if agent_yaml:
        path = manifest(
            tmp_path / "agent",
            """verdict := {"decision": "deny", "reason": "test:agent_approval", "approval": {}} if {
    input.snapshot.interception_point == "pre_model_call"
}""",
            points=AGENT_POINTS,
        )
        assignment = AgentPolicy("one", policy_file=path)
    else:
        policy = Policy(
            lambda context: (
                Verdict.escalate(reason="test:agent_approval")
                if context["interception_point"] == "pre_model_call"
                else ALLOW
            )
        )
        assignment = AgentPolicy("one", (policy,))
    adapter = with_agent_controls(
        backend=Backend.CREWAI,
        runtime=crew,
        policy_file=shared,
        agents={"one": agent},
        agent_policies=[assignment],
        resolver=Approve(),
    )
    with pytest.raises(AdapterBlocked):
        adapter.run(RunRequest({"question": "test"}))
    assert not model.seen
    records = [
        record for record in adapter.records if record.interception_point.value == "pre_model_call"
    ]
    assert len(records) == 1
    assert [summary.reason for summary in records[0].verdicts] == [
        "test:shared_deny",
        "test:agent_approval",
    ]
    assert records[0].verdict.decision == Decision.DENY
    assert_cleaned(adapter)


@pytest.mark.parametrize("mandatory_denial", [False, True])
def test_yaml_approval_cannot_skip_a_mandatory_native_agent_control(
    mandatory_denial: bool, tmp_path: Path
) -> None:
    crew, agent, model = make_crew()
    path = manifest(
        tmp_path,
        """verdict := {"decision": "deny", "reason": "test:acs_approval", "approval": {}} if {
    input.snapshot.interception_point == "pre_model_call"
}
""",
    )
    bound = Policy(
        lambda context: (
            Verdict.deny(reason="test:mandatory")
            if mandatory_denial and context["interception_point"] == "pre_model_call"
            else ALLOW
        )
    )
    adapter = with_agent_controls(
        backend=Backend.CREWAI,
        runtime=crew,
        policy_file=path,
        agents={"agent": agent},
        agent_policies=[AgentPolicy("agent", (bound,))],
        resolver=Approve(),
    )
    if mandatory_denial:
        with pytest.raises(AdapterBlocked):
            adapter.run(RunRequest({"question": "hello"}))
        assert not model.seen
    else:
        assert adapter.run(RunRequest({"question": "hello"})).output == "completed"
        assert len(model.seen) == 1
        assert any(record.resolved_by == "approval" for record in adapter.records)
    assert bound.contexts
    assert_cleaned(adapter)


def test_cedar_legacy_metadata_requirement_stays_explicitly_fail_closed(tmp_path: Path) -> None:
    crew, _, model = make_crew()
    path = tmp_path / "cedar.yaml"
    path.write_text(
        json.dumps(
            {
                "agent_control_specification_version": "0.5.0-alpha.1",
                "policies": {
                    "guard": {"type": "cedar", "policy_set": "permit(principal, action, resource);"}
                },
                "intervention_points": {
                    point: {"policy_target": "$snap.target", "policy": {"id": "guard"}}
                    for point in ALL_POINTS
                },
            }
        )
    )
    adapter = with_agent_controls(backend=Backend.CREWAI, runtime=crew, policy_file=path)
    with pytest.raises(AdapterBlocked):
        adapter.run()
    assert not model.seen
    assert adapter.records[0].verdict.reason == "runtime_error:policy_invocation_failed"
    assert_cleaned(adapter)


def test_symlinked_manifest_matches_native_resource_resolution(tmp_path: Path) -> None:
    original = manifest(tmp_path / "original")
    location = tmp_path / "link"
    location.mkdir()
    link = location / "controls.yaml"
    link.symlink_to(original)
    baseline = load_acs_policy(link)
    crew, _, _ = make_crew()
    captured = Policy()
    adapter = with_agent_controls(
        backend=Backend.CREWAI,
        runtime=crew,
        policy_file=link,
        agents={"agent": crew.agents[0]},
        agent_policies=[AgentPolicy("agent", (captured,))],
    )
    assert adapter.run(RunRequest({"question": "hello"})).output == "completed"
    context = captured.contexts[0]
    expected = baseline.intercept(context)
    assert isinstance(expected, Verdict)
    assert expected.decision == Decision.ALLOW
    assert_cleaned(adapter)


def test_yaml_adapter_has_atomic_single_use_under_concurrent_callers(tmp_path: Path) -> None:
    entered, release = threading.Event(), threading.Event()

    def hold_model() -> None:
        entered.set()
        assert release.wait(3)

    model = NetworkStub(on_call=hold_model)
    crew, _, _ = make_crew(model)
    adapter = with_agent_controls(
        backend=Backend.CREWAI, runtime=crew, policy_file=manifest(tmp_path)
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(adapter.run, RunRequest({"question": "hello"}))
        try:
            assert entered.wait(2)
            with pytest.raises(RuntimeError, match="single-use"):
                adapter.run()
        finally:
            release.set()
        assert first.result(timeout=3).output == "completed"
    assert len(model.seen) == 1
    assert_cleaned(adapter)


def test_nonpreemptible_acs_call_keeps_cleanup_failure_observable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = threading.Event(), threading.Event()
    path = manifest(tmp_path)
    control = load_acs_policy(path)
    original = type(control).intercept

    def blocking(self: Interceptor, context: AgentContext) -> Verdict:
        entered.set()
        assert release.wait(3)
        result = original(self, context)
        assert isinstance(result, Verdict)
        return result

    monkeypatch.setattr(type(control), "intercept", blocking)
    crew, _, model = make_crew()
    adapter = with_agent_controls(
        backend=Backend.CREWAI, runtime=crew, policy_file=path, emit_timeout=0.03
    )
    backend = native(adapter)
    try:
        with pytest.raises(AdapterBlocked):
            adapter.run()
        assert entered.is_set() and not model.seen
        assert adapter.cleanup_error is not None
        assert backend._worker is not None and backend._worker._thread.is_alive()
    finally:
        release.set()
        worker = backend._worker
        if worker is not None:
            worker._thread.join(timeout=2)
            worker.close()
    assert_cleaned(adapter)
