"""Portable calls exercise actual CrewAI/SDK governance with stubbed model I/O."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import groupby
from typing import Literal

import pytest
from agent_hooks import (
    ALLOW,
    AgentContext,
    Decision,
    InterceptionEmitter,
    InterceptionRecord,
    Verdict,
)
from agent_hooks_adapters import (
    Adapter,
    AdapterBlocked,
    AdapterCleanupError,
    AgentPolicy,
    Backend,
    RunRequest,
    RunResult,
    UnsupportedBackendError,
    create_adapter,
)
from agent_hooks_crewai import (
    AdapterBlocked as NativeAdapterBlocked,
)
from agent_hooks_crewai import (
    AdapterCleanupError as NativeAdapterCleanupError,
)
from agent_hooks_crewai import (
    AgentPolicy as NativeAgentPolicy,
)
from agent_hooks_crewai import CrewAIAdapter
from crewai import Agent, Crew, Task
from crewai.crews.crew_output import CrewOutput
from crewai.hooks.dispatch import HookAborted, InterceptionPoint, get_scoped_hooks
from crewai.tools import tool
from crewai_adapter_test_helpers import NetworkStub, Policy, denial, make_crew, tool_calls

Method = Literal["run", "kickoff"]
HookSnapshot = dict[InterceptionPoint, tuple[object, ...]]
PolicyEvent = tuple[str, str, str, str]


def _hooks() -> HookSnapshot:
    return {point: tuple(get_scoped_hooks(point)) for point in InterceptionPoint}


def _native(adapter: Adapter) -> CrewAIAdapter:
    assert isinstance(adapter, CrewAIAdapter)
    return adapter


def _assert_cleaned(adapter: CrewAIAdapter, hooks: HookSnapshot) -> None:
    assert _hooks() == hooks
    worker = adapter._worker
    if worker is not None:
        assert not worker._thread.is_alive()
        assert worker._loop is None
        assert worker._task is None
        assert worker._job is None
        assert not worker._bindings
        assert not worker._callers
        assert not worker.available
    assert not adapter._pending
    assert not adapter._agents
    assert not adapter._tasks
    assert not adapter._iteration_limits
    assert not adapter._shared
    assert not adapter._policies
    assert adapter._resolver is None


def _invoke(adapter: CrewAIAdapter, method: Method) -> RunResult | CrewOutput:
    if method == "run":
        return adapter.run(RunRequest({"question": "hello"}))
    return adapter.kickoff({"question": "hello"})


def _second_agent(crew: Crew, first: Agent, model: NetworkStub) -> Agent:
    second = Agent(
        role=first.role,
        goal=first.goal,
        backstory=first.backstory,
        llm=model,
        tools=first.tools,
        cache=False,
        max_iter=3,
        max_retry_limit=2,
        respect_context_window=False,
    )
    crew.agents.append(second)
    crew.tasks.append(
        Task(description="Second answer {question}", expected_output="A short answer", agent=second)
    )
    return second


def _ordered_policy(label: str, events: list[PolicyEvent]) -> Policy:
    def capture(context: AgentContext) -> Verdict:
        point: str = context["interception_point"]
        agent_id: str = context["agent"]["id"]
        session_id: str = context["session"]["id"]
        events.append((point, agent_id, session_id, label))
        return ALLOW

    return Policy(capture)


@pytest.mark.parametrize("use_factory", [False, True])
def test_run_owns_inputs_and_returns_actual_detached_bounded_audit(use_factory: bool) -> None:
    crew, _, model = make_crew()
    loops: list[asyncio.AbstractEventLoop] = []

    def capture_loop(context: AgentContext) -> Verdict:
        loops.append(asyncio.get_running_loop())
        return ALLOW

    policy = Policy(capture_loop)
    adapter = (
        _native(
            create_adapter(
                backend=Backend.CREWAI, runtime=crew, interceptors=[policy], max_records=3
            )
        )
        if use_factory
        else CrewAIAdapter(crew, interceptors=[policy], max_records=3)
    )
    nested: dict[str, object] = {"values": ["original"]}
    payload: dict[str, object] = {"question": "original", "nested": nested}
    request = RunRequest(payload)
    payload["question"] = "changed"
    nested["values"] = ["changed"]
    detached = request.inputs
    assert detached is not None
    detached["question"] = "tampered"
    hooks = _hooks()
    result = adapter.run(request)
    assert isinstance(result, RunResult)
    assert result.output == "completed"
    assert len(model.seen) == 1
    assert len(policy.contexts) == 6
    assert policy.contexts[1]["target"]["content"] == {
        "question": "original",
        "nested": {"values": ["original"]},
    }
    assert "original" in str(model.seen[0])
    assert isinstance(result.records, tuple)
    assert all(isinstance(record, InterceptionRecord) for record in result.records)
    assert len(result.records) == 3
    assert result.records_dropped == adapter.records_dropped == 3
    assert len(result.records) + result.records_dropped == len(policy.contexts)
    assert result.records == adapter.records
    assert [record.interception_point.value for record in result.records] == [
        context["interception_point"] for context in policy.contexts[-3:]
    ]
    assert [record.session_id for record in result.records] == [
        context["session"]["id"] for context in policy.contexts[-3:]
    ]
    assert [record.sequence for record in result.records] == [
        context["sequence"] for context in policy.contexts[-3:]
    ]
    record = result.records[0]
    original = adapter.records[0]
    assert record is not original
    assert record.trace is not None
    record.trace["trace_id"] = "tampered"
    assert adapter.records[0].trace == original.trace
    assert adapter.records[0].trace != record.trace
    assert loops and all(loop is loops[0] and loop.is_closed() for loop in loops)
    assert adapter.cleanup_error is None
    _assert_cleaned(adapter, hooks)


@pytest.mark.parametrize("request_kind", ["omitted", "none", "default", "empty"])
def test_optional_run_request_uses_the_actual_runner(request_kind: str) -> None:
    crew, _, model = make_crew()
    crew.tasks[0].description = "Answer directly"
    policy = Policy()
    adapter = _native(create_adapter(backend=Backend.CREWAI, runtime=crew, interceptors=[policy]))
    hooks = _hooks()
    if request_kind == "omitted":
        result = adapter.run()
    elif request_kind == "none":
        result = adapter.run(None)
    else:
        result = adapter.run(RunRequest({}) if request_kind == "empty" else RunRequest())
    assert result.output == "completed"
    assert len(model.seen) == 1
    assert policy.contexts[1]["target"]["content"] == {}
    assert len(result.records) == 6 and result.records_dropped == 0
    _assert_cleaned(adapter, hooks)


def test_shared_policies_precede_isolated_same_role_agent_policies_on_model_and_tools() -> None:
    effects: list[str] = []

    @tool
    def echo(value: str) -> str:
        """Record the actual native tool argument."""
        effects.append(value)
        return value

    first_model = NetworkStub(
        native=True, answers=[tool_calls("first"), "Final Answer: first completed"]
    )
    crew, first, _ = make_crew(first_model, tools=[echo], cache=False)
    second_model = NetworkStub(
        native=True, answers=[tool_calls("second"), "Final Answer: second completed"]
    )
    second = _second_agent(crew, first, second_model)
    assert first.role == second.role and first.id != second.id
    events: list[PolicyEvent] = []
    shared_one = _ordered_policy("shared-one", events)
    shared_two = _ordered_policy("shared-two", events)
    first_policy = _ordered_policy("first", events)
    second_policy = _ordered_policy("second", events)
    adapter = _native(
        create_adapter(
            backend=Backend.CREWAI,
            runtime=crew,
            interceptors=[shared_one, shared_two],
            agents={"application-writer": first, "application-reviewer": second},
            agent_policies=[
                AgentPolicy("application-writer", (first_policy,)),
                AgentPolicy("application-reviewer", (second_policy,)),
            ],
        )
    )
    hooks = _hooks()
    result = adapter.run(RunRequest({"question": "hello"}))
    assert result.output == shared_one.contexts[-2]["target"]["content"]
    assert result.output == "Final Answer: second completed"
    assert effects == ["first", "second"]
    assert len(first_model.seen) == len(second_model.seen) == 2
    agent_points = {"pre_model_call", "post_model_call", "pre_tool_call", "post_tool_call"}
    native_labels = {str(first.id): "first", str(second.id): "second"}
    for key, emission in groupby(events, key=lambda event: event[:3]):
        point, agent_id, _ = key
        expected = ["shared-one", "shared-two"]
        if point in agent_points:
            expected.append(native_labels[agent_id])
        assert [event[3] for event in emission] == expected
    for agent, policy in ((first, first_policy), (second, second_policy)):
        assert {context["agent"]["id"] for context in policy.contexts} == {str(agent.id)}
        assert {context["interception_point"] for context in policy.contexts} == agent_points
    assert {context["session"]["id"] for context in first_policy.contexts}.isdisjoint(
        {context["session"]["id"] for context in second_policy.contexts}
    )
    assert len({context["session"]["id"] for context in shared_one.contexts}) == 3
    assert first.id == crew.agents[0].id and second.id == crew.agents[1].id
    _assert_cleaned(adapter, hooks)


def test_factory_snapshots_shared_policies_agent_policies_and_alias_bindings() -> None:
    crew, first, _ = make_crew()
    second = _second_agent(crew, first, NetworkStub())
    shared, one, two, replacement = Policy(), Policy(), Policy(), Policy()
    shared_policies = [shared]
    bindings: dict[str, object] = {"one": first, "two": second}
    policies = [AgentPolicy("one", (one,)), AgentPolicy("two", (two,))]
    adapter = _native(
        create_adapter(
            backend=Backend.CREWAI,
            runtime=crew,
            interceptors=shared_policies,
            agents=bindings,
            agent_policies=policies,
        )
    )
    shared_policies[:] = [replacement]
    bindings["one"], bindings["two"] = second, first
    bindings.clear()
    policies[:] = [AgentPolicy("one", (replacement,))]
    hooks = _hooks()
    assert adapter.run(RunRequest({"question": "hello"})).output == "completed"
    assert shared.contexts and one.contexts and two.contexts
    assert {context["agent"]["id"] for context in one.contexts} == {str(first.id)}
    assert {context["agent"]["id"] for context in two.contexts} == {str(second.id)}
    assert not replacement.contexts
    _assert_cleaned(adapter, hooks)


@pytest.mark.parametrize(
    "kind",
    [
        "unknown",
        "missing-bindings",
        "foreign",
        "unused-foreign",
        "duplicate",
        "unused-duplicate",
        "duplicate-policy",
    ],
)
def test_factory_rejects_invalid_bindings_before_model_or_policy_io(kind: str) -> None:
    crew, agent, model = make_crew()
    _, foreign, foreign_model = make_crew()
    shared, bound = Policy(), Policy()
    bindings: dict[str, object] | None = {"writer": agent}
    policies = [AgentPolicy("writer", (bound,))]
    if kind == "unknown":
        policies = [AgentPolicy("unknown", (bound,))]
    elif kind == "missing-bindings":
        bindings = None
    elif kind in {"foreign", "unused-foreign"}:
        bindings = {"writer": foreign}
    elif kind == "duplicate-policy":
        policies = [AgentPolicy("writer", (bound,)), AgentPolicy("writer", (bound,))]
    else:
        bindings = {"writer": agent, "other-alias": agent}
    if kind.startswith("unused-"):
        policies = []
    hooks = _hooks()
    with pytest.raises(ValueError):
        create_adapter(
            backend=Backend.CREWAI,
            runtime=crew,
            interceptors=[shared],
            agents=bindings,
            agent_policies=policies,
        )
    assert not model.seen and not foreign_model.seen
    assert not shared.contexts and not bound.contexts
    assert _hooks() == hooks


@pytest.mark.parametrize("kind", ["empty-list", "empty-tuple", "iterable-pairs", "scalar"])
def test_factory_rejects_non_mapping_agents_before_model_or_policy_io(kind: str) -> None:
    crew, agent, model = make_crew()
    invalid_bindings: dict[str, object] = {
        "empty-list": [],
        "empty-tuple": (),
        "iterable-pairs": [("worker", agent)],
        "scalar": 42,
    }
    policy = Policy()
    hooks = _hooks()
    with pytest.raises(TypeError):
        create_adapter(
            backend=Backend.CREWAI,
            runtime=crew,
            interceptors=[policy],
            agents=invalid_bindings[kind],  # pyright: ignore[reportArgumentType]
        )
    assert not model.seen and not policy.contexts
    assert _hooks() == hooks


@pytest.mark.parametrize("backend", ["", "crewai", "CrewAI", "openai", "langgraph", None, 1])
def test_invalid_backend_has_no_native_fallback_or_model_io(backend: object) -> None:
    crew, _, model = make_crew()
    policy = Policy()
    hooks = _hooks()
    with pytest.raises(UnsupportedBackendError):
        create_adapter(
            backend=backend,  # pyright: ignore[reportArgumentType]
            runtime=crew,
            interceptors=[policy],
        )
    assert not model.seen and not policy.contexts
    assert _hooks() == hooks


def test_request_identity_fields_cannot_replace_native_agent_session_or_policy_binding() -> None:
    crew, first, _ = make_crew()
    second = _second_agent(crew, first, NetworkStub())
    shared, one, two = Policy(), Policy(), Policy()
    adapter = _native(
        create_adapter(
            backend=Backend.CREWAI,
            runtime=crew,
            interceptors=[shared],
            agents={"writer": first, "reviewer": second},
            agent_policies=[AgentPolicy("writer", (one,)), AgentPolicy("reviewer", (two,))],
        )
    )
    inputs: dict[str, object] = {
        "question": "hello",
        "agent": {"id": str(second.id), "framework": "forged"},
        "agent_id": "reviewer",
        "session": {"id": "forged-session"},
        "session_id": "forged-session",
        "trace": {"trace_id": "forged-trace"},
        "role": first.role,
        "model": "reviewer",
    }
    hooks = _hooks()
    adapter.run(RunRequest(inputs))
    assert shared.contexts[1]["target"]["content"] == inputs
    assert {context["agent"]["id"] for context in one.contexts} == {str(first.id)}
    assert {context["agent"]["id"] for context in two.contexts} == {str(second.id)}
    assert all(context["agent"]["framework"] == "crewai" for context in shared.contexts)
    assert all(context["session"]["id"] != "forged-session" for context in shared.contexts)
    assert all(context["trace"]["trace_id"] != "forged-trace" for context in shared.contexts)
    assert {context["session"]["id"] for context in one.contexts}.isdisjoint(
        {context["session"]["id"] for context in two.contexts}
    )
    _assert_cleaned(adapter, hooks)


def test_actual_logically_bound_denial_is_catchable_as_common_and_native_hook_abort() -> None:
    crew, agent, model = make_crew()
    shared, bound = Policy(), denial("pre_model_call")
    adapter = _native(
        create_adapter(
            backend=Backend.CREWAI,
            runtime=crew,
            interceptors=[shared],
            agents={"application-policy": agent},
            agent_policies=[AgentPolicy("application-policy", (bound,))],
        )
    )
    hooks = _hooks()
    with pytest.raises(
        AdapterBlocked, match=r"^CrewAI execution blocked by governance\.$"
    ) as caught:
        adapter.run(RunRequest({"question": "private prompt"}))
    assert isinstance(caught.value, NativeAdapterBlocked)
    assert isinstance(caught.value, HookAborted)
    assert isinstance(caught.value, RuntimeError)
    assert caught.value.__cause__ is None
    assert not model.seen
    assert {context["agent"]["id"] for context in bound.contexts} == {str(agent.id)}
    assert shared.contexts[-1]["interception_point"] == "agent_shutdown"
    assert any(record.verdict.decision == Decision.DENY for record in adapter.records)
    _assert_cleaned(adapter, hooks)
    for method in ("run", "kickoff"):
        with pytest.raises(RuntimeError, match="single-use"):
            _invoke(adapter, method)
    assert not model.seen


def test_portable_engine_error_fails_closed_without_sensitive_cause_or_model_io(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def broken(self: InterceptionEmitter, context: AgentContext) -> object:
        raise RuntimeError("private engine detail")

    monkeypatch.setattr(InterceptionEmitter, "emit", broken)
    crew, _, model = make_crew()
    policy = Policy()
    adapter = _native(create_adapter(backend=Backend.CREWAI, runtime=crew, interceptors=[policy]))
    hooks = _hooks()
    with pytest.raises(AdapterBlocked) as caught:
        adapter.run(RunRequest({"question": "private prompt"}))
    assert isinstance(caught.value, HookAborted)
    assert caught.value.__cause__ is None
    assert "private" not in str(caught.value)
    assert "private engine detail" not in caplog.text
    assert not model.seen
    assert [context["interception_point"] for context in policy.contexts] == ["agent_shutdown"]
    assert [record.interception_point.value for record in adapter.records] == ["agent_shutdown"]
    assert any(record.msg == "adapter_hook_denied" for record in caplog.records)
    _assert_cleaned(adapter, hooks)


def test_legacy_native_agent_policy_and_kickoff_still_return_crew_output() -> None:
    crew, agent, model = make_crew()
    shared, bound = Policy(), Policy()
    adapter = CrewAIAdapter(
        crew, interceptors=[shared], agent_policies=[NativeAgentPolicy(agent, (bound,))]
    )
    hooks = _hooks()
    output = adapter.kickoff({"question": "hello"})
    assert isinstance(output, CrewOutput)
    assert output.raw == output.tasks_output[-1].raw == "completed"
    assert len(model.seen) == 1
    assert {context["agent"]["id"] for context in bound.contexts} == {str(agent.id)}
    assert len(adapter.records) == 6
    _assert_cleaned(adapter, hooks)
    with pytest.raises(RuntimeError, match="single-use"):
        adapter.run(RunRequest({"question": "hello"}))


@pytest.mark.parametrize("first", ["run", "kickoff"])
@pytest.mark.parametrize("outcome", ["success", "denied", "admission"])
def test_run_and_kickoff_share_single_use_including_denial_and_admission_failure(
    first: Method, outcome: str
) -> None:
    crew, _, model = make_crew()
    policy = denial("input") if outcome == "denied" else Policy()
    if outcome == "admission":
        crew.planning = True
    adapter = CrewAIAdapter(crew, interceptors=[policy])
    hooks = _hooks()
    if outcome == "denied":
        with pytest.raises(AdapterBlocked):
            _invoke(adapter, first)
    elif outcome == "admission":
        with pytest.raises(ValueError, match="Unsupported"):
            _invoke(adapter, first)
        assert not policy.contexts and adapter._worker is None
    else:
        output = _invoke(adapter, first)
        if first == "run":
            assert isinstance(output, RunResult) and output.output == "completed"
        else:
            assert isinstance(output, CrewOutput) and output.raw == "completed"
    expected_calls = 1 if outcome == "success" else 0
    assert len(model.seen) == expected_calls
    _assert_cleaned(adapter, hooks)
    for second in ("run", "kickoff"):
        with pytest.raises(RuntimeError, match="single-use"):
            _invoke(adapter, second)
    assert len(model.seen) == expected_calls


@pytest.mark.parametrize("invalid_request", [{}, [], "hello", object()])
def test_invalid_run_request_type_fails_before_io_and_does_not_consume_a_valid_run(
    invalid_request: object,
) -> None:
    crew, _, model = make_crew()
    policy = Policy()
    adapter = CrewAIAdapter(crew, interceptors=[policy])
    hooks = _hooks()
    with pytest.raises(TypeError):
        adapter.run(invalid_request)  # pyright: ignore[reportArgumentType]
    assert not model.seen and not policy.contexts and adapter._worker is None
    assert _hooks() == hooks
    assert adapter.run(RunRequest({"question": "hello"})).output == "completed"
    assert len(model.seen) == 1
    _assert_cleaned(adapter, hooks)


def test_portable_run_preserves_original_failure_and_reports_common_cleanup_error() -> None:
    original = ValueError("original model failure")
    crew, agent, _ = make_crew(NetworkStub(answers=[original]))
    agent.max_retry_limit = 0
    entered, release = threading.Event(), threading.Event()
    loops: list[asyncio.AbstractEventLoop] = []

    def hold_shutdown(context: AgentContext) -> Verdict:
        loops.append(asyncio.get_running_loop())
        if context["interception_point"] == "agent_shutdown":
            entered.set()
            assert release.wait(timeout=5)
        return ALLOW

    adapter = _native(
        create_adapter(
            backend=Backend.CREWAI,
            runtime=crew,
            interceptors=[Policy(hold_shutdown)],
            emit_timeout=0.1,
        )
    )
    hooks = _hooks()
    try:
        with pytest.raises(ValueError) as caught:
            adapter.run(RunRequest({"question": "hello"}))
        assert caught.value is original and entered.is_set()
        assert isinstance(adapter.cleanup_error, AdapterCleanupError)
        assert isinstance(adapter.cleanup_error, NativeAdapterCleanupError)
        notes: list[str] = getattr(original, "__notes__", [])
        assert any(note == str(adapter.cleanup_error) for note in notes)
        assert all("original model failure" not in note for note in notes)
        assert not adapter._pending and not adapter._agents
    finally:
        release.set()
        worker = adapter._worker
        assert worker is not None
        worker._thread.join(timeout=2)
        assert not worker._thread.is_alive()
        worker.close()
    assert loops and all(loop.is_closed() for loop in loops)
    _assert_cleaned(adapter, hooks)


@pytest.mark.parametrize("methods", [("run", "run"), ("run", "kickoff"), ("kickoff", "kickoff")])
def test_concurrent_calls_admit_exactly_one_actual_run(methods: tuple[Method, Method]) -> None:
    start = threading.Barrier(2)
    entered, release = threading.Event(), threading.Event()

    def hold_model() -> None:
        entered.set()
        assert release.wait(timeout=15)

    crew, _, model = make_crew(NetworkStub(on_call=hold_model))
    policy = Policy()
    adapter = CrewAIAdapter(crew, interceptors=[policy])
    hooks = _hooks()

    def compete(method: Method) -> RunResult | CrewOutput:
        thread_hooks = _hooks()
        start.wait(timeout=5)
        try:
            return _invoke(adapter, method)
        finally:
            assert _hooks() == thread_hooks

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(compete, method) for method in methods]
        try:
            assert entered.wait(timeout=5)
            done, pending = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
            assert len(done) == len(pending) == 1
            with pytest.raises(RuntimeError, match="single-use"):
                next(iter(done)).result(timeout=1)
            winner = next(iter(pending))
        finally:
            release.set()
        output = winner.result(timeout=5)
    if isinstance(output, RunResult):
        assert output.output == "completed"
    else:
        assert output.raw == "completed"
    assert len(model.seen) == 1
    assert len(policy.contexts) == len(adapter.records) == 6
    assert adapter.records_dropped == 0
    _assert_cleaned(adapter, hooks)
