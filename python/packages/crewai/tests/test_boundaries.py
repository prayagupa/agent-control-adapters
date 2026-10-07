"""Transform, approval, independent-crew, and failure-path integration coverage."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from agent_hooks import (
    ALLOW,
    AgentContext,
    ApprovalOutcome,
    ApprovalRequest,
    ApprovalResolution,
    Decision,
    InterceptionEmitter,
    Transform,
    Verdict,
)
from agent_hooks_crewai import AdapterBlocked, AdapterCleanupError, AgentPolicy, CrewAIAdapter
from crewai.hooks.contexts import InputContext, OutputContext
from crewai.hooks.dispatch import InterceptionPoint, dispatch, get_scoped_hooks, scoped_hooks
from crewai.hooks.llm_hooks import LLMCallHookContext
from crewai.tools import tool
from crewai_adapter_test_helpers import NetworkStub, Policy, denial, make_crew, tool_calls
from pydantic import BaseModel


def test_input_and_model_transforms_reach_actual_consumer() -> None:
    seen: list[str] = []

    def inspect_model() -> None:
        seen.append(str(llm.seen[-1]))

    llm = NetworkStub(on_call=inspect_model)
    crew, _, _ = make_crew(llm)

    def control(ctx: AgentContext) -> Verdict:
        match ctx["interception_point"]:
            case "input":
                return Verdict(
                    Decision.TRANSFORM,
                    transform=Transform("$target.content.question", "input changed"),
                )
            case "pre_model_call":
                return Verdict(
                    Decision.TRANSFORM,
                    transform=Transform("$target[0].content", "model prompt changed"),
                )
            case "post_model_call":
                return Verdict(
                    Decision.TRANSFORM,
                    transform=Transform("$target.content", "Final Answer: response changed"),
                )
        return ALLOW

    output = CrewAIAdapter(crew, interceptors=[Policy(control)]).kickoff({"question": "original"})
    assert output.raw == "response changed"
    assert "input changed" in seen[0] and "model prompt changed" in seen[0]


def test_native_tool_transforms_reach_tool_and_followup_model() -> None:
    effects: list[str] = []
    observed_results: list[str] = []

    @tool
    def echo(value: str) -> str:
        """Echo the input text."""
        effects.append(value)
        return value

    llm = NetworkStub(native=True, answers=[tool_calls("original"), "Final Answer: complete"])
    llm.on_call = lambda: observed_results.append(str(llm.seen[-1]))
    crew, _, _ = make_crew(llm, tools=[echo], cache=False)

    def control(ctx: AgentContext) -> Verdict:
        if ctx["interception_point"] == "pre_tool_call":
            return Verdict(Decision.TRANSFORM, transform=Transform("$target.value", "changed args"))
        if ctx["interception_point"] == "post_tool_call":
            return Verdict(Decision.TRANSFORM, transform=Transform("$target", "changed result"))
        return ALLOW

    CrewAIAdapter(crew, interceptors=[Policy(control)]).kickoff({"question": "hello"})
    assert effects == ["changed args"]
    assert "changed result" in observed_results[-1]


def test_cached_tool_argument_transform_fails_closed() -> None:
    effects: list[str] = []

    @tool
    def echo(value: str) -> str:
        """Echo the input text."""
        effects.append(value)
        return value

    from crewai.agents.cache.cache_handler import CacheHandler

    llm = NetworkStub(native=True, answers=[tool_calls("original"), "Final Answer: complete"])
    crew, agent, _ = make_crew(llm, tools=[echo])
    agent.tools_handler.cache = CacheHandler()
    policy = Policy(
        lambda ctx: (
            Verdict(Decision.TRANSFORM, transform=Transform("$target.value", "changed"))
            if ctx["interception_point"] == "pre_tool_call"
            else ALLOW
        )
    )
    with pytest.raises(AdapterBlocked):
        CrewAIAdapter(crew, interceptors=[policy]).kickoff({"question": "hello"})
    assert not effects


class Approve:
    def __init__(self) -> None:
        self.requests: list[ApprovalRequest] = []

    def resolve(self, request: ApprovalRequest) -> ApprovalResolution:
        self.requests.append(request)
        return ApprovalResolution(ApprovalOutcome.APPROVE, request.context_identity, ALLOW)


@pytest.mark.parametrize("later_denies", [False, True])
def test_approval_never_skips_later_per_agent_control(later_denies: bool) -> None:
    crew, agent, llm = make_crew()
    ordering: list[str] = []

    def common(ctx: AgentContext) -> Verdict:
        if ctx["interception_point"] == "pre_model_call":
            ordering.append("shared")
            return Verdict.escalate(reason="test:approval")
        return ALLOW

    def bound(ctx: AgentContext) -> Verdict:
        if ctx["interception_point"] == "pre_model_call":
            ordering.append("bound")
            return Verdict.deny(reason="test:final") if later_denies else ALLOW
        return ALLOW

    resolver = Approve()
    adapter = CrewAIAdapter(
        crew,
        interceptors=[Policy(common)],
        agent_policies=[AgentPolicy(agent, (Policy(bound),))],
        resolver=resolver,
    )
    if later_denies:
        with pytest.raises(AdapterBlocked):
            adapter.kickoff({"question": "hello"})
        assert not llm.seen and not resolver.requests
    else:
        adapter.kickoff({"question": "hello"})
        assert len(resolver.requests) == 1
        assert resolver.requests[0].context_identity is not None
    assert ordering == ["shared", "bound"]
    assert all(
        record.composition.profile.value == "sequential/run_all" for record in adapter.records
    )


def test_independent_concurrent_crews_keep_scopes_and_sessions_separate() -> None:
    rendezvous = threading.Barrier(2)
    first, _, _ = make_crew(NetworkStub(on_call=lambda: rendezvous.wait(timeout=3)))
    second, _, _ = make_crew(NetworkStub(on_call=lambda: rendezvous.wait(timeout=3)))
    one, two = Policy(), Policy()
    a = CrewAIAdapter(first, interceptors=[one])
    b = CrewAIAdapter(second, interceptors=[two])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(adapter.kickoff, {"question": "hello"}) for adapter in (a, b)]
        assert all(future.result(timeout=5).raw == "completed" for future in futures)
    assert {ctx["session"]["id"] for ctx in one.contexts}.isdisjoint(
        {ctx["session"]["id"] for ctx in two.contexts}
    )
    assert len(a.records) == len(b.records) == 6


def test_original_model_failure_preserved_when_shutdown_engine_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ValueError("original model failure")
    llm = NetworkStub(answers=[original])
    crew, agent, _ = make_crew(llm)
    agent.max_retry_limit = 0
    actual_emit = InterceptionEmitter.emit_unchecked

    async def fail_shutdown(self: InterceptionEmitter, ctx: AgentContext) -> Any:
        if ctx["interception_point"] == "agent_shutdown":
            raise RuntimeError("shutdown failed")
        return await actual_emit(self, ctx)

    monkeypatch.setattr(InterceptionEmitter, "emit_unchecked", fail_shutdown)
    adapter = CrewAIAdapter(crew, interceptors=[Policy()])
    with pytest.raises(ValueError) as caught:
        adapter.kickoff({"question": "hello"})
    assert caught.value is original
    assert adapter._worker is not None and not adapter._worker._thread.is_alive()


def test_nonpreemptible_shutdown_preserves_original_and_reports_cleanup() -> None:
    original = ValueError("original model failure")
    crew, agent, _ = make_crew(NetworkStub(answers=[original]))
    agent.max_retry_limit = 0
    release = threading.Event()
    entered = threading.Event()

    def control(ctx: AgentContext) -> Verdict:
        if ctx["interception_point"] == "agent_shutdown":
            entered.set()
            assert release.wait(3)
        return ALLOW

    adapter = CrewAIAdapter(crew, interceptors=[Policy(control)], emit_timeout=0.1)
    try:
        with pytest.raises(ValueError) as caught:
            adapter.kickoff({"question": "hello"})
        assert caught.value is original and entered.is_set()
        assert isinstance(adapter.cleanup_error, AdapterCleanupError)
        assert any("cleanup incomplete" in note for note in original.__notes__)
        assert not adapter._agents and not adapter._pending
    finally:
        release.set()
        assert adapter._worker is not None
        adapter._worker._thread.join(timeout=2)
        adapter._worker.close()


@pytest.mark.parametrize("missing", [False, True])
def test_agent_identity_is_required_even_with_identical_role(missing: bool) -> None:
    _, foreign, _ = make_crew()
    llm = NetworkStub()
    crew, _, _ = make_crew(llm)
    llm.on_call = lambda: dispatch(
        InterceptionPoint.PRE_MODEL_CALL,
        LLMCallHookContext(
            crew=crew,
            agent=None if missing else foreign,
            llm=llm,
            messages=[{"role": "user", "content": "test"}],
        ),
    )
    with pytest.raises(AdapterBlocked):
        CrewAIAdapter(crew, interceptors=[Policy()]).kickoff({"question": "hello"})
    assert len(llm.seen) == 1


def test_existing_scoped_hooks_preserved_and_nested_adapter_cannot_hide_parent() -> None:
    other, _, other_llm = make_crew()
    inner = CrewAIAdapter(other, interceptors=[Policy()])
    crew, _, _ = make_crew(NetworkStub(on_call=lambda: inner.kickoff({"question": "nested"})))
    outer = CrewAIAdapter(crew, interceptors=[Policy()])
    observed: list[object] = []

    def hook(ctx: object) -> None:
        observed.append(ctx)

    with scoped_hooks({InterceptionPoint.INPUT: [hook]}):
        with pytest.raises(AdapterBlocked):
            outer.kickoff({"question": "hello"})
        assert get_scoped_hooks(InterceptionPoint.INPUT) == [hook]
    assert len(observed) == 1 and not other_llm.seen


def test_invalid_context_class_never_fails_open() -> None:
    llm = NetworkStub()
    crew, _, _ = make_crew(llm)
    llm.on_call = lambda: dispatch(InterceptionPoint.OUTPUT, InputContext(crew=crew, payload={}))
    with pytest.raises(AdapterBlocked):
        CrewAIAdapter(crew, interceptors=[Policy()]).kickoff({"question": "hello"})


class StructuredAnswer(BaseModel):
    text: str


@pytest.mark.parametrize("point", ["post_model_call", "output"])
def test_structured_output_transformation_rejected(point: str) -> None:
    llm = NetworkStub(answers=[StructuredAnswer(text="original")])
    crew, _, _ = make_crew(llm)
    crew.tasks[0].output_pydantic = StructuredAnswer
    policy = Policy(
        lambda ctx: (
            Verdict(Decision.TRANSFORM, transform=Transform("$target.content", "changed"))
            if ctx["interception_point"] == point
            else ALLOW
        )
    )
    with pytest.raises(ValueError, match="output_pydantic"):
        CrewAIAdapter(crew, interceptors=[policy]).kickoff({"question": "hello"})
    assert not llm.seen and not policy.contexts


def test_last_nonempty_task_output_is_the_transformed_view() -> None:
    from crewai.crews.crew_output import CrewOutput
    from crewai.tasks.task_output import TaskOutput

    output = CrewOutput(
        raw="original",
        tasks_output=[
            TaskOutput(description="first", expected_output="text", raw="original", agent="a"),
            TaskOutput(description="last", expected_output="text", raw="", agent="a"),
        ],
    )
    llm = NetworkStub()
    crew, _, _ = make_crew(llm)
    llm.on_call = lambda: dispatch(
        InterceptionPoint.OUTPUT, OutputContext(crew=crew, payload=output, output=output)
    )
    policy = Policy(
        lambda ctx: (
            Verdict(Decision.TRANSFORM, transform=Transform("$target.content", "changed"))
            if ctx["interception_point"] == "output"
            else ALLOW
        )
    )
    CrewAIAdapter(crew, interceptors=[policy]).kickoff({"question": "hello"})
    assert output.raw == output.tasks_output[0].raw == "changed"
    assert output.tasks_output[1].raw == ""


@pytest.mark.parametrize("point", ["agent_startup", "pre_tool_call", "post_tool_call"])
def test_unsupported_startup_and_tool_transforms(point: str) -> None:
    @tool
    def echo(value: str) -> str:
        """Echo text."""
        return value

    llm = NetworkStub(native=True, answers=[tool_calls("hello"), "Final Answer: complete"])
    crew, _, _ = make_crew(llm, tools=[echo])
    policy = Policy(
        lambda ctx: (
            Verdict(Decision.TRANSFORM, transform=Transform("$target", ["unsupported"]))
            if ctx["interception_point"] == point
            else ALLOW
        )
    )
    with pytest.raises(AdapterBlocked):
        CrewAIAdapter(crew, interceptors=[policy]).kickoff({"question": "hello"})


def test_sticky_denial_cannot_return_success_after_host_swallows_it() -> None:
    llm = NetworkStub()
    crew, _, _ = make_crew(llm)

    def swallowed_denial() -> None:
        with pytest.raises(AdapterBlocked):
            dispatch(InterceptionPoint.INPUT, InputContext(crew=crew, payload={"deny": True}))

    llm.on_call = swallowed_denial
    policy = Policy(
        lambda ctx: (
            Verdict.deny(reason="test:sticky")
            if ctx["interception_point"] == "input" and ctx["target"]["content"].get("deny")
            else ALLOW
        )
    )
    with pytest.raises(AdapterBlocked):
        CrewAIAdapter(crew, interceptors=[policy]).kickoff({"question": "hello"})
    assert len(llm.seen) == 1


def test_shutdown_denial_is_observation_only() -> None:
    crew, _, _ = make_crew()
    policy = Policy(
        lambda ctx: (
            Verdict.escalate(reason="test:shutdown")
            if ctx["interception_point"] == "agent_shutdown"
            else ALLOW
        )
    )
    resolver = Approve()
    adapter = CrewAIAdapter(crew, interceptors=[policy], resolver=resolver)
    assert adapter.kickoff({"question": "hello"}).raw == "completed"
    assert adapter.records[-1].verdict.decision == Decision.DENY
    assert not resolver.requests


def test_diagnostics_and_sdk_records_share_host_trace(caplog: pytest.LogCaptureFixture) -> None:
    crew, _, _ = make_crew()
    adapter = CrewAIAdapter(crew, interceptors=[denial("pre_model_call")])
    with pytest.raises(AdapterBlocked):
        adapter.kickoff({"question": "private test prompt"})
    records = adapter.records
    trace_ids = {record.trace["trace_id"] for record in records if record.trace is not None}
    assert len(trace_ids) == 1
    diagnostics = [record for record in caplog.records if record.msg == "adapter_hook_denied"]
    assert diagnostics and diagnostics[0].trace_id in trace_ids
    assert diagnostics[0].point == "pre_model_call"
    assert diagnostics[0].error_type == "AdapterBlocked"
    assert "private test prompt" not in caplog.text
    assert records[0].trace is not None
    records[0].trace["trace_id"] = "tampered"
    assert adapter.records[0].trace != records[0].trace


def test_real_parallel_deny_stays_sticky_after_native_runner_catches_abort() -> None:
    started, release = threading.Event(), threading.Event()
    effects: list[str] = []
    admissions: list[str] = []

    @tool
    def echo(value: str) -> str:
        """Signal entry and wait for the sibling policy decision."""
        effects.append(value)
        started.set()
        assert release.wait(3)
        return value

    def control(ctx: AgentContext) -> Verdict:
        if ctx["interception_point"] == "pre_tool_call":
            admissions.append(ctx["tool_call"]["id"])
            if len(admissions) == 2:
                assert started.wait(2)
                release.set()
                return Verdict.deny(reason="test:parallel")
        return ALLOW

    llm = NetworkStub(native=True, answers=[tool_calls("one", "two"), "Final Answer: complete"])
    crew, _, _ = make_crew(llm, tools=[echo])
    adapter = CrewAIAdapter(crew, interceptors=[Policy(control)])
    try:
        with pytest.raises(AdapterBlocked):
            adapter.kickoff({"question": "hello"})
    finally:
        release.set()
    assert len(admissions) == 2 and len(effects) == 1
    assert len(llm.seen) == 1


def test_real_model_boundary_timeout_never_applies_late_target() -> None:
    entered, release = threading.Event(), threading.Event()
    host_messages: list[list[dict[str, object]]] = []

    def remember(ctx: LLMCallHookContext) -> None:
        host_messages.append(ctx.messages)

    def control(ctx: AgentContext) -> Verdict:
        if ctx["interception_point"] == "pre_model_call":
            entered.set()
            assert release.wait(3)
            return Verdict(Decision.TRANSFORM, transform=Transform("$target[0].content", "late"))
        return ALLOW

    crew, _, llm = make_crew()
    adapter = CrewAIAdapter(crew, interceptors=[Policy(control)], emit_timeout=0.1)
    try:
        with (
            scoped_hooks({InterceptionPoint.PRE_MODEL_CALL: [remember]}),
            pytest.raises(AdapterBlocked),
        ):
            adapter.kickoff({"question": "hello"})
        assert entered.is_set() and not llm.seen
        assert isinstance(adapter.cleanup_error, AdapterCleanupError)
    finally:
        release.set()
        assert adapter._worker is not None
        adapter._worker._thread.join(timeout=2)
        adapter._worker.close()
    assert host_messages[0][0]["content"] != "late"
    assert not llm.seen
