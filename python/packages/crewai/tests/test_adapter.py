"""Real Crew.kickoff and released SDK integration tests."""

from __future__ import annotations

import threading
from typing import Any

import pytest
from agent_hooks import (
    ALLOW,
    AgentContext,
    Decision,
    InterceptionEmitter,
    Transform,
    Verdict,
)
from agent_hooks_crewai import AdapterBlocked, AgentPolicy, CrewAIAdapter
from crewai import Agent, Task
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.crews.crew_output import CrewOutput
from crewai.experimental.agent_executor import AgentExecutor
from crewai.hooks.contexts import InputContext
from crewai.hooks.dispatch import InterceptionPoint, dispatch, get_scoped_hooks
from crewai.tools import tool
from crewai_adapter_test_helpers import NetworkStub, Policy, denial, make_crew, tool_calls


@pytest.mark.parametrize("executor", [AgentExecutor, CrewAgentExecutor])
def test_allow_records_single_use_and_cleanup(
    executor: type[AgentExecutor] | type[CrewAgentExecutor],
) -> None:
    crew, _, llm = make_crew(executor_class=executor)
    policy = Policy()
    adapter = CrewAIAdapter(crew, interceptors=[policy])
    output = adapter.kickoff({"question": "hello"})
    assert isinstance(output, CrewOutput)
    assert output.raw == "completed"
    assert len(llm.seen) == 1
    assert [ctx["interception_point"] for ctx in policy.contexts] == [
        "agent_startup",
        "input",
        "pre_model_call",
        "post_model_call",
        "output",
        "agent_shutdown",
    ]
    assert len(adapter.records) == 6
    assert adapter.records_dropped == 0
    assert not get_scoped_hooks(InterceptionPoint.INPUT)
    assert adapter._worker is not None and not adapter._worker._thread.is_alive()
    assert not adapter._pending
    with pytest.raises(RuntimeError, match="single-use"):
        adapter.kickoff()


@pytest.mark.parametrize(
    "point, model_calls",
    [
        ("agent_startup", 0),
        ("input", 0),
        ("pre_model_call", 0),
        ("post_model_call", 1),
        ("output", 1),
    ],
)
def test_deny_never_retries(point: str, model_calls: int) -> None:
    crew, _, llm = make_crew()
    policy = denial(point)
    adapter = CrewAIAdapter(crew, interceptors=[policy])
    with pytest.raises(AdapterBlocked, match=r"^CrewAI execution blocked by governance\.$"):
        adapter.kickoff({"question": "hello"})
    assert len(llm.seen) == model_calls
    assert policy.contexts[-1]["interception_point"] == "agent_shutdown"
    assert adapter._worker is not None and not adapter._worker._thread.is_alive()


def test_output_transform_preserves_crew_output() -> None:
    crew, _, _ = make_crew()
    policy = Policy(
        lambda ctx: (
            Verdict(Decision.TRANSFORM, transform=Transform("$target.content", "changed"))
            if ctx["interception_point"] == "output"
            else ALLOW
        )
    )
    output = CrewAIAdapter(crew, interceptors=[policy]).kickoff({"question": "hello"})
    assert isinstance(output, CrewOutput)
    assert output.raw == output.tasks_output[-1].raw == "changed"


@pytest.mark.parametrize(
    "point, replacement",
    [
        ("input", {"content": [], "role": "user"}),
        ("input", {"content": {}, "role": "system"}),
        ("pre_model_call", [{"role": "root", "content": "bad"}]),
        (
            "post_model_call",
            {"content": "x", "tool_calls": [{"id": "bad"}], "finish_reason": "stop"},
        ),
        ("output", {"content": "x", "session": {"id": "spoof"}}),
    ],
)
def test_invalid_transforms_fail_closed(point: str, replacement: object) -> None:
    crew, _, _ = make_crew()
    policy = Policy(
        lambda ctx: (
            Verdict(Decision.TRANSFORM, transform=Transform("$target", replacement))
            if ctx["interception_point"] == point
            else ALLOW
        )
    )
    with pytest.raises(AdapterBlocked):
        CrewAIAdapter(crew, interceptors=[policy]).kickoff({"question": "hello"})


def test_engine_failure_is_hook_aborted(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("sensitive engine detail")

    monkeypatch.setattr(InterceptionEmitter, "emit", broken)
    crew, _, llm = make_crew()
    with pytest.raises(AdapterBlocked) as caught:
        CrewAIAdapter(crew, interceptors=[Policy()]).kickoff({"question": "hello"})
    assert "sensitive" not in str(caught.value)
    assert llm.seen == []


def test_per_agent_identity_and_isolation() -> None:
    first = NetworkStub()
    crew, agent, _ = make_crew(first)
    second_llm = NetworkStub()
    second = Agent(
        role=agent.role,
        goal=agent.goal,
        backstory=agent.backstory,
        llm=second_llm,
        respect_context_window=False,
    )
    crew.agents.append(second)
    crew.tasks.append(Task(description="Second", expected_output="Text", agent=second))
    common, one, two = Policy(), Policy(), Policy()
    adapter = CrewAIAdapter(
        crew,
        interceptors=[common],
        agent_policies=[
            AgentPolicy(agent, (one,)),
            AgentPolicy(second, (two,)),
        ],
    )
    adapter.kickoff({"question": "hello"})
    assert {ctx["agent"]["id"] for ctx in one.contexts} == {str(agent.id)}
    assert {ctx["agent"]["id"] for ctx in two.contexts} == {str(second.id)}
    sessions = {ctx["session"]["id"] for ctx in common.contexts}
    assert len(sessions) == 3
    assert all(
        ctx["interception_point"] in ("pre_model_call", "post_model_call")
        for ctx in one.contexts + two.contexts
    )


@pytest.mark.parametrize("foreign", [False, True])
def test_missing_or_foreign_crew_identity(foreign: bool) -> None:
    other, _, _ = make_crew()
    llm = NetworkStub(
        on_call=lambda: dispatch(
            InterceptionPoint.INPUT, InputContext(crew=other if foreign else None, payload={})
        )
    )
    crew, _, _ = make_crew(llm)
    with pytest.raises(AdapterBlocked):
        CrewAIAdapter(crew, interceptors=[Policy()]).kickoff({"question": "hello"})
    assert len(llm.seen) == 1


def test_nested_foreign_crew_fails_closed() -> None:
    other, _, other_model = make_crew()
    llm = NetworkStub(on_call=lambda: other.kickoff(inputs={"question": "nested"}))
    crew, _, _ = make_crew(llm)
    with pytest.raises(AdapterBlocked):
        CrewAIAdapter(crew, interceptors=[Policy()]).kickoff({"question": "hello"})
    assert not other_model.seen


@pytest.mark.parametrize("point, executed", [("pre_tool_call", False), ("post_tool_call", True)])
@pytest.mark.parametrize("executor", [AgentExecutor, CrewAgentExecutor])
def test_native_tool_denial_is_sticky(
    point: str, executed: bool, executor: type[AgentExecutor] | type[CrewAgentExecutor]
) -> None:
    effects: list[str] = []

    @tool
    def echo(value: str) -> str:
        """Return the input text."""
        effects.append(value)
        return value

    llm = NetworkStub(native=True, answers=[tool_calls("hello"), "Final Answer: completed"])
    crew, _, _ = make_crew(llm, tools=[echo], executor_class=executor)
    adapter = CrewAIAdapter(crew, interceptors=[denial(point)])
    with pytest.raises(AdapterBlocked):
        adapter.kickoff({"question": "hello"})
    assert bool(effects) is executed
    assert len(llm.seen) == 1


@pytest.mark.parametrize("executor", [AgentExecutor, CrewAgentExecutor])
def test_real_native_parallel_tool_pairing(
    executor: type[AgentExecutor] | type[CrewAgentExecutor],
) -> None:
    barrier = threading.Barrier(2)

    @tool
    def echo(value: str) -> str:
        """Return the input text after both native calls enter."""
        barrier.wait(timeout=3)
        return value

    llm = NetworkStub(native=True, answers=[tool_calls("one", "two"), "Final Answer: completed"])
    crew, _, _ = make_crew(llm, tools=[echo], executor_class=executor)
    policy = Policy()
    CrewAIAdapter(crew, interceptors=[policy]).kickoff({"question": "hello"})
    calls = [ctx for ctx in policy.contexts if "tool_call" in ctx]
    assert len(calls) == 4
    pairs: dict[str, list[AgentContext]] = {}
    for ctx in calls:
        pairs.setdefault(ctx["tool_call"]["id"], []).append(ctx)
    assert len(pairs) == 2
    for pair in pairs.values():
        assert [ctx["interception_point"] for ctx in pair] == ["pre_tool_call", "post_tool_call"]
        assert pair[0]["tool_call"]["args"] == pair[1]["tool_call"]["args"]


def test_capped_audit_and_host_owned_envelope() -> None:
    crew, _, _ = make_crew()
    policy = Policy()
    adapter = CrewAIAdapter(crew, interceptors=[policy], max_records=2)
    adapter.kickoff({"question": "hello", "session": {"id": "spoof"}, "sequence": 500})
    assert len(adapter.records) == 2 and adapter.records_dropped == 4
    ctx = next(ctx for ctx in policy.contexts if ctx["interception_point"] == "input")
    assert ctx["session"]["id"] != "spoof"
    assert ctx["sequence"] != 500
    assert ctx["target"]["content"]["session"]["id"] == "spoof"
