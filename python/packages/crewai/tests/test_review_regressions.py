"""Real-runner regressions for uncovered fallback calls and task-owned tools."""

from __future__ import annotations

from pathlib import Path

import pytest
from agent_control_crewai import AdapterBlocked, CrewAIAdapter
from agent_hooks import ALLOW, Decision, Transform, Verdict
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.experimental.agent_executor import AgentExecutor
from crewai.hooks.dispatch import InterceptionPoint, scoped_hooks
from crewai.hooks.llm_hooks import LLMCallHookContext
from crewai.tools import tool
from crewai_adapter_test_helpers import NetworkStub, Policy, make_crew, tool_calls
from pydantic import BaseModel

EXECUTORS = [AgentExecutor, CrewAgentExecutor]


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("limit", [1, 2, 3])
def test_nonfinal_responses_cannot_reach_unhooked_forced_final(
    executor: type[AgentExecutor] | type[CrewAgentExecutor], limit: int
) -> None:
    effects: list[str] = []

    @tool
    def echo(value: str) -> str:
        """Record a valid nonfinal ReAct action."""
        effects.append(value)
        return value

    nonfinal = [
        'Thought: continue\nAction: echo\nAction Input: {"value":"again-' + str(index) + '"}'
        for index in range(limit)
    ]
    llm = NetworkStub(answers=[*nonfinal, "Final Answer: restricted"])
    crew, agent, _ = make_crew(llm, tools=[echo], cache=False, executor_class=executor)
    agent.max_iter = limit
    agent.respect_context_window = False
    counters: list[int] = []

    def observe(context: LLMCallHookContext) -> None:
        counters.append(context.iterations)

    policy = Policy(
        lambda ctx: (
            Verdict.deny(reason="test:restricted")
            if ctx["interception_point"] == "post_model_call"
            and "restricted" in ctx["target"]["content"]
            else ALLOW
        )
    )
    adapter = CrewAIAdapter(crew, interceptors=[policy])
    with (
        scoped_hooks({InterceptionPoint.PRE_MODEL_CALL: [observe]}),
        pytest.raises(AdapterBlocked),
    ):
        adapter.kickoff({"question": "hello"})
    assert counters == list(range(limit))
    assert len(llm.seen) == limit - 1
    assert len(effects) == limit - 1
    assert (
        sum(ctx["interception_point"] == "pre_model_call" for ctx in policy.contexts) == limit - 1
    )
    assert (
        sum(ctx["interception_point"] == "post_model_call" for ctx in policy.contexts) == limit - 1
    )


@pytest.mark.parametrize("executor", EXECUTORS)
def test_unadapted_runner_demonstrates_forced_final_hook_gap(
    executor: type[AgentExecutor] | type[CrewAgentExecutor],
) -> None:
    @tool
    def echo(value: str) -> str:
        """Enable the runner's strict ReAct parser."""
        return value

    nonfinal = 'Thought: continue\nAction: echo\nAction Input: {"value":"again"}'
    llm = NetworkStub(answers=[nonfinal, "Final Answer: restricted"])
    crew, agent, _ = make_crew(llm, tools=[echo], executor_class=executor)
    agent.max_iter = 1
    counters: list[int] = []
    responses: list[str | None] = []

    def before(context: LLMCallHookContext) -> None:
        counters.append(context.iterations)

    def after(context: LLMCallHookContext) -> None:
        responses.append(context.response)

    with scoped_hooks(
        {
            InterceptionPoint.PRE_MODEL_CALL: [before],
            InterceptionPoint.POST_MODEL_CALL: [after],
        }
    ):
        result = crew.kickoff(inputs={"question": "hello"})
    assert result.raw == "restricted"
    assert len(llm.seen) == 2
    assert counters == [0] and responses == [nonfinal]


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("limit", [1, 2, 3])
def test_literal_malformed_text_is_finalized_but_still_governed(
    executor: type[AgentExecutor] | type[CrewAgentExecutor], limit: int
) -> None:
    llm = NetworkStub(answers=["not a final answer", "Final Answer: restricted"])
    crew, agent, _ = make_crew(llm, executor_class=executor)
    agent.max_iter = limit
    policy = Policy()
    adapter = CrewAIAdapter(crew, interceptors=[policy])
    if limit == 1:
        with pytest.raises(AdapterBlocked):
            adapter.kickoff({"question": "hello"})
        assert not llm.seen
        return
    output = adapter.kickoff({"question": "hello"})
    assert output.raw == "not a final answer"
    assert len(llm.seen) == 1
    responses = [ctx for ctx in policy.contexts if ctx["interception_point"] == "post_model_call"]
    assert len(responses) == 1 and responses[0]["target"]["content"] == output.raw


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("limit", [1, 2, 3])
def test_repeated_native_tools_cannot_reach_unhooked_forced_final(
    executor: type[AgentExecutor] | type[CrewAgentExecutor], limit: int
) -> None:
    effects: list[str] = []

    @tool
    def echo(value: str) -> str:
        """Record the actual tool argument."""
        effects.append(value)
        return value

    llm = NetworkStub(
        native=True, answers=[tool_calls("again")] * limit + ["Final Answer: restricted"]
    )
    crew, agent, _ = make_crew(llm, tools=[echo], cache=False, executor_class=executor)
    agent.max_iter = limit
    agent.respect_context_window = False
    counters: list[int] = []

    def observe(context: LLMCallHookContext) -> None:
        counters.append(context.iterations)

    adapter = CrewAIAdapter(crew, interceptors=[Policy()])
    with (
        scoped_hooks({InterceptionPoint.PRE_MODEL_CALL: [observe]}),
        pytest.raises(AdapterBlocked),
    ):
        adapter.kickoff({"question": "hello"})
    assert counters == list(range(limit))
    assert len(llm.seen) == len(effects) == limit - 1


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("empty", [False, True])
def test_native_argument_transform_never_executes_unauthorized_defaults(
    executor: type[AgentExecutor] | type[CrewAgentExecutor], empty: bool
) -> None:
    received: list[str] = []

    @tool
    def echo(value: str = "unauthorized default") -> str:
        """Capture the real argument received by the native tool."""
        received.append(value)
        return value

    call = [
        {
            "id": "call-default",
            "type": "function",
            "function": {"name": "echo", "arguments": {} if empty else {"value": "original"}},
        }
    ]
    llm = NetworkStub(native=True, answers=[call, "Final Answer: completed"])
    crew, agent, _ = make_crew(llm, tools=[echo], cache=False, executor_class=executor)
    agent.respect_context_window = False
    policy = Policy(
        lambda ctx: (
            Verdict(Decision.TRANSFORM, transform=Transform("$target", {"value": "authorized"}))
            if ctx["interception_point"] == "pre_tool_call"
            else ALLOW
        )
    )
    adapter = CrewAIAdapter(crew, interceptors=[policy])
    if empty and executor is CrewAgentExecutor:
        with pytest.raises(AdapterBlocked):
            adapter.kickoff({"question": "hello"})
        assert received == []
    else:
        assert adapter.kickoff({"question": "hello"}).raw == "Final Answer: completed"
        assert received == ["authorized"]


@pytest.mark.parametrize("executor", EXECUTORS)
def test_task_owned_tool_is_visible_to_startup_denial(
    executor: type[AgentExecutor] | type[CrewAgentExecutor],
) -> None:
    effects: list[str] = []

    @tool
    def probe() -> str:
        """Record whether this task-only tool was invoked."""
        effects.append("executed")
        return "done"

    call = [{"id": "task-tool", "type": "function", "function": {"name": "probe", "arguments": {}}}]
    llm = NetworkStub(native=True, answers=[call, "Final Answer: completed"])
    crew, agent, _ = make_crew(llm, executor_class=executor)
    agent.respect_context_window = False
    assert not agent.tools
    crew.tasks[0].tools = [probe]
    policy = Policy(
        lambda ctx: (
            Verdict.deny(reason="test:task_tool")
            if ctx["interception_point"] == "agent_startup"
            and "probe" in ctx["target"]["tools_registered"]
            else ALLOW
        )
    )
    with pytest.raises(AdapterBlocked):
        CrewAIAdapter(crew, interceptors=[policy]).kickoff({"question": "hello"})
    assert not llm.seen and not effects


class StructuredResult(BaseModel):
    text: str


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("field", ["output_json", "output_pydantic", "response_model"])
def test_structured_repair_rejected_before_model_or_policy(
    executor: type[AgentExecutor] | type[CrewAgentExecutor], field: str
) -> None:
    llm = NetworkStub(answers=["Final Answer: invalid json", '{"text":"restricted"}'])
    crew, agent, _ = make_crew(llm, executor_class=executor)
    agent.respect_context_window = False
    setattr(crew.tasks[0], field, StructuredResult)
    policy = Policy()
    with pytest.raises(ValueError, match=field):
        CrewAIAdapter(crew, interceptors=[policy]).kickoff({"question": "hello"})
    assert not llm.seen and not policy.contexts


@pytest.mark.parametrize(
    "scope, field",
    [
        ("agent", "respect_context_window"),
        ("agent", "function_calling_llm"),
        ("crew", "function_calling_llm"),
    ],
)
def test_uncovered_summarization_and_tool_repair_rejected(scope: str, field: str) -> None:
    crew, agent, llm = make_crew()
    agent.respect_context_window = False
    owner = agent if scope == "agent" else crew
    setattr(owner, field, True if field == "respect_context_window" else NetworkStub())
    policy = Policy()
    with pytest.raises(ValueError, match=field):
        CrewAIAdapter(crew, interceptors=[policy]).kickoff({"question": "hello"})
    assert not llm.seen and not policy.contexts


@pytest.mark.parametrize("skill", [Path("/nonexistent/adapter-test-skill"), "registry:test/skill"])
def test_crew_skills_rejected_before_public_loader(
    skill: Path | str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import crewai.crews.utils

    crew, agent, llm = make_crew()
    agent.respect_context_window = False
    crew.skills = [skill]
    loaded: list[object] = []

    def forbidden_loader(*args: object, **kwargs: object) -> None:
        loaded.append(args)
        raise AssertionError("Skill loader must not run")

    monkeypatch.setattr(crewai.crews.utils, "load_skills", forbidden_loader)
    policy = Policy()
    with pytest.raises(ValueError, match="skills"):
        CrewAIAdapter(crew, interceptors=[policy]).kickoff({"question": "hello"})
    assert not loaded and not llm.seen and not policy.contexts


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("field", ["respect_context_window", "function_calling_llm"])
def test_retained_executor_forbidden_settings_rejected_before_effects(
    executor: type[AgentExecutor] | type[CrewAgentExecutor], field: str
) -> None:
    effects: list[str] = []

    @tool
    def echo(value: str) -> str:
        """Expose any execution that escaped admission."""
        effects.append(value)
        return value

    llm = NetworkStub(native=True, answers=[tool_calls("probe"), "Final Answer: completed"])
    repair_llm = NetworkStub()
    crew, agent, _ = make_crew(llm, tools=[echo], cache=False, executor_class=executor)
    agent.crew = crew
    assert agent.agent_executor is None
    if field == "respect_context_window":
        agent.respect_context_window = True
    else:
        agent.function_calling_llm = repair_llm
        crew.function_calling_llm = repair_llm
    agent.create_agent_executor(task=crew.tasks[0])
    retained = agent.agent_executor
    assert type(retained) is executor
    agent.respect_context_window = False
    agent.function_calling_llm = None
    crew.function_calling_llm = None
    assert retained is not None
    assert getattr(retained, field) is (True if field == "respect_context_window" else repair_llm)

    policy = Policy()
    adapter = CrewAIAdapter(crew, interceptors=[policy])
    with pytest.raises(ValueError, match=field):
        adapter.kickoff({"question": "hello"})
    assert agent.agent_executor is retained
    assert not llm.seen and not repair_llm.seen and not effects and not policy.contexts
    assert adapter._worker is None and not adapter.records


@pytest.mark.parametrize("executor", EXECUTORS)
def test_valid_retained_executor_remains_supported(
    executor: type[AgentExecutor] | type[CrewAgentExecutor],
) -> None:
    crew, agent, llm = make_crew(executor_class=executor)
    agent.crew = crew
    assert agent.agent_executor is None
    agent.create_agent_executor(task=crew.tasks[0])
    retained = agent.agent_executor
    assert type(retained) is executor
    assert retained is not None
    assert retained.respect_context_window is False and retained.function_calling_llm is None
    policy = Policy()
    output = CrewAIAdapter(crew, interceptors=[policy]).kickoff({"question": "hello"})
    assert output.raw == "completed"
    assert agent.agent_executor is retained
    assert len(llm.seen) == 1
    assert sum(ctx["interception_point"] == "pre_model_call" for ctx in policy.contexts) == 1
    assert sum(ctx["interception_point"] == "post_model_call" for ctx in policy.contexts) == 1
