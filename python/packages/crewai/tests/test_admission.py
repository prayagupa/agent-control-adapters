"""Unsupported host modes fail before any model, tool, or SDK callback effects."""

from __future__ import annotations

import pytest
from agent_control_crewai import AgentPolicy, CrewAIAdapter
from crewai import Process
from crewai_adapter_test_helpers import Policy, make_crew


@pytest.mark.parametrize(
    "scope, field",
    [
        ("crew", field)
        for field in (
            "stream",
            "checkpoint",
            "checkpoint_inputs",
            "checkpoint_train",
            "checkpoint_kickoff_event_id",
            "before_kickoff_callbacks",
            "after_kickoff_callbacks",
            "planning",
            "memory",
            "knowledge",
            "knowledge_sources",
            "step_callback",
            "task_callback",
            "manager_agent",
            "manager_llm",
        )
    ]
    + [
        ("agent", field)
        for field in (
            "allow_delegation",
            "step_callback",
            "memory",
            "knowledge",
            "knowledge_sources",
            "reasoning",
            "planning",
            "planning_config",
            "guardrail",
            "allow_code_execution",
            "max_execution_time",
            "apps",
            "mcps",
            "callbacks",
            "checkpoint",
            "checkpoint_kickoff_event_id",
            "skills",
        )
    ]
    + [
        ("task", field)
        for field in (
            "async_execution",
            "human_input",
            "guardrail",
            "guardrails",
            "callback",
            "checkpoint_original_description",
            "checkpoint_original_expected_output",
            "output_file",
        )
    ]
    + [("model", "stream")],
)
def test_unsupported_modes(scope: str, field: str) -> None:
    crew, agent, llm = make_crew()
    owner = {"crew": crew, "agent": agent, "task": crew.tasks[0], "model": llm}[scope]
    # Set the public option without triggering construction-time external effects.
    owner.__dict__[field] = True
    policy = Policy()
    adapter = CrewAIAdapter(crew, interceptors=[policy])
    with pytest.raises(ValueError, match="Unsupported"):
        adapter.kickoff({"question": "hello"})
    assert not policy.contexts and not llm.seen and adapter._worker is None


def test_hierarchical_rejected() -> None:
    crew, _, llm = make_crew()
    object.__setattr__(crew, "process", Process.hierarchical)
    with pytest.raises(ValueError, match="sequential"):
        CrewAIAdapter(crew, interceptors=[Policy()]).kickoff()
    assert not llm.seen


@pytest.mark.parametrize("kind", ["missing_shared", "empty", "duplicate", "foreign"])
def test_policy_admission(kind: str) -> None:
    crew, agent, llm = make_crew()
    _, foreign, _ = make_crew()
    policy = Policy()
    bindings = {
        "missing_shared": [],
        "empty": [AgentPolicy(agent, ())],
        "duplicate": [AgentPolicy(agent, (policy,)), AgentPolicy(agent, (policy,))],
        "foreign": [AgentPolicy(foreign, (policy,))],
    }
    adapter = CrewAIAdapter(
        crew,
        interceptors=[] if kind == "missing_shared" else [policy],
        agent_policies=bindings[kind],
    )
    with pytest.raises(ValueError):
        adapter.kickoff()
    assert not llm.seen and not policy.contexts


@pytest.mark.parametrize(
    "option, value",
    [
        ("timeout", 0),
        ("timeout", float("nan")),
        ("emit_timeout", float("inf")),
        ("max_records", 0),
        ("max_records", True),
        ("max_records", 100_001),
    ],
)
def test_invalid_resource_limits(option: str, value: object) -> None:
    crew, _, _ = make_crew()
    with pytest.raises(ValueError):
        CrewAIAdapter(crew, interceptors=[Policy()], **{option: value})
