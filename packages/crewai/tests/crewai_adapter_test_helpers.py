"""Deterministic tests replace only the model network boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_hooks import ALLOW, AgentContext, Verdict
from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from pydantic import Field


class NetworkStub(BaseLLM):
    """A deterministic replacement for model I/O, not for CrewAI or SDK execution."""

    model: str = "test/network-boundary"
    answers: list[Any] = Field(default_factory=lambda: ["Final Answer: completed"])
    seen: list[Any] = Field(default_factory=list)
    native: bool = False
    on_call: Callable[[], None] | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model="test/network-boundary", **kwargs)

    def call(self, messages: Any, **kwargs: Any) -> Any:
        self.seen.append(messages)
        if self.on_call is not None:
            self.on_call()
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def supports_function_calling(self) -> bool:
        return self.native


def make_crew(
    llm: NetworkStub | None = None, **agent_options: Any
) -> tuple[Crew, Agent, NetworkStub]:
    model = llm if llm is not None else NetworkStub()
    agent = Agent(
        role="same-role",
        goal="Complete the task",
        backstory="Test agent",
        llm=model,
        max_iter=3,
        max_retry_limit=2,
        respect_context_window=False,
        **agent_options,
    )
    task = Task(description="Answer {question}", expected_output="A short answer", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], cache=False, tracing=False, verbose=False)
    return crew, agent, model


class Policy:
    """Capture SDK contexts and optionally decide at the actual policy boundary."""

    def __init__(self, callback: Callable[[AgentContext], Verdict] | None = None) -> None:
        self.callback = callback
        self.contexts: list[AgentContext] = []

    def intercept(self, context: AgentContext) -> Verdict:
        self.contexts.append(context)
        return self.callback(context) if self.callback is not None else ALLOW


def denial(point: str) -> Policy:
    """Deny exactly one SDK interception point."""
    return Policy(
        lambda context: (
            Verdict.deny(reason="test:blocked") if context["interception_point"] == point else ALLOW
        )
    )


def tool_calls(*values: str) -> list[dict[str, object]]:
    """Supply native model tool-call responses for deterministic integration tests."""
    return [
        {
            "id": f"call-{index}",
            "type": "function",
            "function": {"name": "echo", "arguments": {"value": value}},
        }
        for index, value in enumerate(values)
    ]
