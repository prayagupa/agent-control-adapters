"""Adapt CrewAI's agent-scoped model hooks: python -m examples.crewai.multi_agent."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import cast

# Disable telemetry before importing CrewAI, including when this file is copied elsewhere.
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"

from agent_hooks import AgentContext, Decision, Verdict
from agent_hooks_adapters import AgentPolicy, Backend, RunRequest, with_agent_hooks
from crewai import Agent, Crew, Process, Task
from crewai.llms.providers.openai.completion import OpenAICompletion


def json_object(value: object) -> dict[str, object]:
    """Validate an external JSON object before reading it."""
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise ValueError("Expected string JSON keys")
    return cast(dict[str, object], raw)


class NoTools:
    """These text-only agents must never invoke tools."""

    def intercept(self, context: AgentContext) -> Verdict:
        if context["interception_point"] == "pre_tool_call":
            return Verdict.deny(reason="example:no_tools")
        return Verdict(decision=Decision.ALLOW)


@dataclass
class AgentTextLimit:
    """Limit an agent's response length and retain the host identities it checked."""

    max_characters: int
    seen_agent_ids: set[str] = field(default_factory=set[str])

    def intercept(self, context: AgentContext) -> Verdict:
        envelope = json_object(context["agent"])
        identifier = envelope.get("id")
        if not isinstance(identifier, str):
            return Verdict.deny(reason="example:missing_agent")
        self.seen_agent_ids.add(identifier)
        if context["interception_point"] == "post_model_call":
            response = json_object(context["target"])
            content = response.get("content")
            if not isinstance(content, str) or len(content) > self.max_characters:
                return Verdict.deny(reason="example:text_limit")
        return Verdict(decision=Decision.ALLOW)


def main() -> None:
    """Run researcher then editor and verify their policies stay isolated."""
    model = os.getenv("LOCAL_MODEL", "qwen2.5:latest")
    if ":" not in model:
        model += ":latest"
    llm = OpenAICompletion(
        model=model,
        custom_openai=True,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0,
        max_tokens=256,
        timeout=60,
        max_retries=0,
    )
    researcher = Agent(
        role="Researcher",
        goal="Find concise, useful software facts.",
        backstory="You answer the assigned task concisely and accurately.",
        llm=llm,
        respect_context_window=False,
        max_iter=3,
        max_retry_limit=0,
    )
    editor = Agent(
        role="Editor",
        goal="Turn research into a short recommendation.",
        backstory="You answer the assigned task concisely and accurately.",
        llm=llm,
        respect_context_window=False,
        max_iter=3,
        max_retry_limit=0,
    )
    research = Task(
        description="Give two short reasons to review code before merging it.",
        expected_output="Two short bullet points.",
        agent=researcher,
    )
    editing = Task(
        description="Use the supplied research to write one short recommendation.",
        expected_output="One concise sentence.",
        context=[research],
        agent=editor,
    )
    crew = Crew(
        agents=[researcher, editor],
        tasks=[research, editing],
        process=Process.sequential,
        cache=False,
    )
    researcher_policy = AgentTextLimit(max_characters=1600)
    editor_policy = AgentTextLimit(max_characters=800)
    adapter = with_agent_hooks(
        backend=Backend.CREWAI,
        runtime=crew,
        interceptors=[NoTools()],
        agents={"researcher": researcher, "editor": editor},
        agent_policies=[
            AgentPolicy("researcher", (researcher_policy,)),
            AgentPolicy("editor", (editor_policy,)),
        ],
    )
    result = adapter.run(RunRequest())
    if researcher_policy.seen_agent_ids != {str(researcher.id)} or (
        editor_policy.seen_agent_ids != {str(editor.id)}
    ):
        raise RuntimeError("Per-agent policy routing was not isolated")
    print("Recommendation:", json.dumps(result.output, ensure_ascii=True))
    print("Per-agent policies stayed isolated.")


if __name__ == "__main__":
    main()
