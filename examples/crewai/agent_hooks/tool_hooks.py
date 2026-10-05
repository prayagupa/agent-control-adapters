"""Adapt tool safety: python -m examples.crewai.agent_hooks.tool_hooks."""

from __future__ import annotations

import os
from typing import cast

# Disable telemetry before importing CrewAI, including when this file is copied elsewhere.
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"

from agent_hooks import AgentContext, Decision, Verdict
from agent_hooks_adapters import AdapterBlocked, Backend, RunRequest, with_agent_controls
from crewai import Agent, Crew, Process, Task
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.tools import tool


def json_object(value: object) -> dict[str, object]:
    """Validate trusted tool metadata before selecting a policy decision."""
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise ValueError("Expected string JSON keys")
    return cast(dict[str, object], raw)


class BlockDestructiveTools:
    """Reject destructive tools before execution, using the host-owned tool name."""

    def intercept(self, context: AgentContext) -> Verdict:
        if context["interception_point"] == "pre_tool_call":
            name = json_object(context.get("tool_call")).get("name")
            if not isinstance(name, str) or name in {
                "delete_file",
                "drop_table",
                "system_shutdown",
            }:
                return Verdict.deny(reason="example:destructive_tool")
        return Verdict(decision=Decision.ALLOW)


@tool
def delete_file(path: str) -> str:
    """Demonstration-only deletion request; never read or delete any file."""
    raise RuntimeError("The denied demonstration tool must never execute")


def main() -> None:
    """Let the real model request a tool, then deny it before its body executes."""
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
    agent = Agent(
        role="Maintenance assistant",
        goal="Use the provided tool to perform the requested demonstration.",
        backstory="You must call a provided tool rather than claim an action without using it.",
        llm=llm,
        tools=[delete_file],
        cache=False,
        respect_context_window=False,
        max_iter=3,
        max_retry_limit=0,
    )
    task = Task(
        description="Call delete_file with path='demo.txt'. This is a harmless tool-safety test.",
        expected_output="Confirmation of the tool result.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, cache=False)
    agent_control = with_agent_controls(
        backend=Backend.CREWAI, runtime=crew, interceptors=[BlockDestructiveTools()]
    )
    try:
        agent_control.run(RunRequest())
    except AdapterBlocked:
        if not any(
            record.verdict.reason == "example:destructive_tool" for record in agent_control.records
        ):
            raise RuntimeError("The model did not exercise the destructive-tool policy") from None
        print("Blocked: delete_file did not execute.")
    else:
        raise RuntimeError("The model did not request the policy-blocked tool")


if __name__ == "__main__":
    main()
