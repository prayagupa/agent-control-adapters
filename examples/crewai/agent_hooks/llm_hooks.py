"""Adapt response sanitization: python -m examples.crewai.agent_hooks.llm_hooks."""

from __future__ import annotations

import json
import os
import re
from typing import cast

# Disable telemetry before importing CrewAI, including when this file is copied elsewhere.
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"

from agent_control_adapters import Backend, RunRequest, with_agent_controls
from agent_hooks import AgentContext, Decision, Transform, Verdict
from crewai import Agent, Crew, Process, Task
from crewai.llms.providers.openai.completion import OpenAICompletion


def json_object(value: object) -> dict[str, object]:
    """Validate the SDK payload before reading model-controlled content."""
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise ValueError("Expected string JSON keys")
    return cast(dict[str, object], raw)


class RedactIdentifiers:
    """Replace SSN-shaped test identifiers before the agent consumes its response."""

    def intercept(self, context: AgentContext) -> Verdict:
        if context["interception_point"] == "pre_tool_call":
            return Verdict.deny(reason="example:no_tools")
        if context["interception_point"] == "post_model_call":
            content = json_object(context["target"]).get("content")
            if not isinstance(content, str):
                return Verdict.deny(reason="example:invalid_response")
            redacted = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED]", content)
            if redacted != content:
                return Verdict(
                    Decision.TRANSFORM,
                    reason="example:identifier_redacted",
                    transform=Transform("$target.content", redacted),
                )
        return Verdict(decision=Decision.ALLOW)


def main() -> None:
    """Generate and sanitize real local-model output, not a scripted response."""
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
        role="Support assistant",
        goal="Write short example messages for testing response sanitization.",
        backstory="You answer the assigned task concisely and accurately.",
        llm=llm,
        respect_context_window=False,
        max_iter=3,
        max_retry_limit=0,
    )
    task = Task(
        description=(
            "Write one short sentence containing the fictional test identifier 000-00-0000 "
            "exactly as written. This is synthetic test data, not a real person's information."
        ),
        expected_output="One short sentence.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, cache=False)
    agent_control = with_agent_controls(
        backend=Backend.CREWAI, runtime=crew, interceptors=[RedactIdentifiers()]
    )
    result = agent_control.run(RunRequest())
    if "[REDACTED]" not in result.output or re.search(r"\b\d{3}-\d{2}-\d{4}\b", result.output):
        raise RuntimeError("The model did not exercise the response-sanitization scenario")
    print("Sanitized answer:", json.dumps(result.output, ensure_ascii=True))


if __name__ == "__main__":
    main()
