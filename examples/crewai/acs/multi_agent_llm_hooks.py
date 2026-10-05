"""Use YAML/Rego LLM controls: python -m examples.crewai.acs.multi_agent_llm_hooks."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"

from agent_hooks_adapters import AgentPolicy, Backend, RunRequest, with_agent_controls
from crewai import Agent, Crew, Process, Task
from crewai.llms.providers.openai.completion import OpenAICompletion


def main() -> None:
    """Run researcher then editor and verify their real per-agent verdicts."""
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
    controls = Path(__file__).parent
    agent_control = with_agent_controls(
        backend=Backend.CREWAI,
        runtime=crew,
        policy_file=controls / "shared.yaml",
        agents={"researcher": researcher, "editor": editor},
        agent_policies=[
            AgentPolicy("researcher", policy_file=controls / "researcher.yaml"),
            AgentPolicy("editor", policy_file=controls / "editor.yaml"),
        ],
    )
    result = agent_control.run(RunRequest())
    sessions = {
        reason: {
            record.session_id
            for record in result.records
            if record.interception_point.value == "post_model_call"
            and any(verdict.reason == reason for verdict in record.verdicts)
        }
        for reason in ("example:researcher_text_limit", "example:editor_text_limit")
    }
    if any(len(identities) != 1 for identities in sessions.values()) or (
        len({session for identities in sessions.values() for session in identities}) != 2
    ):
        raise RuntimeError("Per-agent YAML policy routing was not isolated")
    print("Recommendation:", json.dumps(result.output, ensure_ascii=True))
    print("Per-agent YAML policies stayed isolated.")


if __name__ == "__main__":
    main()
