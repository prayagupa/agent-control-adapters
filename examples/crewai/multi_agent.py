"""Two real agents with separate policy instances bound to their host identities."""

from __future__ import annotations

from agent_hooks_crewai import AgentPolicy, CrewAIAdapter
from crewai import Task

from examples.crewai.common import (
    AgentTextLimit,
    NoTools,
    local_llm,
    make_agent,
    make_crew,
    report,
)


def main() -> None:
    """Run researcher then editor; prove each interceptor saw only its own agent."""
    with local_llm() as (llm, run):
        researcher = make_agent(llm, "Researcher", "Find concise, useful software facts.")
        editor = make_agent(llm, "Editor", "Turn research into a short recommendation.")
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
        researcher_policy = AgentTextLimit(max_characters=1600)
        editor_policy = AgentTextLimit(max_characters=800)
        adapter = CrewAIAdapter(
            make_crew([researcher, editor], [research, editing]),
            interceptors=[NoTools()],
            agent_policies=[
                AgentPolicy(researcher, (researcher_policy,)),
                AgentPolicy(editor, (editor_policy,)),
            ],
        )
        output = adapter.kickoff()
        expected_researcher = {str(researcher.id)}
        expected_editor = {str(editor.id)}
        if (
            researcher_policy.seen_agent_ids != expected_researcher
            or editor_policy.seen_agent_ids != expected_editor
        ):
            raise RuntimeError("Per-agent policy routing was not isolated")
        report(
            example="multi_agent",
            status="allowed",
            output=output.raw,
            run=run,
            records=adapter.records,
            details={
                "researcher_agent_id": str(researcher.id),
                "editor_agent_id": str(editor.id),
                "researcher_policy_agents": sorted(researcher_policy.seen_agent_ids),
                "editor_policy_agents": sorted(editor_policy.seen_agent_ids),
                "task_outputs": [task.raw for task in output.tasks_output],
            },
        )


if __name__ == "__main__":
    main()
