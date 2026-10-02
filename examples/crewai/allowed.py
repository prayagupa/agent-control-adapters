"""Allow a harmless, real local-model task: python -m examples.crewai.allowed."""

from __future__ import annotations

from agent_hooks_crewai import CrewAIAdapter
from crewai import Task

from examples.crewai.common import NoTools, local_llm, make_agent, make_crew, report


def main() -> None:
    """Run a local agent and return its original, governed answer."""
    with local_llm() as (llm, run):
        agent = make_agent(llm, "Explainer", "Explain technical ideas simply.")
        task = Task(
            description="Explain in one short sentence why unit tests are useful.",
            expected_output="One short sentence.",
            agent=agent,
        )
        adapter = CrewAIAdapter(make_crew([agent], [task]), interceptors=[NoTools()])
        result = adapter.kickoff()
        report(
            example="allowed",
            status="allowed",
            output=result.raw,
            run=run,
            records=adapter.records,
        )


if __name__ == "__main__":
    main()
