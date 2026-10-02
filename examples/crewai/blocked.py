"""Block release of an actual model response, without printing the blocked content."""

from __future__ import annotations

from agent_hooks import AgentContext, Decision, Verdict
from agent_hooks_crewai import AdapterBlocked, CrewAIAdapter
from crewai import Task

from examples.crewai.common import local_llm, make_agent, make_crew, report


class DraftsStayInternal:
    """This application requires all generated drafts to stay unreleased."""

    def intercept(self, context: AgentContext) -> Verdict:
        if context["interception_point"] == "post_model_call":
            return Verdict.deny(reason="example:draft_not_releasable")
        return Verdict(decision=Decision.ALLOW)


def main() -> None:
    """Generate locally, then deny at the response boundary before normal output."""
    with local_llm() as (llm, run):
        agent = make_agent(llm, "Draft writer", "Write a brief draft announcement.")
        task = Task(
            description="Draft one sentence announcing a new community library.",
            expected_output="One sentence.",
            agent=agent,
        )
        adapter = CrewAIAdapter(make_crew([agent], [task]), interceptors=[DraftsStayInternal()])
        try:
            adapter.kickoff()
        except AdapterBlocked:
            report(
                example="blocked",
                status="blocked",
                output=None,
                run=run,
                records=adapter.records,
            )
        else:
            raise RuntimeError("The draft-release policy did not block the run")


if __name__ == "__main__":
    main()
