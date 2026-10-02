"""Request a bounded human decision on a genuinely generated draft (POSIX terminal)."""

from __future__ import annotations

import json
import os
import select
import sys

from agent_hooks import (
    AgentContext,
    ApprovalOutcome,
    ApprovalRequest,
    ApprovalResolution,
    Decision,
    Verdict,
)
from agent_hooks_crewai import AdapterBlocked, CrewAIAdapter
from crewai import Task

from examples.crewai.common import json_object, local_llm, make_agent, make_crew, report


class ReviewBeforeRelease:
    """Escalate final output; no resolver means the output remains denied."""

    def intercept(self, context: AgentContext) -> Verdict:
        if context["interception_point"] == "output":
            return Verdict.escalate(reason="example:human_review_required")
        return Verdict(decision=Decision.ALLOW)


class ConsoleApproval:
    """Approve only an explicit 'approve'; EOF, invalid input and timeout reject."""

    def __init__(self) -> None:
        self.decisions: list[str] = []
        self.input_mode = "interactive-terminal" if sys.stdin.isatty() else "piped-test-input"

    def resolve(self, request: ApprovalRequest) -> ApprovalResolution:
        content = json_object(request.context["target"]).get("content")
        if not isinstance(content, str):
            raise ValueError("Approval requires a textual draft")
        # Escape controls: untrusted model content must not become terminal commands.
        print(f"Draft (untrusted): {json.dumps(content, ensure_ascii=True)}", flush=True)
        print(
            "Release this draft? Type 'approve' within 15 seconds; anything else rejects.",
            flush=True,
        )
        ready, _, _ = select.select([sys.stdin], [], [], 15)
        response = os.read(sys.stdin.fileno(), 128) if ready else b""
        approved = response in {b"approve\n", b"approve\r\n"}
        self.decisions.append("approve" if approved else "reject")
        return ApprovalResolution(
            outcome=ApprovalOutcome.APPROVE if approved else ApprovalOutcome.REJECT,
            context_identity=request.context_identity,
            verdict=(
                Verdict(decision=Decision.ALLOW)
                if approved
                else Verdict.deny(reason="example:operator_rejected")
            ),
        )


def main() -> None:
    """Demonstrate SDK escalation, identity-bound approval, and safe rejection."""
    with local_llm() as (llm, run):
        agent = make_agent(llm, "Draft writer", "Write an announcement for review.")
        task = Task(
            description="Write one welcoming sentence about a community coding workshop.",
            expected_output="One short announcement.",
            agent=agent,
        )
        resolver = ConsoleApproval()
        adapter = CrewAIAdapter(
            make_crew([agent], [task]),
            interceptors=[ReviewBeforeRelease()],
            resolver=resolver,
            timeout=20,
            emit_timeout=30,
        )
        output: str | None = None
        status = "blocked"
        try:
            output = adapter.kickoff().raw
            status = "allowed"
        except AdapterBlocked:
            pass  # Expected rejection; the receipt below reports it explicitly.
        report(
            example="human_approval",
            status=status,
            output=output,
            run=run,
            records=adapter.records,
            details={"operator_decisions": resolver.decisions, "input_mode": resolver.input_mode},
        )


if __name__ == "__main__":
    main()
