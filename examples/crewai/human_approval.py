"""Adapt CrewAI's tool-approval hook: python -m examples.crewai.human_approval."""

from __future__ import annotations

import json
import os
import select
import sys
from typing import cast

# Disable telemetry before importing CrewAI, including when this file is copied elsewhere.
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"

from agent_hooks import (
    AgentContext,
    ApprovalOutcome,
    ApprovalRequest,
    ApprovalResolution,
    Decision,
    Verdict,
)
from agent_hooks_adapters import AdapterBlocked, Backend, RunRequest, with_agent_hooks
from crewai import Agent, Crew, Process, Task
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.tools import tool


def json_object(value: object) -> dict[str, object]:
    """Validate an external JSON object before reading it."""
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise ValueError("Expected string JSON keys")
    return cast(dict[str, object], raw)


class ReviewBeforeSend:
    """Require approval for send_email, and reject any unexpected tool."""

    def intercept(self, context: AgentContext) -> Verdict:
        if context["interception_point"] == "pre_tool_call":
            if json_object(context.get("tool_call")).get("name") == "send_email":
                return Verdict.escalate(reason="example:tool_approval_required")
            return Verdict.deny(reason="example:unexpected_tool")
        return Verdict(decision=Decision.ALLOW)


class ConsoleApproval:
    """Approve only 'approve'; invalid input, EOF and the 15-second timeout reject."""

    def resolve(self, request: ApprovalRequest) -> ApprovalResolution:
        message = json_object(request.context["target"]).get("message")
        if json_object(request.context.get("tool_call")).get(
            "name"
        ) != "send_email" or not isinstance(message, str):
            raise ValueError("Approval requires a send_email request with a textual message")
        print(f"Tool message (untrusted): {json.dumps(message, ensure_ascii=True)}", flush=True)
        print(
            "Approve this preview? Type 'approve' within 15 seconds; anything else rejects.",
            flush=True,
        )
        ready, _, _ = select.select([sys.stdin], [], [], 15)
        response = os.read(sys.stdin.fileno(), 128) if ready else b""
        approved = response in {b"approve\n", b"approve\r\n"}
        return ApprovalResolution(
            outcome=ApprovalOutcome.APPROVE if approved else ApprovalOutcome.REJECT,
            context_identity=request.context_identity,
            verdict=(
                Verdict(decision=Decision.ALLOW)
                if approved
                else Verdict.deny(reason="example:operator_rejected")
            ),
        )


@tool
def send_email(message: str) -> str:
    """Preview a message after approval; this demonstration never sends email."""
    print("Approved preview (not sent):", json.dumps(message, ensure_ascii=True))
    return message


def main() -> None:
    """Approve or reject a real model's tool request before running its preview."""
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
        role="Communications assistant",
        goal="Prepare a welcoming announcement for review.",
        backstory="Use the provided preview tool exactly once before giving your final answer.",
        llm=llm,
        tools=[send_email],
        cache=False,
        respect_context_window=False,
        max_iter=3,
        max_retry_limit=0,
    )
    task = Task(
        description=(
            "Write one welcoming sentence about a community coding workshop and call "
            "send_email with that message to preview it. No email is sent. "
            "Then give a short confirmation of the approved preview."
        ),
        expected_output="One concise confirmation after the preview tool has run.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, cache=False)
    adapter = with_agent_hooks(
        backend=Backend.CREWAI,
        runtime=crew,
        interceptors=[ReviewBeforeSend()],
        resolver=ConsoleApproval(),
        timeout=20,
        emit_timeout=30,
    )
    try:
        result = adapter.run(RunRequest())
    except AdapterBlocked:
        if not any(record.resolved_by == "rejection" for record in adapter.records):
            raise RuntimeError("The model did not exercise the tool-approval rejection") from None
        print("Blocked: send_email did not execute.")
    else:
        if not any(
            record.interception_point.value == "post_tool_call" for record in result.records
        ):
            raise RuntimeError("The model did not request the approval-gated tool")
        print("Answer after approval:", json.dumps(result.output, ensure_ascii=True))


if __name__ == "__main__":
    main()
