"""Let real local Qwen request a harmless demonstration tool; actual ACS/Rego denies it."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
from agent_hooks import Decision
from agent_hooks_adapters import AdapterBlocked, Backend, RunRequest, with_agent_controls
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from openai import AsyncOpenAI


async def main() -> None:
    """Require an actual model tool request, zero effects and the actual policy DENY record."""
    effects: list[str] = []
    model_requests = 0

    async def delete_file(path: str) -> dict[str, str]:
        """Demonstration only: record the request; no file operation is performed."""
        effects.append(path)
        return {"result": "No file was changed."}

    async def count_request(request: httpx.Request) -> None:
        nonlocal model_requests
        if request.method == "POST" and request.url.path == "/v1/chat/completions":
            model_requests += 1

    http = httpx.AsyncClient(
        timeout=httpx.Timeout(60, connect=3),
        trust_env=False,
        follow_redirects=False,
        event_hooks={"request": [count_request]},
    )
    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        http_client=http,
        max_retries=0,
        timeout=60,
    )
    runner: Runner | None = None
    try:
        model = LiteLlm(
            model="openai/qwen2.5:latest",
            client=client,
            api_base="http://localhost:11434/v1",
            api_key="ollama",
            num_retries=0,
            max_retries=0,
            timeout=60,
            temperature=0,
            max_tokens=128,
            tool_choice={"type": "function", "function": {"name": "delete_file"}},
        )
        agent = LlmAgent(
            name="local_qwen",
            mode="chat",
            model=model,
            tools=[delete_file],
            instruction="Call the provided demonstration tool when asked. Do not claim success.",
        )
        runner = Runner(
            agent=agent,
            app_name="caller_app",
            session_service=InMemorySessionService(),
        )
        adapter = with_agent_controls(
            backend=Backend.GOOGLE_ADK,
            runtime=runner,
            interceptors=None,
            policy_file=Path(__file__).with_name("tool_hooks.yaml"),
            agents=None,
            agent_policies=(),
            resolver=None,
            timeout=5,
            emit_timeout=30,
            max_records=128,
        )
        try:
            await adapter.arun(
                RunRequest(
                    {
                        "prompt": (
                            "Call delete_file with path='demo.txt'. "
                            "This is a harmless demonstration."
                        ),
                    }
                )
            )
        except AdapterBlocked:
            if (
                model_requests < 1
                or effects
                or not any(
                    record.interception_point.value == "pre_tool_call"
                    and record.verdict.decision is Decision.DENY
                    and record.verdict.reason == "example:destructive_tool"
                    for record in adapter.records
                )
            ):
                raise RuntimeError(
                    "The real model did not exercise the ACS tool-denial policy"
                ) from None
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "model": "qwen2.5:latest",
                        "model_requests": model_requests,
                        "tool_effects": len(effects),
                        "records_dropped": adapter.records_dropped,
                    }
                )
            )
        else:
            raise RuntimeError("The real model did not request the policy-blocked tool")
        for record in adapter.records:
            print(
                json.dumps(
                    {
                        "audit": True,
                        "point": record.interception_point.value,
                        "decision": record.verdict.decision.value,
                        "sequence": record.sequence,
                        "session_id": record.session_id,
                    }
                )
            )
    finally:
        if runner is not None:
            await runner.close()
        await client.close()
        await http.aclose()


if __name__ == "__main__":
    asyncio.run(main())
