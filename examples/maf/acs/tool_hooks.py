"""Let real local Qwen request a harmless demonstration tool; ACS denies it."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx2
from agent_control_adapters import AdapterBlocked, Backend, RunRequest, with_agent_controls
from agent_framework import Agent, ChatOptions, tool
from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI


async def main() -> None:
    """Require a real model tool request, deny before effects, and print actual audit."""
    effects: list[str] = []

    @tool
    async def delete_file(path: str) -> str:
        """Demonstration only: record the request, never read or delete a file."""
        effects.append(path)
        return "No file was changed."

    model = os.getenv("LOCAL_MODEL", "qwen2.5:latest")
    if ":" not in model:
        model += ":latest"
    model_requests = 0

    async def count_model_request(request: httpx2.Request) -> None:
        nonlocal model_requests
        if request.method == "POST" and request.url.path in {
            "/v1/chat/completions",
            "/v1/responses",
        }:
            model_requests += 1

    http = httpx2.AsyncClient(
        timeout=httpx2.Timeout(60, connect=3),
        trust_env=False,
        follow_redirects=False,
        event_hooks={"request": [count_model_request]},
    )
    transport: AsyncOpenAI | None = None
    try:
        transport = AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            http_client=http,
            max_retries=0,
        )
        client = OpenAIChatClient(
            model=model,
            async_client=transport,
            function_invocation_configuration={"allow_concurrent_invocation": False},
        )
        agent = Agent[ChatOptions[None]](
            client,
            instructions=(
                "Use the provided demonstration tool when the user asks. Do not claim success."
            ),
            tools=[delete_file],
            default_options={
                "temperature": 0,
                "seed": 7,
                "max_tokens": 128,
                "tool_choice": "required",
            },
        )
        adapter = with_agent_controls(
            backend=Backend.MAF,
            runtime=agent,
            policy_file=Path(__file__).with_name("tool_hooks.yaml"),
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
            if effects or not any(
                record.interception_point.value == "pre_tool_call"
                and record.verdict.reason == "example:destructive_tool"
                for record in adapter.records
            ):
                raise RuntimeError(
                    "The real model did not exercise the tool-denial policy"
                ) from None
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "message": "Tool request blocked by governance.",
                        "tool_effects": len(effects),
                        "model": model,
                        "model_requests": model_requests,
                    }
                )
            )
        else:
            raise RuntimeError("The model did not request the policy-blocked tool")
        finally:
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
            print(json.dumps({"records_dropped": adapter.records_dropped}))
    finally:
        if transport is not None:
            await transport.close()
        await http.aclose()


if __name__ == "__main__":
    asyncio.run(main())
