"""Run a real local Qwen text request with native MAF and YAML/Rego controls."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx2
from agent_control_adapters import AdapterBlocked, Backend, RunRequest, with_agent_controls
from agent_framework import Agent, ChatOptions
from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI


async def main() -> None:
    """Own the local transport; run one ACS-controlled request on this loop."""
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
            instructions="Answer the user's question briefly in plain text.",
            default_options={"temperature": 0, "seed": 7, "max_tokens": 128},
        )
        adapter = with_agent_controls(
            backend=Backend.MAF,
            runtime=agent,
            policy_file=Path(__file__).with_name("llm_hooks.yaml"),
        )
        try:
            result = await adapter.arun(
                RunRequest(
                    {
                        "prompt": (
                            "In one sentence, what is the purpose of a safety policy "
                            "for an AI agent?"
                        ),
                    }
                )
            )
        except AdapterBlocked:
            print(json.dumps({"status": "blocked", "message": "Request blocked by governance."}))
            raise
        else:
            print(
                json.dumps(
                    {
                        "status": "allowed",
                        "output": result.output,
                        "model": model,
                        "model_requests": model_requests,
                    }
                )
            )
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
