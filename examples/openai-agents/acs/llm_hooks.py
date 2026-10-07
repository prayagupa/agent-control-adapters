"""Complete real ACS/Rego LLM controls example against an operator-managed local Qwen."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx2
from agent_hooks_adapters import AdapterBlocked, Backend, RunRequest, with_agent_controls
from agent_hooks_openai_agents import OpenAIAgentsAdapter
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel
from openai import AsyncOpenAI


async def main() -> int:
    """Show native setup, actual ACS manifest loading, model output and owned cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt", nargs="?", default="Explain one practical benefit of validating tool inputs."
    )
    prompt = parser.parse_args().prompt
    http = httpx2.AsyncClient(trust_env=False, follow_redirects=False)
    transport = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        http_client=http,
        max_retries=0,
        timeout=60,
    )
    try:
        model = OpenAIChatCompletionsModel(model="qwen2.5:latest", openai_client=transport)
        agent = Agent[object](
            name="local-qwen",
            model=model,
            instructions="Give a concise plain-text explanation. Do not use external tools.",
            model_settings=ModelSettings(
                temperature=0,
                max_tokens=64,
                parallel_tool_calls=False,
            ),
        )
        adapter = with_agent_controls(
            backend=Backend("openai-agents"),
            runtime=agent,
            policy_file=Path(__file__).with_name("controls.yaml"),
            timeout=5,
            emit_timeout=30,
            max_records=256,
        )
        if not isinstance(adapter, OpenAIAgentsAdapter):
            raise TypeError("The selected factory did not return the installed OpenAI adapter")
        try:
            result = await adapter.arun(RunRequest({"prompt": prompt}))
        except AdapterBlocked as error:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "error": str(error),
                        "records": [
                            {
                                "point": record.interception_point.value,
                                "decision": record.verdict.decision.value,
                            }
                            for record in adapter.records
                        ],
                        "records_dropped": adapter.records_dropped,
                    }
                )
            )
            return 2
        print(
            json.dumps(
                {
                    "status": "allowed",
                    "output": result.output,
                    "records": [
                        {
                            "point": record.interception_point.value,
                            "decision": record.verdict.decision.value,
                        }
                        for record in result.records
                    ],
                    "records_dropped": result.records_dropped,
                }
            )
        )
        if adapter.cleanup_error is not None:
            raise adapter.cleanup_error
        return 0
    finally:
        await transport.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
