"""Complete native local-tool setup and actual ACS/Rego enforcement against local Qwen."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx2
from agent_hooks_adapters import AdapterBlocked, Backend, RunRequest, with_agent_controls
from agent_hooks_openai_agents import OpenAIAgentsAdapter
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, function_tool
from openai import AsyncOpenAI


async def main() -> int:
    """Run the actual Runner/tools, expose actual effects/audit and close owned transport."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Use word_count on 'policy gates protect local tools' and explain the result.",
    )
    prompt = parser.parse_args().prompt
    effects: list[str] = []

    @function_tool(failure_error_function=None)
    async def word_count(text: str) -> str:
        """Count whitespace-separated words in the supplied plain text."""
        effects.append(text)
        return json.dumps({"words": len(text.split())})

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
            name="local-qwen-tools",
            model=model,
            tools=[word_count],
            instructions=(
                "Use word_count once for the requested text, then explain its actual result."
            ),
            model_settings=ModelSettings(
                temperature=0,
                max_tokens=64,
                parallel_tool_calls=False,
                tool_choice="required",
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
                        "tool_effects": effects,
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
                    "tool_effects": effects,
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
