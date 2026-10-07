"""Run real local Qwen with one application-owned native create_agent middleware."""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx2
from agent_hooks import AgentContext, Verdict
from agent_hooks_adapters import (
    AdapterBlocked,
    Backend,
    NativeRuntime,
    RunRequest,
    with_agent_controls,
)
from agent_hooks_langchain import LangChainControls
from agent_hooks_langgraph.example_telemetry import ExampleTelemetry
from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI
from pydantic import SecretStr


class LengthGuard:
    """Deny oversized actual output; every decision runs on the shared policy lane."""

    def intercept(self, context: AgentContext, /) -> Verdict:
        if context["interception_point"] == "post_model_call":
            content: object = context["target"]["content"]
            if isinstance(content, str) and len(content) > 800:
                return Verdict.deny(reason="output-too-long")
        return Verdict.allow()


async def main(synthetic_data: str | None = None) -> None:
    """Build the native agent visibly and close caller-owned clients in all outcomes."""
    telemetry = ExampleTelemetry(synthetic_data=synthetic_data)

    sync_http = httpx2.Client(trust_env=False, follow_redirects=False)
    async_http: httpx2.AsyncClient | None = None
    sync_client: OpenAI | None = None
    async_client: AsyncOpenAI | None = None
    try:
        async_http = httpx2.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            transport=telemetry.transport(httpx2.AsyncHTTPTransport(trust_env=False, retries=0)),
            event_hooks={"request": [telemetry.request], "response": [telemetry.response]},
        )
        sync_client = OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            timeout=60,
            max_retries=0,
            http_client=sync_http,
        )
        async_client = AsyncOpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            timeout=60,
            max_retries=0,
            http_client=async_http,
        )
        model = ChatOpenAI(
            model="qwen2.5:latest",
            api_key=SecretStr("ollama"),
            base_url="http://localhost:11434/v1",
            temperature=0,
            max_completion_tokens=128,
            timeout=60,
            max_retries=0,
            client=sync_client.chat.completions,
            async_client=async_client.chat.completions,
            root_client=sync_client,
            root_async_client=async_client,
        )

        def build(controls: LangChainControls) -> NativeRuntime:
            middleware = controls.middleware("assistant")
            graph = create_agent(
                model=model,
                tools=[],
                system_prompt="Give one concise plain-text answer.",
                middleware=[middleware],
            )
            return NativeRuntime(graph, {"assistant": graph})

        adapter = with_agent_controls(
            backend=Backend.LANGCHAIN,
            runtime=build,
            interceptors=[LengthGuard()],
        )
        try:
            result = await adapter.arun(
                RunRequest({"prompt": "Explain safe tool access in one short sentence."})
            )
        except AdapterBlocked:
            print(
                json.dumps(
                    telemetry.summary(
                        {
                            "status": "blocked",
                            "records": [
                                {
                                    "point": record.interception_point.value,
                                    "decision": record.verdict.decision.value,
                                }
                                for record in adapter.records
                            ],
                            "cleanup_error": adapter.cleanup_error is not None,
                        }
                    )
                )
            )
            raise
        print(
            json.dumps(
                telemetry.summary(
                    {
                        "status": "allowed",
                        "output": result.output,
                        "records_dropped": result.records_dropped,
                        "records": [
                            {
                                "point": record.interception_point.value,
                                "decision": record.verdict.decision.value,
                            }
                            for record in result.records
                        ],
                        "cleanup_error": adapter.cleanup_error is not None,
                    }
                )
            )
        )
    finally:
        try:
            if async_client is not None:
                await async_client.close()
        finally:
            try:
                if sync_client is not None:
                    sync_client.close()
            finally:
                try:
                    if async_http is not None:
                        await async_http.aclose()
                finally:
                    sync_http.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-diagnostics", choices=("local-toy-v1",))
    asyncio.run(main(parser.parse_args().synthetic_diagnostics))
