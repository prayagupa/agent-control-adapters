"""Run real local Qwen with executable pre/post-model native Agent Hooks transforms."""

from __future__ import annotations

import asyncio
import json

import httpx
from agent_hooks import ALLOW, AgentContext, Decision, Transform, Verdict
from agent_hooks_adapters import Backend, RunRequest, with_agent_controls
from agent_hooks_adapters.json import normalize, object_value
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from openai import AsyncOpenAI


class ModelControls:
    """Trim actual request/response text; do not invent or fixate model output."""

    def intercept(self, context: AgentContext) -> Verdict:
        point = context["interception_point"]
        if point == "pre_model_call":
            messages = normalize(context["target"])
            if not isinstance(messages, list):
                raise ValueError("Expected actual model messages")
            replacement = [object_value(message) for message in messages]
            for message in replacement:
                if isinstance(message.get("content"), str):
                    content = message["content"]
                    if isinstance(content, str):
                        message["content"] = content.strip()
            return Verdict(Decision.TRANSFORM, transform=Transform("$target", replacement))
        if point == "post_model_call":
            target = object_value(context["target"])
            content = target["content"]
            if not isinstance(content, str):
                raise ValueError("Expected real plain model output")
            return Verdict(
                Decision.TRANSFORM, transform=Transform("$target.content", content.strip())
            )
        return ALLOW


async def main() -> None:
    """Own the caller client and close it on this same loop after actual native execution."""
    model_requests = 0

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
        )
        agent = LlmAgent(
            name="local_qwen",
            mode="chat",
            model=model,
            instruction="Answer briefly in plain text. Do not claim actions you did not perform.",
        )
        runner = Runner(
            agent=agent,
            app_name="caller_app",
            session_service=InMemorySessionService(),
        )
        adapter = with_agent_controls(
            backend=Backend.GOOGLE_ADK,
            runtime=runner,
            interceptors=[ModelControls()],
            policy_file=None,
            agents=None,
            agent_policies=(),
            resolver=None,
            timeout=5,
            emit_timeout=30,
            max_records=128,
        )
        result = await adapter.arun(RunRequest({"prompt": "  Briefly explain an audit trail.  "}))
        if model_requests < 1 or not result.output.strip():
            raise RuntimeError("Real local model execution was not observed")
        for point in ("pre_model_call", "post_model_call"):
            if not any(
                record.interception_point.value == point
                and record.verdict.decision is Decision.TRANSFORM
                for record in result.records
            ):
                raise RuntimeError("The real model did not exercise both model transforms")
        print(
            json.dumps(
                {
                    "status": "allowed",
                    "model": "qwen2.5:latest",
                    "model_requests": model_requests,
                    "output": result.output,
                    "records_dropped": result.records_dropped,
                }
            )
        )
        for record in result.records:
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
