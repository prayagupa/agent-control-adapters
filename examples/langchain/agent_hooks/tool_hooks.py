"""Exercise real local Qwen tool selection with actual native create_agent middleware."""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx2
from agent_control_adapters import (
    AdapterBlocked,
    Backend,
    NativeRuntime,
    RunRequest,
    with_agent_controls,
)
from agent_control_langchain import LangChainControls
from agent_control_langgraph.example_telemetry import (
    ExampleTelemetry,
    record_native_result,
    record_run_result,
)
from agent_hooks import AgentContext, Decision, Verdict
from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionNamedToolChoiceParam
from pydantic import Field, SecretStr


class LocalToolModel(ChatOpenAI):
    """Use native tool-choice parameters; every model response comes from real Qwen."""

    registered_tool_name: str = Field(frozen=True)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        choice: ChatCompletionNamedToolChoiceParam = {
            "type": "function",
            "function": {"name": self.registered_tool_name},
        }
        kwargs["parallel_tool_calls"] = False
        kwargs["tool_choice"] = (
            "none" if any(isinstance(message, ToolMessage) for message in messages) else choice
        )
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        record_native_result(result)
        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        choice: ChatCompletionNamedToolChoiceParam = {
            "type": "function",
            "function": {"name": self.registered_tool_name},
        }
        kwargs["parallel_tool_calls"] = False
        kwargs["tool_choice"] = (
            "none" if any(isinstance(message, ToolMessage) for message in messages) else choice
        )
        result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        record_native_result(result)
        return result


class ToolGuard:
    """Deny an explicit registered native tool name before the actual body."""

    def intercept(self, context: AgentContext, /) -> Verdict:
        if (
            context["interception_point"] == "pre_tool_call"
            and context["tool_call"]["name"] == "blocked_echo"
        ):
            return Verdict.deny(reason="tool-not-permitted")
        return Verdict.allow()


async def main(mode: str, synthetic_data: str | None = None) -> None:
    telemetry = ExampleTelemetry(synthetic_data=synthetic_data)

    effects: list[str] = []

    def echo(value: str) -> str:
        """Echo text and count an actual native tool effect."""
        effects.append(value)
        return value

    async def aecho(value: str) -> str:
        """Explicit asynchronous tool body, not a native executor fallback."""
        effects.append(value)
        return value

    tool = StructuredTool.from_function(
        echo,
        coroutine=aecho,
        name="echo" if mode == "allow" else "blocked_echo",
        description="Echo the supplied local text once.",
    )
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
        model = LocalToolModel(
            registered_tool_name=tool.name,
            model="qwen2.5:latest",
            api_key=SecretStr("ollama"),
            base_url="http://localhost:11434/v1",
            temperature=0,
            max_completion_tokens=256,
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
                tools=[tool],
                system_prompt="Call the provided local tool exactly once, then answer briefly.",
                middleware=[middleware],
            )
            return NativeRuntime(graph, {"assistant": graph})

        adapter = with_agent_controls(
            backend=Backend.LANGCHAIN,
            runtime=build,
            interceptors=[ToolGuard()],
        )
        try:
            result = await adapter.arun(
                RunRequest(
                    {
                        "prompt": (
                            f"Call the provided {tool.name} tool once with value "
                            "'local test', then give a short final answer."
                        )
                    }
                )
            )
        except AdapterBlocked as error:
            denied = any(
                record.interception_point.value == "pre_tool_call"
                and record.verdict.decision is Decision.DENY
                for record in adapter.records
            )
            if mode != "deny" or effects or not denied:
                raise
            print(
                json.dumps(
                    telemetry.summary(
                        {
                            "status": "blocked",
                            "tool_effects": len(effects),
                            "tool_effect_values": effects,
                            "records_dropped": adapter.records_dropped,
                            "native_error": type(error).__name__,
                            "sdk_cause": (
                                {
                                    "type": type(error.__cause__).__name__,
                                    "message": str(error.__cause__),
                                }
                                if error.__cause__ is not None
                                else None
                            ),
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
            return
        record_run_result(result, effects)
        if mode != "allow" or len(effects) != 1:
            raise RuntimeError(
                "The native tool demonstration did not produce exactly one allowed effect"
            )
        print(
            json.dumps(
                telemetry.summary(
                    {
                        "status": "allowed",
                        "output": result.output,
                        "tool_effects": len(effects),
                        "tool_effect_values": effects,
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
    parser.add_argument("--mode", choices=("allow", "deny"), default="deny")
    parser.add_argument("--synthetic-diagnostics", choices=("local-toy-v1",))
    args = parser.parse_args()
    asyncio.run(main(args.mode, args.synthetic_diagnostics))
