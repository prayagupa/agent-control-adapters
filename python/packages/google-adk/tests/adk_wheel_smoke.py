"""Wheel-only public-factory integration; stub model transport, never the native runner."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import subprocess
import sys
from collections.abc import AsyncGenerator
from pathlib import Path

import agent_control_adapters
import agent_control_google_adk
from agent_control_adapters import Backend, RunRequest, with_agent_controls
from agent_hooks import ALLOW, AgentContext, Verdict
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


class Allow:
    """Evaluate a real native SDK allow interceptor."""

    def intercept(self, context: AgentContext) -> Verdict:
        return ALLOW


class TransportBoundary(BaseLlm):
    """Replace only model I/O for the wheel integration test."""

    calls: int = 0

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        if stream:
            raise ValueError("Wheel integration rejects streaming")
        self.calls += 1
        part = (
            types.Part(
                function_call=types.FunctionCall(
                    name="echo",
                    id="wheel-call",
                    args={"value": "wheel-argument"},
                )
            )
            if self.calls == 1
            else types.Part(text="wheel test model transport")
        )
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


async def smoke() -> None:
    """Require non-editable imports and actual SDK records for all eight native points."""
    prefix = Path(sys.prefix).resolve()
    for module in (agent_control_adapters, agent_control_google_adk):
        if module.__file__ is None or not Path(module.__file__).resolve().is_relative_to(prefix):
            raise RuntimeError("Wheel smoke rejects editable/source fallback")
    effects: list[str] = []

    async def echo(value: str) -> dict[str, str]:
        """Record an actual harmless tool invocation."""
        effects.append(value)
        return {"result": value}

    model = TransportBoundary(model="wheel-test-model")
    agent = LlmAgent(name="wheel_actor", mode="chat", model=model, tools=[echo])
    caller = Runner(
        agent=agent,
        app_name="wheel_caller",
        session_service=InMemorySessionService(),
    )
    try:
        adapter = with_agent_controls(
            backend=Backend.GOOGLE_ADK,
            runtime=caller,
            interceptors=[Allow()],
        )
        result = await adapter.arun(RunRequest({"prompt": "Wheel public-factory integration test"}))
        points = {record.interception_point.value for record in result.records}
        if (
            effects != ["wheel-argument"]
            or model.calls != 2
            or points
            != {
                "agent_startup",
                "input",
                "pre_model_call",
                "post_model_call",
                "pre_tool_call",
                "post_tool_call",
                "output",
                "agent_shutdown",
            }
            or adapter.cleanup_error is not None
        ):
            raise RuntimeError("Actual wheel integration contract failed")
        print(
            json.dumps(
                {
                    "status": "PASSED",
                    "prefix": str(prefix),
                    "namespace_file": agent_control_google_adk.__file__,
                    "core_file": agent_control_adapters.__file__,
                    "backend": Backend.GOOGLE_ADK.value,
                    "model_calls": model.calls,
                    "tool_effects": effects,
                    "actual_sdk_records": len(result.records),
                    "versions": {
                        name: importlib.metadata.version(name)
                        for name in (
                            "google-adk",
                            "agent-control-google-adk",
                            "agent-control-adapters",
                            "agent-hooks-sdk",
                            "agent-control-spec",
                        )
                    },
                },
                indent=2,
            )
        )
    finally:
        await caller.close()


class Arguments(argparse.Namespace):
    """Typed optional isolated interpreter selection."""

    python: Path | None = None


def main() -> None:
    """Run directly in an isolated wheel environment or launch that exact interpreter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    if arguments.python is not None:
        subprocess.run(
            [str(arguments.python.absolute()), "-I", str(Path(__file__).resolve())],
            check=True,
            timeout=120,
        )
    else:
        asyncio.run(smoke())


if __name__ == "__main__":
    main()
