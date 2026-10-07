"""Deterministic model transport only; runner, plugin, SDK and policy engines remain real."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from agent_control_adapters import AgentPolicy, AsyncAdapter, Backend, with_agent_controls
from agent_hooks import ALLOW, AgentContext, Decision, Interceptor, Transform, Verdict
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import Field

POINTS = (
    "agent_startup",
    "input",
    "pre_model_call",
    "post_model_call",
    "pre_tool_call",
    "post_tool_call",
    "output",
    "agent_shutdown",
)


class Allow:
    """Use an actual SDK allow verdict."""

    def intercept(self, context: AgentContext) -> Verdict:
        return ALLOW


class Control:
    """Deterministic interceptor, never a replacement engine or model example."""

    def __init__(self, point: str, *, replacement: object = None, deny: bool = False) -> None:
        self.point = point
        self.replacement = replacement
        self.deny = deny
        self.seen: list[AgentContext] = []

    def intercept(self, context: AgentContext) -> Verdict:
        self.seen.append(context)
        if context["interception_point"] != self.point:
            return ALLOW
        if self.deny:
            return Verdict(decision=Decision.DENY, reason="private policy test detail")
        return Verdict(
            decision=Decision.TRANSFORM, transform=Transform("$target", self.replacement)
        )


def _responses() -> list[LlmResponse]:
    return []


def _requests() -> list[LlmRequest]:
    return []


class BoundaryModel(BaseLlm):
    """Stub only model I/O; use the installed native runner everywhere else."""

    responses: list[LlmResponse] = Field(default_factory=_responses)
    seen: list[LlmRequest] = Field(default_factory=_requests)
    repeat: bool = False

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        if stream:
            raise ValueError("Deterministic transport rejects streaming")
        self.seen.append(llm_request.model_copy(deep=True))
        index = 0 if self.repeat else len(self.seen) - 1
        if index >= len(self.responses):
            raise RuntimeError("Unexpected deterministic model transport call")
        yield self.responses[index].model_copy(deep=True)


@dataclass
class NativeCase:
    runner: Runner
    agent: LlmAgent
    model: BoundaryModel
    effects: list[str]


def text_response(text: str = "native answer") -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        model_version="native-version",
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=3,
            candidates_token_count=2,
            total_token_count=5,
        ),
    )


def tool_response(args: dict[str, object] | None = None, *, calls: int = 1) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id=f"call-{index}",
                        name="ping",
                        args={"value": "native"} if args is None else args,
                    )
                )
                for index in range(calls)
            ],
        )
    )


def native_case(*, use_tool: bool = False) -> NativeCase:
    effects: list[str] = []

    async def ping(value: str) -> dict[str, str]:
        """Return a harmless input after recording the actual tool body effect."""
        effects.append(value)
        return {"result": value}

    model = BoundaryModel(
        model="deterministic-model-boundary",
        responses=[tool_response(), text_response()] if use_tool else [text_response()],
    )
    agent = LlmAgent(
        name="trusted_actor",
        mode="chat",
        model=model,
        instruction="Trusted system text",
        tools=[ping] if use_tool else [],
    )
    runner = Runner(
        agent=agent,
        app_name="caller_app",
        session_service=InMemorySessionService(),
    )
    return NativeCase(runner, agent, model, effects)


def controlled(
    case: NativeCase,
    *,
    interceptors: Sequence[Interceptor] | None = None,
    policy_file: Path | None = None,
    max_records: int = 1000,
    agents: Mapping[str, object] | None = None,
    agent_policies: Sequence[AgentPolicy] = (),
) -> AsyncAdapter:
    return with_agent_controls(
        backend=Backend.GOOGLE_ADK,
        runtime=case.runner,
        interceptors=[Allow()] if interceptors is None and policy_file is None else interceptors,
        policy_file=policy_file,
        agents=agents,
        agent_policies=agent_policies,
        resolver=None,
        timeout=5,
        emit_timeout=30,
        max_records=max_records,
    )
