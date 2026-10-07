"""Native graph test transport and real SDK policies; never imported by examples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from agent_hooks import AgentContext, Decision, Transform, Verdict
from agent_hooks_adapters import AsyncAdapter, NativeRuntime
from agent_hooks_adapters.json import JSON, normalize, object_value
from agent_hooks_langgraph import LangGraphControls
from agent_hooks_langgraph.portable import create_backend
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    MessagesState,
    StateGraph,
)
from langgraph.prebuilt import tools_condition
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


def _seen_messages() -> list[list[BaseMessage]]:
    return []


class NativeTransport(BaseChatModel):
    """Stub only chat I/O; BaseChatModel, compiled graph and middleware remain native."""

    responses: list[AIMessage]
    calls: int = Field(default=0, exclude=True)
    seen: list[list[BaseMessage]] = Field(default_factory=_seen_messages, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "unit-test-native-transport"

    def bind_tools(
        self, tools: Sequence[object], *, tool_choice: str | None = None, **kwargs: object
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    def _response(self, messages: list[BaseMessage]) -> ChatResult:
        self.seen.append(list(messages))
        response = self.responses[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        return self._response(messages)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        return self._response(messages)


class NativePolicy:
    """Use the real SDK interceptor contract with deterministic boundary verdicts."""

    def __init__(self, *, deny: str | None = None, transform: bool = False) -> None:
        self.deny = deny
        self.transform = transform
        self.points: list[str] = []

    def intercept(self, ctx: AgentContext, /) -> Verdict:
        point = str(ctx["interception_point"])
        self.points.append(point)
        if point == self.deny:
            return Verdict.deny(reason="unit-boundary-deny")
        if not self.transform:
            return Verdict.allow()
        target = normalize(ctx["target"])
        if point == "input":
            target = {"role": "user", "content": "transformed input"}
        elif point == "pre_model_call":
            messages = cast("list[dict[str, JSON]]", target)
            target = [
                {
                    **message,
                    "content": "transformed system"
                    if message["role"] == "system"
                    else "transformed request",
                }
                if message["role"] in ("system", "user")
                else message
                for message in messages
            ]
        elif point == "post_model_call":
            target = {**object_value(target), "content": "transformed model"}
        elif point == "pre_tool_call":
            target = {"value": "transformed arguments"}
        elif point == "post_tool_call":
            target = "transformed tool result"
        elif point == "output":
            target = {"content": "transformed output"}
        else:
            return Verdict.allow()
        return Verdict(decision=Decision.TRANSFORM, transform=Transform("$target", target))


def model_and_tool(*, use_tool: bool = True) -> tuple[NativeTransport, StructuredTool, list[str]]:
    """Create explicit sync/async native tool bodies and deterministic chat transport."""
    effects: list[str] = []

    def ping(value: str) -> str:
        """Echo an argument and record the actual native effect."""
        effects.append(value)
        return value

    async def aping(value: str) -> str:
        """Echo an argument with explicit native asynchronous I/O."""
        effects.append(value)
        return value

    tool = StructuredTool.from_function(ping, coroutine=aping, name="ping", description="Echo text")
    responses = [
        AIMessage(
            content="native final",
            id="response-final",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        )
    ]
    if use_tool:
        responses.insert(
            0,
            AIMessage(
                content="",
                id="response-tool",
                tool_calls=[{"id": "call-1", "name": "ping", "args": {"value": "original"}}],
                response_metadata={"finish_reason": "tool_calls"},
                usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            ),
        )
    return NativeTransport(responses=responses), tool, effects


@dataclass
class GraphCase:
    model: NativeTransport
    tool: StructuredTool
    effects: list[str]
    graph: object | None = None
    controls: LangGraphControls | None = None

    def build(self, controls: LangGraphControls) -> NativeRuntime:
        self.controls = controls
        model_node = controls.model_node("model", self.model, [self.tool])
        tool_node = controls.tool_node("tools", [self.tool])
        builder = StateGraph(MessagesState)
        builder.add_node("model", model_node)  # pyright: ignore[reportUnknownMemberType]
        builder.add_node("tools", tool_node)  # pyright: ignore[reportUnknownMemberType]
        builder.add_edge(START, "model")
        builder.add_conditional_edges("model", tools_condition, {"tools": "tools", "__end__": END})
        builder.add_edge("tools", "model")
        self.graph = builder.compile()  # pyright: ignore[reportUnknownMemberType]
        return NativeRuntime(self.graph, {"model": model_node, "tools": tool_node})


def graph_case() -> GraphCase:
    model, tool, effects = model_and_tool()
    return GraphCase(model, tool, effects)


def graph_adapter(
    case: GraphCase, policy: NativePolicy, *, max_records: int = 1000
) -> AsyncAdapter:
    return create_backend(
        runtime=case.build,
        interceptors=[policy],
        policy_file=None,
        agents=None,
        agent_policies=(),
        resolver=None,
        timeout=5,
        emit_timeout=30,
        max_records=max_records,
    )
