"""Single-use native Runner scope with actual policy records and fixed denial boundaries."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TypeVar
from uuid import uuid4

from agent_hooks import InterceptionRecord
from agent_hooks_adapters import AdapterBlocked, AdapterCleanupError, RunRequest, RunResult
from agent_hooks_adapters._worker import ContextFactory
from agent_hooks_adapters.bindings import PreparedControls
from agent_hooks_adapters.codecs import (
    input_text,
    model_text,
    output_text,
    text_messages,
    text_value,
    unchanged,
    wire_equal,
)
from agent_hooks_adapters.json import JSON, normalize, object_value
from agent_hooks_adapters.runtime import NativeTaskOwner, PolicySession, RunAdmission
from agents import Agent, ModelSettings, RunConfig, Runner
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import ModelResponse, TResponseInputItem
from agents.models.interface import ModelTracing
from agents.run_config import ToolExecutionConfig
from agents.tool import Tool
from agents.tool_context import ToolContext
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.response_prompt_param import ResponsePromptParam

from ._codec import project_input, project_output, raw_arguments
from ._model import GovernedModel
from ._profile import Profile, ToolSpec, profile, revalidate, validate_arguments, validate_settings

_BLOCK_MESSAGE = "OpenAI Agents execution blocked by governance."
_TOOL_DRAIN_TIMEOUT = 2.0
_LOG = logging.getLogger(__name__)
_T = TypeVar("_T")


class NativeScope:
    """Bind the actual caller actor to one run-owned model/tools clone and sticky lane."""

    def __init__(self, native: Profile, session: PolicySession) -> None:
        self.native = native
        self.session = session
        self.binding = session.binding(native.agent)
        self.model_id = f"{type(native.model).__module__}.{type(native.model).__qualname__}"
        self.owned: Profile | None = None
        self.calls: dict[str, tuple[ToolSpec, ResponseFunctionToolCall]] = {}
        self.seen_calls: set[str] = set()
        self.tool_tasks: set[asyncio.Task[object]] = set()
        self.cleanup_error: AdapterCleanupError | None = None

    def unsupported(self, reason: str) -> None:
        """Deny an unsupported native mode without invoking it."""
        raise ValueError(reason)

    def ensure(self) -> None:
        """Check sticky state and both ownership fingerprints before every effect."""
        if self.session.last_error is not None:
            raise AdapterBlocked(_BLOCK_MESSAGE) from None
        revalidate(self.native)
        if self.owned is not None:
            revalidate(self.owned)

    def checked(self, action: Callable[[], _T]) -> _T:
        """Keep original validation/control errors private at SDK logging boundaries."""
        try:
            self.ensure()
            return action()
        except Exception as error:
            if not self.session.policy_failed:
                self.session.fail(error, policy=True)
            raise AdapterBlocked(_BLOCK_MESSAGE) from None

    async def emit(
        self, factory: ContextFactory, validate: Callable[[object], object] = normalize
    ) -> object:
        """Emit actual SDK policy through its bounded owned thread, never an executor."""
        self.checked(lambda: None)
        try:
            target = await self.session.aemit(self.binding, factory, validate)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self.session.policy_failed:
                self.session.fail(error, policy=True)
            raise AdapterBlocked(_BLOCK_MESSAGE) from None
        self.checked(lambda: None)
        return target

    def validate_model_call(
        self,
        settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> None:
        """Close all actual Model entry parameters, including stored prompt/resume."""
        if previous_response_id is not None or conversation_id is not None or prompt is not None:
            raise ValueError("Stored prompts and resumed conversations are unsupported")
        if output_schema is not None or handoffs or tracing is not ModelTracing.DISABLED:
            raise ValueError("Rich output, handoffs and native tracing are unsupported")
        validate_settings(settings)
        owned = self.owned
        if (
            owned is None
            or len(tools) != len(owned.tools)
            or any(tool is not spec.tool for tool, spec in zip(tools, owned.tools, strict=True))
        ):
            raise ValueError("Foreign or replaced native model tool membership")

    async def before_model(
        self, instructions: str | None, items: str | list[TResponseInputItem]
    ) -> tuple[str | None, list[TResponseInputItem], str]:
        """Reconstruct actual typed model input, including the separate system parameter."""
        correlation = str(uuid4())
        projection = self.checked(lambda: project_input(instructions, items))
        self.checked(lambda: self.session.budget.before_model(self.binding, correlation))
        tools: list[dict[str, JSON]] = [
            {
                "name": spec.tool.name,
                "description": spec.tool.description,
                "parameters": spec.schema,
            }
            for spec in self.native.tools
        ]
        target = await self.emit(
            lambda builder: builder.pre_model_call(
                model_id=self.model_id,
                messages=projection.messages,
                tools=tools,
                request_id=correlation,
            ),
            lambda target: text_messages(target, projection.messages),
        )
        reconstructed = self.checked(lambda: projection.rebuild(target))
        return *reconstructed, correlation

    async def after_model(self, correlation: str, response: ModelResponse) -> ModelResponse:
        """Validate raw calls before native SDK parsers and return a real ModelResponse."""
        projection = self.checked(lambda: project_output(response))
        for call in projection.calls:
            self.checked(lambda call=call: self._register_call(call))
        self.checked(
            lambda: self.session.budget.after_model(
                self.binding, correlation, [call.call_id for call in projection.calls]
            )
        )
        target = projection.target
        transformed = await self.emit(
            lambda builder: builder.post_model_call(
                model_id=self.model_id,
                content=target["content"],
                tool_calls=[object_value(call) for call in target["tool_calls"]]
                if isinstance(target["tool_calls"], list)
                else [],
                finish_reason=text_value(target["finish_reason"]),
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                request_id=correlation,
            ),
            lambda changed: model_text(changed, target),
        )
        return self.checked(lambda: projection.rebuild(transformed))

    def _register_call(self, call: ResponseFunctionToolCall) -> None:
        spec = next((spec for spec in self.native.tools if spec.tool.name == call.name), None)
        if spec is None or call.call_id in self.seen_calls:
            raise ValueError("Unknown or replayed native function call")
        validate_arguments(spec.schema, raw_arguments(call.arguments))
        self.seen_calls.add(call.call_id)
        self.calls[call.call_id] = spec, call.model_copy(deep=True)

    def clone(self) -> Agent[object]:
        """Clone agent/settings/tools/schema lists without copying or closing clients."""
        tools: list[Tool] = []
        for spec in self.native.tools:
            tools.append(
                replace(
                    spec.tool,
                    params_json_schema=object_value(spec.schema),
                    on_invoke_tool=self._callback(spec),
                    tool_input_guardrails=[],
                    tool_output_guardrails=[],
                )
            )
        agent = replace(
            self.native.agent,
            model=GovernedModel(self.native.model, self),
            tools=tools,
            model_settings=replace(self.native.agent.model_settings, parallel_tool_calls=False),
            handoffs=[],
            mcp_servers=[],
            mcp_config={},
            input_guardrails=[],
            output_guardrails=[],
        )
        self.owned = profile(agent)
        return agent

    def _callback(self, spec: ToolSpec) -> Callable[[ToolContext[object], str], Awaitable[JSON]]:
        async def invoke(context: ToolContext[object], raw: str) -> JSON:
            task: asyncio.Task[object] | None = asyncio.current_task()
            if task is None:
                self.checked(lambda: self.unsupported("A native tool task is required"))
                raise AssertionError("Unreachable native tool task")
            self.tool_tasks.add(task)
            try:
                args = self.checked(lambda: self._before_tool(spec, context, raw))
                changed = await self.emit(
                    lambda builder: builder.pre_tool_call(
                        call_id=context.tool_call_id, name=spec.tool.name, args=args
                    ),
                    lambda value: validate_arguments(spec.schema, value),
                )
                actual = self.checked(lambda: validate_arguments(spec.schema, changed))
                serialized = json.dumps(actual, ensure_ascii=True, allow_nan=False)
                call = context.tool_call
                copied = replace(
                    context,
                    tool_arguments=serialized,
                    tool_call=call.model_copy(deep=True, update={"arguments": serialized})
                    if call is not None
                    else None,
                )
                self.checked(lambda: None)
                original: object = await spec.tool.on_invoke_tool(copied, serialized)
                result = self.checked(lambda: normalize(original))
                transformed = await self.emit(
                    lambda builder: builder.post_tool_call(
                        call_id=context.tool_call_id,
                        name=spec.tool.name,
                        args=actual,
                        value=result,
                    )
                )
                self.checked(
                    lambda: self.session.budget.after_tool(self.binding, context.tool_call_id)
                )
                self.calls.pop(context.tool_call_id)
                return self.checked(lambda: normalize(transformed))
            finally:
                self.tool_tasks.discard(task)

        return invoke

    def _before_tool(
        self, spec: ToolSpec, context: ToolContext[object], raw: str
    ) -> dict[str, JSON]:
        owned = self.owned
        pending = self.calls.get(context.tool_call_id)
        if (
            owned is None
            or context.agent is not owned.agent
            or pending is None
            or pending[0] is not spec
            or context.tool_name != spec.tool.name
            or context.tool_namespace is not None
            or context.tool_call is None
            or context.tool_call.call_id != context.tool_call_id
            or context.tool_call.name != spec.tool.name
        ):
            raise ValueError("Native tool call does not match its host-owned binding")
        args = validate_arguments(spec.schema, raw_arguments(raw))
        if not wire_equal(args, raw_arguments(pending[1].arguments)):
            raise ValueError("Native SDK changed original raw arguments before governance")
        self.session.budget.before_tool(self.binding, context.tool_call_id)
        return args

    async def drain_tools(self, primary: BaseException | None) -> None:
        """Cancel/drain only retained native tool tasks, containing resistant callbacks."""
        pending = set(self.tool_tasks)
        if not pending:
            return
        for task in pending:
            task.cancel()
        try:
            _, remaining = await asyncio.wait(pending, timeout=_TOOL_DRAIN_TIMEOUT)
        except asyncio.CancelledError:
            if any(not task.done() for task in pending):
                self.cleanup_error = AdapterCleanupError("Native tool cleanup interrupted.")
                _LOG.error(
                    "adapter_native_tool_cleanup_retained",
                    extra={"trace_id": self.session.trace_id, "error_category": "cleanup"},
                )
            raise
        if remaining:
            self.cleanup_error = AdapterCleanupError("Native tool cleanup incomplete.")
            _LOG.error(
                "adapter_native_tool_cleanup_retained",
                extra={"trace_id": self.session.trace_id, "error_category": "cleanup"},
            )
            if primary is None:
                raise self.cleanup_error
            primary.add_note("Native tool cleanup incomplete.")


class OpenAIAgentsAdapter:
    """One actual Runner run on the application loop; clients and original agents stay open."""

    def __init__(self, native: Profile, controls: PreparedControls) -> None:
        self._session = PolicySession("openai-agents", controls)
        self._scope = NativeScope(native, self._session)
        self._admission = RunAdmission()
        self._owner = NativeTaskOwner(self._session.fail, trace_id=self._session.trace_id)

    @property
    def records(self) -> tuple[InterceptionRecord, ...]:
        """Actual bounded, detached SDK audit records, including after failure."""
        return self._session.records

    @property
    def records_dropped(self) -> int:
        """Exact number of records evicted by the shared lane."""
        return self._session.records_dropped

    @property
    def cleanup_error(self) -> AdapterCleanupError | None:
        """Report containment separately without replacing a primary failure."""
        return self._owner.cleanup_error or self._scope.cleanup_error or self._session.cleanup_error

    def run(self, request: RunRequest | None = None) -> RunResult:
        """Refuse before admission; the same instance can then be awaited correctly."""
        raise RuntimeError(
            "OpenAI Agents requires await adapter.arun(request) on the caller's event loop."
        )

    async def arun(self, request: RunRequest | None = None) -> RunResult:
        """Admit exactly one strict prompt and own bounded native/policy cleanup."""
        prompt = self._admission.admit(request)

        async def execute() -> str:
            scope, session = self._scope, self._session
            scope.checked(lambda: None)
            agent = scope.checked(scope.clone)
            session.begin()
            primary: BaseException | None = None
            try:
                startup: dict[str, JSON] = {
                    "tools_registered": [spec.tool.name for spec in scope.native.tools]
                }
                await scope.emit(
                    lambda builder: builder.agent_startup(
                        tools_registered=[spec.tool.name for spec in scope.native.tools]
                    ),
                    lambda value: unchanged(value, startup),
                )
                transformed = await scope.emit(
                    lambda builder: builder.input(content=prompt), input_text
                )
                actual_prompt = scope.checked(
                    lambda: input_text({"content": transformed, "role": "user"})
                )
                scope.checked(lambda: None)
                result = await Runner.run(
                    agent,
                    actual_prompt,
                    max_turns=10,
                    run_config=RunConfig(
                        tracing_disabled=True,
                        trace_include_sensitive_data=False,
                        tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1),
                    ),
                )
                scope.checked(lambda: None)
                output = scope.checked(lambda: output_text({"content": result.final_output}))
                if scope.calls or session.budget.snapshot.pending:
                    scope.checked(lambda: scope.unsupported("Incomplete native tool correlation"))
                transformed_output = await scope.emit(
                    lambda builder: builder.output(content=output), output_text
                )
                return scope.checked(lambda: output_text({"content": transformed_output}))
            except BaseException as error:
                primary = error
                raise
            finally:
                try:
                    await scope.drain_tools(primary)
                    await session.aemit(
                        scope.binding,
                        lambda builder: builder.agent_shutdown(
                            reason="completed" if primary is None else "failed"
                        ),
                        audit=True,
                    )
                except Exception as cleanup_error:
                    _LOG.error(
                        "adapter_shutdown_failed",
                        extra={
                            "trace_id": session.trace_id,
                            "error_type": type(cleanup_error).__name__,
                        },
                    )
                    if primary is None:
                        raise
                    primary.add_note("Adapter shutdown audit or native cleanup failed.")
                finally:
                    scope.calls.clear()
                    await session.aclose(primary)

        try:
            output = await self._owner.run(execute)
            self._scope.checked(lambda: None)
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._session.policy_failed:
                raise AdapterBlocked(_BLOCK_MESSAGE) from self._session.last_error
            raise
        return RunResult(output, self.records, self.records_dropped)
