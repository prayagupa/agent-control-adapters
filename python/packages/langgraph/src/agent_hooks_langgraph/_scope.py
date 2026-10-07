"""Graph-coupled native lifecycle and write-back, delegating all policy work to core."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast
from uuid import uuid4

import langsmith
from agent_hooks import InterceptionRecord
from agent_hooks_adapters import AdapterBlocked, AdapterCleanupError, RunRequest, RunResult
from agent_hooks_adapters._worker import ContextFactory
from agent_hooks_adapters.bindings import PreparedControls
from agent_hooks_adapters.codecs import input_text, output_text, unchanged
from agent_hooks_adapters.json import JSON, normalize, object_value
from agent_hooks_adapters.runtime import (
    NativeExecutionLimits,
    NativeTaskOwner,
    PolicySession,
    RunAdmission,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolCall
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState  # pyright: ignore[reportMissingTypeStubs]

from ._inspection import GraphProof, attribute
from ._native import (
    ToolProfile,
    call_value,
    final_message,
    model_identity,
    model_profile,
    model_value,
    project_messages,
    replace_messages,
    replace_model,
)
from ._routing import Route, RoutingLedger

_T = TypeVar("_T")
_LOG = logging.getLogger(__name__)
_BLOCKED = "Native graph execution blocked by governance."


class NativeGraphRunner(Protocol):
    """Narrow, typed invocation surface of the already validated actual compiled graph."""

    def invoke(self, input: MessagesState, config: RunnableConfig) -> object: ...

    async def ainvoke(self, input: MessagesState, config: RunnableConfig) -> object: ...


class TracingScope(Protocol):
    """Typed boundary for the upstream scoped tracing API's incomplete annotations."""

    def __call__(self, *, enabled: bool) -> AbstractContextManager[None]: ...


tracing_context = cast("TracingScope", attribute(langsmith, "tracing_context"))


@dataclass(frozen=True)
class ModelStep:
    """Host-owned origin and correlation for exactly one native model invocation."""

    actor: object
    model: BaseChatModel
    tools: tuple[ToolProfile, ...]
    request_id: str


@dataclass(frozen=True)
class ToolStep:
    """Proven destination and source accounting for one actual native tool invocation."""

    actor: object
    call: ToolCall
    tool: ToolProfile
    source: int
    identifier: str


class NativeScope:
    """Own graph proof and routing only; core owns evaluator, worker and run budgets."""

    def __init__(
        self,
        framework: str,
        controls: PreparedControls,
        proof: GraphProof,
        entry: object,
        tools: tuple[tuple[object, tuple[ToolProfile, ...]], ...],
        routes: tuple[Route, ...],
        revalidate: Callable[[], None],
    ) -> None:
        self.session = PolicySession(framework, controls)
        self.proof = proof
        self.entry = entry
        self.tools = tools
        self.ledger = RoutingLedger(self.session, routes)
        self._revalidate = revalidate
        self.active = False
        self.model_calls = 0
        self.before_agent_calls = 0
        self.last_model: object | None = None

    def revalidate(self) -> None:
        """Check actual graph and native objects without activating inactive guards."""
        self.proof.revalidate()
        self._revalidate()
        for _, profiles in self.tools:
            for tool in profiles:
                tool.revalidate()

    def checked(self, operation: Callable[[], _T]) -> _T:
        """Fail closed at codec/budget/policy seams with a fixed native error."""
        try:
            if not self.active:
                raise ValueError("Governed graph scope is inactive")
            self.revalidate()
            return operation()
        except Exception as error:
            if not self.session.policy_failed:
                self.session.fail(error, policy=True)
            raise AdapterBlocked(_BLOCKED) from None

    def emit(
        self,
        actor: object,
        factory: ContextFactory,
        validate: Callable[[object], object] = normalize,
    ) -> object:
        """Submit synchronous policy work only to the shared private lane."""
        return self.checked(
            lambda: self.session.emit(self.session.binding(actor), factory, validate)
        )

    async def aemit(
        self,
        actor: object,
        factory: ContextFactory,
        validate: Callable[[object], object] = normalize,
    ) -> object:
        """Await the shared bounded policy lane, never an executor bridge."""
        self.checked(lambda: None)
        try:
            return await self.session.aemit(self.session.binding(actor), factory, validate)
        except Exception as error:
            if not self.session.policy_failed:
                self.session.fail(error, policy=True)
            raise AdapterBlocked(_BLOCKED) from None

    def _model_before(
        self,
        actor: object,
        model: BaseChatModel,
        tools: tuple[ToolProfile, ...],
        messages: list[AnyMessage],
    ) -> tuple[ModelStep, ContextFactory]:
        model_profile(model)
        projected = project_messages(messages)
        step = ModelStep(actor, model, tools, str(uuid4()))
        self.session.budget.before_model(self.session.binding(actor), step.request_id)
        return step, lambda builder: builder.pre_model_call(
            model_id=model_identity(model),
            messages=projected,
            request_id=step.request_id,
            tools=[{"name": tool.name, "parameters": tool.schema_json} for tool in tools],
        )

    def before_model(
        self,
        actor: object,
        model: BaseChatModel,
        tools: tuple[ToolProfile, ...],
        messages: list[AnyMessage],
    ) -> tuple[ModelStep, list[AnyMessage]]:
        """Replace actual pre-model messages before native transport invocation."""
        step, factory = self.checked(lambda: self._model_before(actor, model, tools, messages))
        result = self.emit(actor, factory, lambda value: replace_messages(value, messages))
        self.checked(self.revalidate)
        return step, cast("list[AnyMessage]", result)

    async def abefore_model(
        self,
        actor: object,
        model: BaseChatModel,
        tools: tuple[ToolProfile, ...],
        messages: list[AnyMessage],
    ) -> tuple[ModelStep, list[AnyMessage]]:
        """Apply executable pre-model replacement on the caller's native loop."""
        step, factory = self.checked(lambda: self._model_before(actor, model, tools, messages))
        result = await self.aemit(actor, factory, lambda value: replace_messages(value, messages))
        self.checked(self.revalidate)
        return step, cast("list[AnyMessage]", result)

    def _model_after(self, step: ModelStep, value: object) -> tuple[AIMessage, ContextFactory]:
        projected = model_value(value, step.tools)
        if not isinstance(value, AIMessage):
            raise ValueError("Unsupported native model response")
        calls = cast("list[dict[str, JSON]]", projected["tool_calls"])
        usage = (
            {
                key: value.usage_metadata[key]
                for key in ("input_tokens", "output_tokens", "total_tokens")
            }
            if value.usage_metadata is not None
            else None
        )
        return value, lambda builder: builder.post_model_call(
            model_id=model_identity(step.model),
            content=projected["content"],
            tool_calls=calls,
            finish_reason=cast("str", projected["finish_reason"]),
            usage=usage,
            request_id=step.request_id,
        )

    def _complete_model(self, step: ModelStep, message: AIMessage) -> AIMessage:
        projected = model_value(message, step.tools)
        calls = cast("list[dict[str, JSON]]", projected["tool_calls"])
        for call in calls:
            self.ledger.reserve(step.actor, call)
        self.session.budget.after_model(
            self.session.binding(step.actor),
            step.request_id,
            [cast("str", call["id"]) for call in calls],
        )
        self.model_calls += 1
        self.last_model = step.actor
        return message

    def after_model(self, step: ModelStep, value: object) -> AIMessage:
        """Transform one actual native AIMessage, then reserve its closed tool route."""
        message, factory = self.checked(lambda: self._model_after(step, value))
        result = self.emit(
            step.actor, factory, lambda target: replace_model(target, message, step.tools)
        )
        return self.checked(lambda: self._complete_model(step, cast("AIMessage", result)))

    async def aafter_model(self, step: ModelStep, value: object) -> AIMessage:
        """Return the actual transformed native message with unchanged routing/usage."""
        message, factory = self.checked(lambda: self._model_after(step, value))
        result = await self.aemit(
            step.actor, factory, lambda target: replace_model(target, message, step.tools)
        )
        return self.checked(lambda: self._complete_model(step, cast("AIMessage", result)))

    def _tool_before(
        self, actor: object, value: object, tools: tuple[ToolProfile, ...]
    ) -> tuple[ToolStep, ContextFactory]:
        call, tool = call_value(value, tools)
        identifier = call["id"]
        if not isinstance(identifier, str):
            raise ValueError("Native tool correlation must be text")
        canonical: dict[str, JSON] = {
            "id": call["id"],
            "name": call["name"],
            "args": object_value(call["args"]),
        }
        source = self.ledger.before(actor, canonical)
        step = ToolStep(actor, call, tool, source, identifier)
        return step, lambda builder: builder.pre_tool_call(
            call_id=identifier,
            name=tool.name,
            args=call["args"],
        )

    def before_tool(
        self, actor: object, value: object, tools: tuple[ToolProfile, ...]
    ) -> tuple[ToolStep, dict[str, JSON]]:
        """Validate original args and destination before policy and the native parser."""
        step, factory = self.checked(lambda: self._tool_before(actor, value, tools))
        target = self.emit(actor, factory, step.tool.arguments)
        return step, object_value(target)

    async def abefore_tool(
        self, actor: object, value: object, tools: tuple[ToolProfile, ...]
    ) -> tuple[ToolStep, dict[str, JSON]]:
        """Authorize one proven tool effect without blocking the caller loop."""
        step, factory = self.checked(lambda: self._tool_before(actor, value, tools))
        target = await self.aemit(actor, factory, step.tool.arguments)
        return step, object_value(target)

    def _tool_factory(
        self, step: ToolStep, arguments: dict[str, JSON], result: object
    ) -> ContextFactory:
        value = normalize(result)
        return lambda builder: builder.post_tool_call(
            call_id=step.identifier,
            name=step.tool.name,
            args=arguments,
            value=value,
        )

    def _complete_tool(self, step: ToolStep, value: object) -> JSON:
        canonical: dict[str, JSON] = {
            "id": step.call["id"],
            "name": step.call["name"],
            "args": object_value(step.call["args"]),
        }
        self.ledger.after(step.actor, canonical, step.source)
        return normalize(value)

    def after_tool(self, step: ToolStep, arguments: dict[str, JSON], result: object) -> JSON:
        """Record and replace the actual completed result, never fabricate missing posts."""
        factory = self.checked(lambda: self._tool_factory(step, arguments, result))
        target = self.emit(step.actor, factory)
        return self.checked(lambda: self._complete_tool(step, target))

    async def aafter_tool(self, step: ToolStep, arguments: dict[str, JSON], result: object) -> JSON:
        """Apply actual native tool result replacement and complete source accounting."""
        factory = self.checked(lambda: self._tool_factory(step, arguments, result))
        target = await self.aemit(step.actor, factory)
        return self.checked(lambda: self._complete_tool(step, target))

    def startup(self, actor: object) -> tuple[ContextFactory, Callable[[object], object]]:
        """Build a notification-only startup target from registered native tools."""
        profiles = next((items for native, items in self.tools if native is actor), ())
        names = [tool.name for tool in profiles]
        return (
            lambda builder: builder.agent_startup(tools_registered=names),
            lambda value: unchanged(value, {"tools_registered": names}),
        )

    def finished(self, value: object, *, before_agent: bool) -> tuple[object, str]:
        """Require positive native model coverage and no pending route before success."""
        if (
            self.session.policy_failed
            or self.model_calls < 1
            or self.last_model is None
            or self.session.budget.snapshot.pending
            or self.ledger.pending
            or (before_agent and self.before_agent_calls != 1)
        ):
            raise ValueError("Native governance coverage or tool correlation is incomplete")
        message = final_message(value)
        return self.last_model, cast("str", message.content)


class GraphAdapter:
    """Real invoke/ainvoke graph execution; native clients and loops remain caller-owned."""

    def __init__(self, scope: NativeScope, *, before_agent: bool = False) -> None:
        self._scope = scope
        self._before_agent = before_agent
        self._admission = RunAdmission()
        self._owner = NativeTaskOwner(scope.session.fail, trace_id=scope.session.trace_id)
        self._runner = cast("NativeGraphRunner", scope.proof.graph)

    @property
    def records(self) -> tuple[InterceptionRecord, ...]:
        """Return only actual, detached SDK audit records."""
        return self._scope.session.records

    @property
    def records_dropped(self) -> int:
        """Expose exact bounded SDK record eviction."""
        return self._scope.session.records_dropped

    @property
    def cleanup_error(self) -> AdapterCleanupError | None:
        """Report retained native work or incomplete policy-lane cleanup."""
        return self._owner.cleanup_error or self._scope.session.cleanup_error

    def _config(self) -> RunnableConfig:
        limits = NativeExecutionLimits()
        return {"recursion_limit": limits.recursion, "max_concurrency": limits.tool_concurrency}

    def _shutdown_error(self, error: Exception, primary: BaseException | None) -> None:
        _LOG.error(
            "adapter_shutdown_failed",
            extra={"trace_id": self._scope.session.trace_id, "error_type": type(error).__name__},
        )
        if primary is None:
            raise error
        primary.add_note("Governance shutdown audit failed.")

    def run(self, request: RunRequest | None = None) -> RunResult:
        """Invoke synchronously on this thread; application transport deadlines apply."""
        prompt = self._admission.admit(request)
        scope = self._scope
        scope.revalidate()
        scope.session.begin()
        scope.active = True
        primary: BaseException | None = None
        try:
            for actor in scope.session.controls.actors:
                factory, validate = scope.startup(actor.native)
                scope.emit(actor.native, factory, validate)
            transformed = scope.emit(
                scope.entry, lambda builder: builder.input(content=prompt), input_text
            )
            with tracing_context(enabled=False):
                result: object = self._runner.invoke(
                    {"messages": [HumanMessage(content=cast("str", transformed))]}, self._config()
                )
            actor, text = scope.checked(
                lambda: scope.finished(result, before_agent=self._before_agent)
            )
            output = scope.emit(actor, lambda builder: builder.output(content=text), output_text)
        except BaseException as error:
            primary = error
            if scope.session.policy_failed:
                raise AdapterBlocked(_BLOCKED) from scope.session.last_error
            raise
        finally:
            try:
                for index in range(len(scope.session.controls.actors)):
                    scope.session.emit(
                        index,
                        lambda builder: builder.agent_shutdown(
                            reason="completed" if primary is None else "failed"
                        ),
                        audit=True,
                    )
            except Exception as error:
                self._shutdown_error(error, primary)
            finally:
                scope.active = False
                scope.ledger.close()
                scope.session.close(primary)
        return RunResult(cast("str", output), self.records, self.records_dropped)

    async def arun(self, request: RunRequest | None = None) -> RunResult:
        """Await native ainvoke on the caller loop with shared deadline/drain ownership."""
        prompt = self._admission.admit(request)
        scope = self._scope
        scope.revalidate()

        async def execute() -> str:
            scope.session.begin()
            scope.active = True
            primary: BaseException | None = None
            try:
                for actor in scope.session.controls.actors:
                    factory, validate = scope.startup(actor.native)
                    await scope.aemit(actor.native, factory, validate)
                transformed = await scope.aemit(
                    scope.entry, lambda builder: builder.input(content=prompt), input_text
                )
                with tracing_context(enabled=False):
                    result: object = await self._runner.ainvoke(
                        {"messages": [HumanMessage(content=cast("str", transformed))]},
                        self._config(),
                    )
                actor, text = scope.checked(
                    lambda: scope.finished(result, before_agent=self._before_agent)
                )
                output = await scope.aemit(
                    actor, lambda builder: builder.output(content=text), output_text
                )
                return cast("str", output)
            except BaseException as error:
                primary = error
                if scope.session.policy_failed:
                    raise AdapterBlocked(_BLOCKED) from scope.session.last_error
                raise
            finally:
                try:
                    for index in range(len(scope.session.controls.actors)):
                        await scope.session.aemit(
                            index,
                            lambda builder: builder.agent_shutdown(
                                reason="completed" if primary is None else "failed"
                            ),
                            audit=True,
                        )
                except Exception as error:
                    self._shutdown_error(error, primary)
                finally:
                    scope.active = False
                    scope.ledger.close()
                    await scope.session.aclose(primary)

        output = await self._owner.run(execute)
        return RunResult(output, self.records, self.records_dropped)
