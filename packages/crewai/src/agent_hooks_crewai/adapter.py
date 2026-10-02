"""Public single-use adapter and validated host hook mapping."""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Protocol, cast
from uuid import uuid4

from agent_hooks import ApprovalResolver, InterceptionRecord, Interceptor
from crewai import Agent, Crew, Process, Task
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.crews.crew_output import CrewOutput
from crewai.experimental.agent_executor import AgentExecutor
from crewai.hooks.contexts import (
    ExecutionEndContext,
    ExecutionStartContext,
    InputContext,
    OutputContext,
)
from crewai.hooks.dispatch import InterceptionPoint, get_scoped_hooks, scoped_hooks
from crewai.hooks.llm_hooks import LLMCallHookContext
from crewai.hooks.tool_hooks import ToolCallHookContext
from crewai.utilities.types import LLMMessage

from ._json import JSON, envelope, messages_value, normalize, object_value, text_value
from ._worker import Binding, ContextFactory, Worker
from .errors import AdapterBlocked, AdapterCleanupError

logger = logging.getLogger(__name__)
HookContext = (
    ExecutionStartContext
    | ExecutionEndContext
    | InputContext
    | OutputContext
    | LLMCallHookContext
    | ToolCallHookContext
)

_CONTEXT_TYPES: dict[InterceptionPoint, type[HookContext]] = {
    InterceptionPoint.EXECUTION_START: ExecutionStartContext,
    InterceptionPoint.INPUT: InputContext,
    InterceptionPoint.OUTPUT: OutputContext,
    InterceptionPoint.EXECUTION_END: ExecutionEndContext,
    InterceptionPoint.PRE_MODEL_CALL: LLMCallHookContext,
    InterceptionPoint.POST_MODEL_CALL: LLMCallHookContext,
    InterceptionPoint.PRE_TOOL_CALL: ToolCallHookContext,
    InterceptionPoint.POST_TOOL_CALL: ToolCallHookContext,
}


class _SynchronousCrew(Protocol):
    def kickoff(self, inputs: dict[str, JSON] | None = None) -> object: ...


@dataclass(frozen=True)
class AgentPolicy:
    """Additional ordered controls bound to one actual CrewAI Agent, not its role."""

    agent: Agent
    interceptors: tuple[Interceptor, ...]


class CrewAIAdapter:
    """Govern one synchronous sequential crew kickoff with execution-scoped hooks.

    Shared controls precede agent controls under SDK sequential/run_all composition.
    Unsupported modes fail admission. Instances are single-use, including failures.
    Agents must disable context summarization; structured tasks are unsupported.
    The last configured iteration is reserved and denied before its model request
    so neither pinned executor can enter its unhooked forced-final call.
    Policy timeouts deny the run; noncooperative cleanup is exposed in cleanup_error
    and attached as a note to any already-propagating exception.
    """

    def __init__(
        self,
        crew: Crew,
        *,
        interceptors: Sequence[Interceptor],
        agent_policies: Sequence[AgentPolicy] = (),
        resolver: ApprovalResolver | None = None,
        timeout: float = 5,
        emit_timeout: float = 30,
        max_records: int = 1000,
    ) -> None:
        if not isinstance(cast("object", crew), Crew):
            raise TypeError("crew must be a Crew")
        for duration in (timeout, emit_timeout):
            if isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
                raise ValueError("Timeouts must be positive finite seconds")
        if type(max_records) is not int or not 1 <= max_records <= 100_000:
            raise ValueError("max_records must be between 1 and 100000")
        self._crew = crew
        self._run_id = str(uuid4())
        self._trace_id = self._run_id.replace("-", "")
        self._shared = tuple(interceptors)
        self._policies = tuple(agent_policies)
        self._resolver = resolver
        self._timeout = timeout
        self._emit_timeout = emit_timeout
        self._max_records = max_records
        self._lock = threading.RLock()
        self._used = False
        self._denied = False
        self._started = False
        self._ended = False
        self._output_seen = False
        self._agents: tuple[Agent, ...] = ()
        self._tasks: tuple[Task, ...] = ()
        self._iteration_limits: dict[int, int] = {}
        self._pending: dict[tuple[int, int, str, str], str] = {}
        self._worker: Worker | None = None
        self.cleanup_error: AdapterCleanupError | None = None

    @property
    def records(self) -> tuple[InterceptionRecord, ...]:
        """Detached bounded SDK audit snapshot; missing host emissions are not fabricated."""
        worker = self._worker
        return worker.records if worker is not None else ()

    @property
    def records_dropped(self) -> int:
        """Number of older SDK records evicted by the adapter's record cap."""
        worker = self._worker
        return worker.records_dropped if worker is not None else 0

    def _admit(self) -> dict[int, Binding]:
        crew = self._crew
        if type(crew) is not Crew or crew.process != Process.sequential or not self._shared:
            raise ValueError("A sequential crew and shared interceptors are required")
        self._reject_fields(
            crew,
            (
                "stream",
                "checkpoint",
                "checkpoint_inputs",
                "checkpoint_train",
                "checkpoint_kickoff_event_id",
                "after_kickoff_callbacks",
                "before_kickoff_callbacks",
                "planning",
                "memory",
                "knowledge",
                "knowledge_sources",
                "step_callback",
                "task_callback",
                "manager_agent",
                "manager_llm",
                "function_calling_llm",
                "skills",
            ),
        )
        if not crew.agents or len(crew.agents) > 128 or len(crew.tasks) > 1000:
            raise ValueError("Crew membership exceeds the supported profile")
        agents: list[Agent] = []
        for agent in crew.agents:
            if type(agent) is not Agent or any(
                agent is known or agent.id == known.id for known in agents
            ):
                raise ValueError("Crew requires unique Agent objects")
            if agent.executor_class not in (AgentExecutor, CrewAgentExecutor) or (
                agent.agent_executor is not None
                and type(agent.agent_executor) not in (AgentExecutor, CrewAgentExecutor)
            ):
                raise ValueError("Unsupported CrewAI executor")
            if type(agent.max_iter) is not int or agent.max_iter < 1:
                raise ValueError("Agent max_iter must be a positive integer")
            self._reject_fields(
                agent,
                (
                    "allow_delegation",
                    "step_callback",
                    "memory",
                    "knowledge",
                    "knowledge_sources",
                    "reasoning",
                    "planning",
                    "planning_config",
                    "guardrail",
                    "guardrails",
                    "allow_code_execution",
                    "max_execution_time",
                    "apps",
                    "mcps",
                    "callbacks",
                    "checkpoint",
                    "checkpoint_kickoff_event_id",
                    "skills",
                    "respect_context_window",
                    "function_calling_llm",
                ),
            )
            if agent.agent_executor is not None:
                self._reject_fields(
                    agent.agent_executor,
                    ("respect_context_window", "function_calling_llm"),
                )
            for model in (agent.llm, agent.function_calling_llm):
                if model is not None:
                    self._reject_fields(model, ("stream",))
            agents.append(agent)
            self._iteration_limits[id(agent)] = agent.max_iter
        self._agents = tuple(agents)
        for task in crew.tasks:
            if type(task) is not Task or not any(task.agent is agent for agent in self._agents):
                raise ValueError("Each ordinary task must bind an admitted Agent")
            self._reject_fields(
                task,
                (
                    "async_execution",
                    "human_input",
                    "guardrail",
                    "guardrails",
                    "callback",
                    "checkpoint_original_description",
                    "checkpoint_original_expected_output",
                    "output_file",
                    "output_json",
                    "output_pydantic",
                    "response_model",
                ),
            )
        self._tasks = tuple(crew.tasks)
        controls = {id(agent): self._shared for agent in self._agents}
        seen: set[int] = set()
        for policy in self._policies:
            key = id(policy.agent)
            if key not in controls or key in seen or not policy.interceptors:
                raise ValueError("Agent policies must be unique, nonempty, and crew-owned")
            if not isinstance(cast("object", policy.interceptors), tuple):
                raise TypeError("AgentPolicy.interceptors must be a tuple")
            seen.add(key)
            controls[key] += policy.interceptors
        bindings = {0: Binding(self._run_id, self._shared)}
        for agent in self._agents:
            bindings[id(agent)] = Binding(str(agent.id), controls[id(agent)])
        return bindings

    @staticmethod
    def _reject_fields(owner: object, fields: tuple[str, ...]) -> None:
        for name in fields:
            if getattr(owner, name, None):
                raise ValueError(f"Unsupported CrewAI profile option: {name}")

    def kickoff(self, inputs: dict[str, object] | None = None) -> CrewOutput:
        """Execute once, raising AdapterBlocked on denial; always attempt bounded cleanup."""
        with self._lock:
            if self._used:
                raise RuntimeError("CrewAIAdapter is single-use")
            self._used = True
        original: BaseException | None = None
        try:
            bindings = self._admit()
            payload = object_value(inputs) if inputs is not None else None
            self._worker = Worker(
                bindings, self._resolver, self._timeout, self._emit_timeout, self._max_records
            )
            hooks: dict[InterceptionPoint, list[Callable[[object], object]]] = {
                point: [
                    *get_scoped_hooks(point),
                    *([partial(self._hook, point)] if point in _CONTEXT_TYPES else []),
                ]
                for point in InterceptionPoint
            }
            with scoped_hooks(hooks):
                # CrewAI's optional file-input annotation is unresolved in the release.
                # This profile deliberately exposes only the synchronous inputs boundary.
                result = cast("_SynchronousCrew", self._crew).kickoff(inputs=payload)
                with self._lock:
                    if (
                        self._denied
                        or self._worker.denied
                        or not self._started
                        or not self._ended
                        or not self._output_seen
                        or not isinstance(result, CrewOutput)
                    ):
                        raise AdapterBlocked()
                return result
        except BaseException as error:
            original = error
            raise
        finally:
            try:
                worker = self._worker
                if worker is not None:
                    if self._started and not self._ended:
                        self._shutdown("failed", preserve=True)
                    try:
                        worker.close()
                    except AdapterCleanupError as error:
                        self.cleanup_error = error
                        if original is not None:
                            original.add_note(str(error))
                        else:
                            raise
            finally:
                with self._lock:
                    self._pending.clear()
                    self._agents = ()
                    self._tasks = ()
                    self._iteration_limits.clear()
                    self._policies = ()
                    self._shared = ()
                    self._resolver = None

    def _identity(self, context: HookContext, *, agent_required: bool) -> int:
        if context.crew is not self._crew:
            raise ValueError("Missing or foreign crew identity")
        if len(self._crew.agents) != len(self._agents) or any(
            actual is not frozen
            for actual, frozen in zip(self._crew.agents, self._agents, strict=True)
        ):
            raise ValueError("Crew membership changed")
        agent: object = context.agent
        if agent is not None and not any(agent is known for known in self._agents):
            raise ValueError("Foreign agent identity")
        if agent_required and agent is None:
            raise ValueError("Missing agent identity")
        task: object = context.task
        if task is not None and not any(task is known for known in self._tasks):
            raise ValueError("Foreign task identity")
        if isinstance(task, Task) and agent is not None and task.agent is not agent:
            raise ValueError("Agent does not own this task")
        return id(agent) if agent is not None else 0

    def _hook(self, point: InterceptionPoint, raw: object) -> object:
        try:
            if not isinstance(
                raw,
                (
                    ExecutionStartContext,
                    ExecutionEndContext,
                    InputContext,
                    OutputContext,
                    LLMCallHookContext,
                    ToolCallHookContext,
                ),
            ):
                raise ValueError("Unknown hook context")
            if not isinstance(raw, _CONTEXT_TYPES[point]):
                raise ValueError("Hook context does not match its interception point")
            with self._lock:
                binding = self._identity(
                    raw, agent_required=isinstance(raw, (LLMCallHookContext, ToolCallHookContext))
                )
                if point is not InterceptionPoint.EXECUTION_END and (
                    self._denied or (self._worker is not None and self._worker.denied)
                ):
                    raise AdapterBlocked()
            if isinstance(raw, ExecutionEndContext):
                self._shutdown(raw.status, preserve=raw.error is not None or self._denied)
                return None
            return self._map(point, raw, binding)
        except Exception as error:
            with self._lock:
                self._denied = True
                if self._worker is not None:
                    self._worker.deny()
            logger.warning(
                "adapter_hook_denied",
                extra={
                    "trace_id": self._trace_id,
                    "point": point.value,
                    "error_type": type(error).__name__,
                },
            )
            raise AdapterBlocked() from None

    def _emit(
        self,
        binding: int,
        factory: ContextFactory,
        validate: Callable[[object], object] = normalize,
    ) -> object:
        worker = self._worker
        if worker is None:
            raise AdapterBlocked()
        result = worker.emit(binding, factory, validate)
        with self._lock:
            if self._denied or worker.denied:
                raise AdapterBlocked()
        return result

    def _shutdown(self, reason: str, *, preserve: bool) -> None:
        with self._lock:
            if self._ended:
                return
            self._ended = True
        worker = self._worker
        if worker is None or not worker.available:
            logger.error("adapter_shutdown_not_emitted", extra={"trace_id": self._trace_id})
            return
        try:
            worker.emit(0, lambda builder: builder.agent_shutdown(reason=reason), audit=True)
        except Exception as error:
            logger.error(
                "adapter_shutdown_failed",
                extra={"trace_id": self._trace_id, "error_type": type(error).__name__},
            )
            if not preserve:
                raise AdapterBlocked() from None

    def _call_id(self, binding: int, family: str, name: str, *, pre: bool) -> str:
        key = (threading.get_ident(), binding, family, name)
        with self._lock:
            if pre:
                if key not in self._pending and len(self._pending) >= 1024:
                    raise ValueError("Pending correlation limit exceeded")
                call_id = str(uuid4())
                self._pending[key] = call_id
                return call_id
            call_id = self._pending.pop(key, None)
        if call_id is None:
            logger.warning(
                "adapter_unpaired_post", extra={"trace_id": self._trace_id, "family": family}
            )
            return f"unpaired-{uuid4()}"
        return call_id

    def _map(self, point: InterceptionPoint, context: HookContext, binding: int) -> object:
        if isinstance(context, ExecutionStartContext):
            self._started = True
            tools = sorted({tool.name for agent in self._agents for tool in agent.tools or []})
            tools = sorted(
                set(tools) | {tool.name for task in self._tasks for tool in task.tools or []}
            )

            def startup_target(value: object) -> object:
                target = normalize(value)
                if target != {"tools_registered": tools}:
                    raise ValueError("Startup registration transforms are unsupported")
                return target

            self._emit(
                0, lambda builder: builder.agent_startup(tools_registered=tools), startup_target
            )
            return None
        if isinstance(context, InputContext):
            content = object_value(context.payload) if context.payload is not None else {}

            def input_target(value: object) -> dict[str, JSON]:
                target = envelope(value, {"content", "role"})
                if target["role"] != "user":
                    raise ValueError("Input role cannot change")
                return object_value(target["content"])

            result = object_value(
                self._emit(0, lambda builder: builder.input(content=content), input_target)
            )
            context.payload = result if result or context.payload is not None else None
            return None
        if isinstance(context, OutputContext):
            output: object = context.payload
            if not isinstance(output, CrewOutput):
                raise ValueError("Expected CrewOutput")
            final = next((task for task in reversed(output.tasks_output) if task.raw), None)

            def output_target(value: object) -> str:
                result = text_value(envelope(value, {"content"})["content"])
                if result != output.raw and (
                    final is None
                    or output.pydantic is not None
                    or output.json_dict is not None
                    or final.pydantic is not None
                    or final.json_dict is not None
                ):
                    raise ValueError("Structured output transforms are unsupported")
                return result

            result = text_value(
                self._emit(
                    0, lambda builder: builder.output(content=text_value(output.raw)), output_target
                )
            )
            if result != output.raw and final is not None:
                output.raw = result
                final.raw = result
            self._output_seen = True
            return None
        if isinstance(context, LLMCallHookContext):
            return self._model(point, context, binding)
        if isinstance(context, ToolCallHookContext):
            return self._tool(point, context, binding)
        raise ValueError("Unsupported hook point")

    def _model(self, point: InterceptionPoint, context: LLMCallHookContext, binding: int) -> object:
        pre = point is InterceptionPoint.PRE_MODEL_CALL
        model_id = text_value(getattr(context.llm, "model", context.llm))
        request_id = self._call_id(binding, "model", model_id, pre=pre)
        if pre:
            self._guard_model_iteration(context, binding)
            messages = messages_value(context.messages)
            target = self._emit(
                binding,
                lambda builder: builder.pre_model_call(
                    model_id=model_id, messages=messages, request_id=request_id
                ),
                messages_value,
            )
            transformed = messages_value(target)
            # The runtime guard above validates the pinned CrewAI TypedDict shape.
            context.messages[:] = cast("list[LLMMessage]", transformed)
            return None
        original = text_value(context.response)
        task: object = context.task
        structured = bool(
            (
                isinstance(task, Task)
                and (task.output_json or task.output_pydantic or task.response_model)
            )
            or getattr(context.llm, "response_format", None)
        )

        def response_target(value: object) -> str:
            response = envelope(value, {"content", "tool_calls", "finish_reason"})
            if response["tool_calls"] != [] or response["finish_reason"] != "stop":
                raise ValueError("Structured model response transforms are unsupported")
            result = text_value(response["content"])
            if result != original and structured:
                raise ValueError("Structured model output transforms are unsupported")
            return result

        target = self._emit(
            binding,
            lambda builder: builder.post_model_call(
                model_id=model_id,
                content=original,
                tool_calls=[],
                finish_reason="stop",
                request_id=request_id,
            ),
            response_target,
        )
        return text_value(target)

    def _guard_model_iteration(self, context: LLMCallHookContext, binding: int) -> None:
        executor = context.executor
        agent = next(agent for agent in self._agents if id(agent) == binding)
        limit = self._iteration_limits[binding]
        if (
            type(executor) not in (AgentExecutor, CrewAgentExecutor)
            or not isinstance(executor, (AgentExecutor, CrewAgentExecutor))
            or executor.agent is not agent
            or executor is not agent.agent_executor
            or executor.max_iter != limit
            or agent.max_iter != limit
            or type(context.iterations) is not int
            or context.iterations < 0
            or context.iterations != executor.iterations
        ):
            raise ValueError("Unexpected model iteration context")
        # Both pinned runners call the model directly after exhausting max_iter.
        # Reserve the last normal iteration so that uncovered call is unreachable.
        if context.iterations >= limit - 1:
            raise ValueError("Reserved final iteration cannot execute an ungoverned fallback")

    def _tool(self, point: InterceptionPoint, context: ToolCallHookContext, binding: int) -> object:
        pre = point is InterceptionPoint.PRE_TOOL_CALL
        name = text_value(context.tool_name)
        args = object_value(context.tool_input)
        call_id = self._call_id(binding, "tool", name, pre=pre)
        if pre:
            agent = next(agent for agent in self._agents if id(agent) == binding)
            cache_enabled = agent.tools_handler.cache is not None
            legacy_empty = not args and type(agent.agent_executor) is CrewAgentExecutor

            def args_target(value: object) -> dict[str, JSON]:
                target = object_value(value)
                if target != args and cache_enabled:
                    raise ValueError("Tool argument transforms require disabled tool caching")
                if target != args and legacy_empty:
                    raise ValueError("Legacy executor cannot apply empty-argument transforms")
                return target

            target = self._emit(
                binding,
                lambda builder: builder.pre_tool_call(call_id=call_id, name=name, args=args),
                args_target,
            )
            transformed: dict[str, JSON] = object_value(target)
            context.tool_input.clear()
            context.tool_input.update(transformed)
            return None
        value = text_value(context.tool_result)
        target = self._emit(
            binding,
            lambda builder: builder.post_tool_call(
                call_id=call_id, name=name, args=args, value=value
            ),
            text_value,
        )
        return text_value(target)
