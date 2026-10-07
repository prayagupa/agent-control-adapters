"""A bounded, single-threaded SDK policy lane with sync and async submission."""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from agent_hooks import (
    AgentContext,
    AgentContextBuilder,
    ApprovalResolver,
    CompositionConfig,
    EmitOutcome,
    InterceptionBlocked,
    InterceptionEmitter,
    InterceptionPoint,
    InterceptionRecord,
    Interceptor,
)

from .errors import AdapterBlocked, AdapterCleanupError
from .json import JSON, normalize, object_value

logger = logging.getLogger(__name__)
ContextFactory = Callable[[AgentContextBuilder], AgentContext]
_CLEANUP_TIMEOUT = 1.0
_MAX_CALLERS = 9


@dataclass(frozen=True)
class Binding:
    """Immutable policy identity; an optional fixed session supports external builders."""

    identity: str
    interceptors: tuple[Interceptor, ...]
    session_id: str | None = None


@dataclass(eq=False)
class _Notification:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[None]
    pending: bool = False
    closed: bool = False


@dataclass(eq=False)
class _Job:
    binding: int
    factory: ContextFactory | None
    validate: Callable[[object], object]
    audit: bool
    caller: object
    mode: Literal["target", "outcome", "unchecked"] = "target"
    context: dict[str, JSON] | None = None
    notification: _Notification | None = None
    done: threading.Event = field(default_factory=threading.Event)
    target: object = None
    error: BaseException | None = None
    started: bool = False
    cancel_requested: bool = False


class Worker:
    """Run actual SDK emissions serially, admitting one active and eight queued calls.

    Factories and validators execute only on the owned SDK thread. Async callers
    wait on their own loop without executors. Cancellation or a wall timeout
    poisons active work; nonpreemptible work retains its slot until actual exit.
    Cleanup is bounded for callers and reports a retained thread explicitly.
    """

    def __init__(
        self,
        bindings: dict[int, Binding],
        resolver: ApprovalResolver | None,
        timeout: float,
        emit_timeout: float,
        max_records: int,
        *,
        framework: str = "crewai",
        blocked_type: type[AdapterBlocked] = AdapterBlocked,
        cleanup_type: type[AdapterCleanupError] = AdapterCleanupError,
        trace_id: str | None = None,
    ) -> None:
        if trace_id is not None and (
            type(trace_id) is not str
            or not 1 <= len(trace_id) <= 128
            or trace_id.strip() != trace_id
            or any(ord(character) < 32 for character in trace_id)
        ):
            raise ValueError("A supplied trace ID must be bounded, nonblank text")
        self._bindings = {
            key: Binding(
                value.identity,
                tuple(value.interceptors),
                value.session_id if value.session_id is not None else str(uuid4()),
            )
            for key, value in bindings.items()
        }
        self._owns_trace = trace_id is not None
        self._trace_id = (
            trace_id if trace_id is not None else self._bindings[0].identity.replace("-", "")
        )
        self._framework = framework
        self._blocked_type = blocked_type
        self._cleanup_type = cleanup_type
        self._resolver = resolver
        self._timeout = timeout
        self._deadline = emit_timeout
        self._condition = threading.Condition()
        self._job: _Job | None = None
        self._queue: deque[_Job] = deque()
        self._callers: set[object] = set()
        self._closers: set[_Notification] = set()
        self._close_tasks: set[asyncio.Task[None]] = set()
        self._stopping = False
        self._poisoned = False
        self._denied = False
        self._last_error: BaseException | None = None
        self._records: deque[InterceptionRecord] = deque(maxlen=max_records)
        self._dropped = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[object] | None = None
        self._cleanup_failed = False
        self._exited = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"agent-hooks-{uuid4()}", daemon=True
        )
        self._thread.start()

    @property
    def records(self) -> tuple[InterceptionRecord, ...]:
        """Return detached actual SDK records in retention order."""
        with self._condition:
            return copy.deepcopy(tuple(self._records))

    @property
    def records_dropped(self) -> int:
        """Return the exact number of evicted actual records."""
        with self._condition:
            return self._dropped

    @property
    def available(self) -> bool:
        """Whether the lane can accept work, including shutdown audit."""
        with self._condition:
            return not (self._poisoned or self._stopping)

    @property
    def denied(self) -> bool:
        """Whether subsequent governed actions must stop."""
        with self._condition:
            return self._denied or self._poisoned

    @property
    def last_error(self) -> BaseException | None:
        """Original private failure for outer application-boundary cause mapping."""
        with self._condition:
            return self._last_error

    def deny(self) -> None:
        """Set sticky denial before notifying any submitted caller."""
        with self._condition:
            self._denied = True
            self._reject_queue_locked()
            self._condition.notify_all()

    def _remember_error_locked(self, error: BaseException) -> None:
        if self._last_error is None:
            self._last_error = error

    def _wake(self, notification: _Notification) -> None:
        with self._condition:
            notification.pending = False
            if not notification.future.done():
                notification.future.set_result(None)

    def _notify_locked(self, notification: _Notification) -> None:
        if notification.closed:
            return
        if notification.pending and not notification.loop.is_closed():
            return
        notification.pending = True
        try:
            notification.loop.call_soon_threadsafe(self._wake, notification)
        except RuntimeError as error:
            notification.pending = False
            notification.closed = True
            logger.error(
                "adapter_caller_loop_closed",
                extra={"trace_id": self._trace_id, "error_type": type(error).__name__},
            )
            self._poison_locked(error)

    def _notify_job_locked(self, job: _Job) -> None:
        if job.notification is not None:
            self._notify_locked(job.notification)
        self._condition.notify_all()

    def _reject_queue_locked(self) -> None:
        queued = tuple(self._queue)
        self._queue.clear()
        for job in queued:
            if not (self._poisoned or self._stopping) and job.audit:
                self._queue.append(job)
                continue
            self._callers.discard(job.caller)
            job.error = self._last_error if self._last_error is not None else self._blocked_type()
            job.done.set()
            self._notify_job_locked(job)

    def _poison_locked(self, error: BaseException | None = None) -> None:
        if error is not None:
            self._remember_error_locked(error)
        if not self._poisoned:
            logger.error(
                "adapter_worker_poisoned",
                extra={
                    "trace_id": self._trace_id,
                    "error_type": type(error).__name__ if error is not None else "AdapterBlocked",
                },
            )
        self._poisoned = True
        self._reject_queue_locked()
        if self._job is not None:
            self._notify_job_locked(self._job)
            self._schedule_cancel_locked(self._job)
        self._condition.notify_all()

    def poison(self, error: BaseException | None = None) -> None:
        """Synchronously fail closed, preserving the first error and actual active lease."""
        with self._condition:
            self._poison_locked(error)

    def _poison(self) -> None:
        self.poison()

    def _claim_locked(self) -> None:
        if self._job is None and self._queue and not (self._poisoned or self._stopping):
            self._job = self._queue.popleft()
            self._job.started = True
            self._condition.notify_all()

    def _admit_locked(self, job: _Job) -> None:
        if self._denied and not job.audit:
            raise self._blocked_type() from self._last_error
        if (
            self._stopping
            or self._poisoned
            or threading.get_ident() == self._thread.ident
            or job.caller in self._callers
            or len(self._callers) >= _MAX_CALLERS
        ):
            self._poison_locked()
            raise self._blocked_type() from self._last_error
        if job.binding not in self._bindings:
            error = ValueError("Unknown policy binding.")
            self._remember_error_locked(error)
            self._denied = True
            self._reject_queue_locked()
            raise self._blocked_type() from error
        self._callers.add(job.caller)
        self._queue.append(job)
        self._claim_locked()
        self._condition.notify_all()

    def _result_locked(self, job: _Job, deadline: float) -> object:
        if time.monotonic() >= deadline:
            self._poison_locked(TimeoutError("Governance emission wall deadline exceeded."))
        if self._poisoned or self._stopping:
            error = job.error if job.error is not None else self._last_error
            raise self._blocked_type() from error
        if job.error is not None:
            raise self._blocked_type() from job.error
        if self._denied and not job.audit and job.mode != "unchecked":
            raise self._blocked_type() from self._last_error
        return job.target

    def emit(
        self,
        binding: int,
        factory: ContextFactory,
        validate: Callable[[object], object] = normalize,
        *,
        audit: bool = False,
    ) -> object:
        """Submit synchronously; raise configured blocked errors on denial/failure."""
        deadline = time.monotonic() + self._deadline
        job = _Job(binding, factory, validate, audit, ("sync", threading.get_ident()))
        with self._condition:
            try:
                self._admit_locked(job)
                while not job.done.is_set() and not (self._poisoned or self._stopping):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._poison_locked(
                            TimeoutError("Governance emission wall deadline exceeded.")
                        )
                        break
                    self._condition.wait(remaining)
                return self._result_locked(job, deadline)
            except BaseException:
                if job in self._queue:
                    self._queue.remove(job)
                    self._callers.discard(job.caller)
                elif self._job is job:
                    self._poison_locked()
                raise

    async def aemit(
        self,
        binding: int,
        factory: ContextFactory,
        validate: Callable[[object], object] = normalize,
        *,
        audit: bool = False,
    ) -> object:
        """Submit without blocking the caller loop; preserve synchronous semantics."""
        return await self._asubmit(binding, factory, validate, audit)

    async def aemit_outcome(self, binding: int, context: AgentContext) -> EmitOutcome:
        """Emit a detached external context without rebuilding it or advancing sequence."""
        detached = self._detach(context)
        result = await self._asubmit(
            binding, None, normalize, False, mode="outcome", context=detached
        )
        if not isinstance(result, EmitOutcome):
            raise TypeError("SDK did not return an EmitOutcome.")
        return result

    async def aemit_unchecked(self, binding: int, context: AgentContext) -> InterceptionRecord:
        """Return the actual record; denied actions become sticky, shutdown is audit-only."""
        detached = self._detach(context)
        result = await self._asubmit(
            binding,
            None,
            normalize,
            detached.get("interception_point") == InterceptionPoint.AGENT_SHUTDOWN.value,
            mode="unchecked",
            context=detached,
        )
        if not isinstance(result, InterceptionRecord):
            raise TypeError("SDK did not return an InterceptionRecord.")
        return result

    def _detach(self, context: object) -> dict[str, JSON]:
        try:
            return object_value(context)
        except ValueError as error:
            with self._condition:
                self._remember_error_locked(error)
                self._denied = True
                self._reject_queue_locked()
            logger.warning(
                "adapter_context_invalid",
                extra={"trace_id": self._trace_id, "error_type": type(error).__name__},
            )
            raise self._blocked_type() from error

    async def _asubmit(
        self,
        binding: int,
        factory: ContextFactory | None,
        validate: Callable[[object], object],
        audit: bool,
        *,
        mode: Literal["target", "outcome", "unchecked"] = "target",
        context: dict[str, JSON] | None = None,
    ) -> object:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        notification = _Notification(loop, loop.create_future())
        job = _Job(
            binding, factory, validate, audit, ("async", loop, task), mode, context, notification
        )
        deadline = time.monotonic() + self._deadline
        try:
            with self._condition:
                self._admit_locked(job)
            await asyncio.wait({notification.future}, timeout=max(0, deadline - time.monotonic()))
            with self._condition:
                return self._result_locked(job, deadline)
        except asyncio.CancelledError as error:
            with self._condition:
                if job.started:
                    self._poison_locked(error)
                elif job in self._queue:
                    self._queue.remove(job)
                    self._callers.discard(job.caller)
                    self._condition.notify_all()
            raise
        finally:
            if not notification.future.done():
                notification.future.cancel()

    def _schedule_cancel_locked(self, job: _Job) -> None:
        if job.cancel_requested or self._loop is None:
            return
        job.cancel_requested = True
        try:
            self._loop.call_soon_threadsafe(self._cancel_job, job)
        except RuntimeError as error:
            self._remember_error_locked(error)
            self._cleanup_failed = True
            logger.error(
                "adapter_sdk_cancel_failed",
                extra={"trace_id": self._trace_id, "error_type": type(error).__name__},
            )

    def _cancel_job(self, job: _Job) -> None:
        with self._condition:
            if self._job is job and self._task is not None:
                self._task.cancel()

    def _context(self, job: _Job, builder: AgentContextBuilder) -> dict[str, JSON]:
        if job.context is None:
            if job.factory is None:
                raise ValueError("Missing context factory.")
            return object_value(job.factory(builder))
        context = job.context
        binding = self._bindings[job.binding]
        agent, session = context.get("agent"), context.get("session")
        sequence, point = context.get("sequence"), context.get("interception_point")
        if (
            not isinstance(agent, dict)
            or agent.get("id") != binding.identity
            or agent.get("framework") != self._framework
            or not isinstance(session, dict)
            or session.get("id") != binding.session_id
            or type(sequence) is not int
            or sequence < 0
            or not isinstance(point, str)
            or point not in {item.value for item in InterceptionPoint}
        ):
            raise ValueError("Context does not match trusted policy binding.")
        if self._owns_trace:
            trace = context.get("trace", {})
            if not isinstance(trace, dict):
                raise ValueError("Context trace must be a JSON object.")
            context["trace"] = {**trace, "trace_id": self._trace_id}
        return context

    async def _emit(
        self, job: _Job, builder: AgentContextBuilder, emitter: InterceptionEmitter
    ) -> object:
        try:
            context = self._context(job, builder)
            if job.mode == "unchecked" or job.audit:
                record = await emitter.emit_unchecked(context)
                if not record.proceeds and not job.audit:
                    with self._condition:
                        self._denied = True
                        self._remember_error_locked(InterceptionBlocked(record))
                return record if job.mode == "unchecked" else None
            outcome = await emitter.emit(context)
            target = normalize(outcome.target)
            if job.mode == "outcome":
                return outcome
            return job.validate(target)
        finally:
            records = emitter.take_records()
            with self._condition:
                for record in records:
                    if len(self._records) == self._records.maxlen:
                        self._dropped += 1
                    self._records.append(record)

    def _loop_error(self, loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        error = context.get("exception")
        with self._condition:
            self._cleanup_failed = True
            if isinstance(error, BaseException):
                self._remember_error_locked(error)
            self._poison_locked()
        logger.error(
            "adapter_loop_task_failed",
            extra={"trace_id": self._trace_id, "error_type": type(error).__name__},
        )

    def _finish_locked(self, job: _Job) -> None:
        self._job = None
        self._callers.discard(job.caller)
        job.done.set()
        if self._denied or self._poisoned or self._stopping:
            self._reject_queue_locked()
        self._notify_job_locked(job)
        self._claim_locked()

    async def _drain(self, pending: set[asyncio.Task[object]]) -> None:
        if not pending:
            return
        for task in pending:
            task.cancel()
        _, remaining = await asyncio.wait(pending, timeout=min(_CLEANUP_TIMEOUT, self._deadline))
        if remaining:
            with self._condition:
                self._cleanup_failed = True
            logger.error("adapter_background_cleanup_retained", extra={"trace_id": self._trace_id})
        # Keep the owned loop/thread alive until cancellation-resistant work really exits.
        results = await asyncio.gather(*pending, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                with self._condition:
                    self._cleanup_failed = True
                    self._remember_error_locked(result)
                logger.error(
                    "adapter_background_task_cleanup_failed",
                    extra={"trace_id": self._trace_id, "error_type": type(result).__name__},
                )

    async def _shutdown(self, loop: asyncio.AbstractEventLoop) -> None:
        current = asyncio.current_task()
        deadline = time.monotonic() + min(_CLEANUP_TIMEOUT, self._deadline)
        while True:
            pending = {task for task in asyncio.all_tasks(loop) if task is not current}
            if pending:
                await self._drain(pending)
                if time.monotonic() >= deadline:
                    with self._condition:
                        self._cleanup_failed = True
                    logger.error(
                        "adapter_background_cleanup_deadline", extra={"trace_id": self._trace_id}
                    )
                continue
            await loop.shutdown_asyncgens()
            if not any(task is not current for task in asyncio.all_tasks(loop)):
                break
        await loop.shutdown_default_executor()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(self._loop_error)
        states: dict[int, tuple[AgentContextBuilder, InterceptionEmitter]] = {}
        with self._condition:
            self._loop = loop
        try:
            for key, binding in self._bindings.items():
                if binding.session_id is None:
                    raise ValueError("Missing trusted policy session.")
                emitter = InterceptionEmitter(
                    resolver=self._resolver,
                    timeout=self._timeout,
                    composition=CompositionConfig.run_all(),
                )
                emitter.set_max_records(1)
                for interceptor in binding.interceptors:
                    emitter.register(interceptor)
                states[key] = (
                    AgentContextBuilder(
                        agent_id=binding.identity,
                        framework=self._framework,
                        session_id=binding.session_id,
                    ).with_l2(trace={"trace_id": self._trace_id}),
                    emitter,
                )
            while True:
                with self._condition:
                    while self._job is None and not self._stopping:
                        self._condition.wait()
                    if self._job is None:
                        break
                    job = self._job
                try:
                    with self._condition:
                        if self._poisoned or self._stopping:
                            raise self._blocked_type()
                        builder, emitter = states[job.binding]
                    task = loop.create_task(self._emit(job, builder, emitter))
                    with self._condition:
                        self._task = task
                    job.target = loop.run_until_complete(task)
                except BaseException as error:
                    with self._condition:
                        job.error = error
                        self._remember_error_locked(error)
                        if not job.audit:
                            self._denied = True
                    logger.warning(
                        "adapter_emit_failed",
                        extra={"trace_id": self._trace_id, "error_type": type(error).__name__},
                    )
                finally:
                    with self._condition:
                        self._task = None
                        self._finish_locked(job)
        except BaseException as error:
            logger.error(
                "adapter_worker_failed",
                extra={"trace_id": self._trace_id, "error_type": type(error).__name__},
            )
            with self._condition:
                self._poison_locked(error)
                if self._job is not None:
                    self._job.error = error
                    self._finish_locked(self._job)
        finally:
            try:
                loop.run_until_complete(self._shutdown(loop))
            except BaseException as error:
                with self._condition:
                    self._cleanup_failed = True
                    self._remember_error_locked(error)
                logger.error(
                    "adapter_loop_cleanup_failed",
                    extra={"trace_id": self._trace_id, "error_type": type(error).__name__},
                )
            finally:
                with self._condition:
                    loop.close()
                    self._loop = None
                    self._stopping = True
                    self._bindings.clear()
                    states.clear()
                    self._resolver = None
                    self._exited.set()
                    for notification in tuple(self._closers):
                        self._notify_locked(notification)
                    self._condition.notify_all()

    def _stop_locked(self) -> None:
        self._stopping = True
        self._reject_queue_locked()
        if self._job is not None:
            self._notify_job_locked(self._job)
            self._schedule_cancel_locked(self._job)
        self._condition.notify_all()

    def _check_cleanup(self) -> None:
        with self._condition:
            failed = self._thread.is_alive() or self._cleanup_failed
        if failed:
            logger.error("adapter_cleanup_incomplete", extra={"trace_id": self._trace_id})
            raise self._cleanup_type()

    def close(self) -> None:
        """Join cooperatively; report retained work or SDK cleanup failures."""
        with self._condition:
            self._stop_locked()
        if threading.get_ident() != self._thread.ident:
            self._thread.join(timeout=min(_CLEANUP_TIMEOUT, self._deadline))
        self._check_cleanup()

    async def aclose(self) -> None:
        """Stop and await owned thread exit without blocking or closing the caller loop."""
        loop = asyncio.get_running_loop()
        notification = _Notification(loop, loop.create_future())
        deadline = time.monotonic() + min(_CLEANUP_TIMEOUT, self._deadline)
        with self._condition:
            self._stop_locked()
            if len(self._closers) >= _MAX_CALLERS or len(self._close_tasks) >= _MAX_CALLERS:
                logger.error("adapter_close_capacity_exceeded", extra={"trace_id": self._trace_id})
                raise self._cleanup_type()
            self._closers.add(notification)
            if self._exited.is_set():
                self._notify_locked(notification)
            cleanup = loop.create_task(self._await_exit(notification, deadline))
            self._close_tasks.add(cleanup)
            cleanup.add_done_callback(self._close_finished)
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as primary:
            logger.warning("adapter_close_cancelled", extra={"trace_id": self._trace_id})
            try:
                await asyncio.wait({cleanup}, timeout=max(0, deadline - time.monotonic()))
            except asyncio.CancelledError:
                logger.warning("adapter_close_cancelled_again", extra={"trace_id": self._trace_id})
            raise primary

    def _close_finished(self, task: asyncio.Task[None]) -> None:
        with self._condition:
            self._close_tasks.discard(task)
        if task.cancelled():
            logger.error("adapter_close_monitor_cancelled", extra={"trace_id": self._trace_id})
            return
        error = task.exception()
        if error is not None:
            with self._condition:
                self._remember_error_locked(error)
            logger.error(
                "adapter_close_monitor_failed",
                extra={"trace_id": self._trace_id, "error_type": type(error).__name__},
            )

    async def _await_exit(self, notification: _Notification, deadline: float) -> None:
        try:
            await asyncio.wait({notification.future}, timeout=max(0, deadline - time.monotonic()))
            while self._exited.is_set() and self._thread.is_alive():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                tick: asyncio.Future[None] = notification.loop.create_future()
                timer = notification.loop.call_later(min(0.001, remaining), self._tick, tick)
                try:
                    await tick
                finally:
                    timer.cancel()
            if not self._thread.is_alive():
                self._thread.join(timeout=0)
            self._check_cleanup()
        finally:
            with self._condition:
                self._closers.discard(notification)
            if not notification.future.done():
                notification.future.cancel()

    @staticmethod
    def _tick(future: asyncio.Future[None]) -> None:
        if not future.done():
            future.set_result(None)
