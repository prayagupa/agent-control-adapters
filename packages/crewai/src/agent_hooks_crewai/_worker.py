"""One bounded, private policy execution lane per adapter."""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import uuid4

from agent_hooks import (
    AgentContext,
    AgentContextBuilder,
    ApprovalResolver,
    CompositionConfig,
    InterceptionEmitter,
    InterceptionRecord,
    Interceptor,
)

from ._json import normalize, object_value
from .errors import AdapterBlocked, AdapterCleanupError

logger = logging.getLogger(__name__)
ContextFactory = Callable[[AgentContextBuilder], AgentContext]


@dataclass(frozen=True)
class Binding:
    identity: str
    interceptors: tuple[Interceptor, ...]


@dataclass
class _Job:
    binding: int
    factory: ContextFactory
    validate: Callable[[object], object]
    audit: bool
    done: threading.Event = field(default_factory=threading.Event)
    target: object = None
    error: BaseException | None = None


class Worker:
    """Serialize SDK calls with one active job and at most eight waiting callers."""

    def __init__(
        self,
        bindings: dict[int, Binding],
        resolver: ApprovalResolver | None,
        timeout: float,
        emit_timeout: float,
        max_records: int,
    ) -> None:
        self._bindings = bindings
        self._trace_id = bindings[0].identity.replace("-", "")
        self._resolver = resolver
        self._timeout = timeout
        self._deadline = emit_timeout
        self._condition = threading.Condition()
        self._job: _Job | None = None
        self._callers: set[int] = set()
        self._stopping = False
        self._poisoned = False
        self._denied = False
        self._records: deque[InterceptionRecord] = deque(maxlen=max_records)
        self._dropped = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[object] | None = None
        self._cleanup_failed = False
        self._thread = threading.Thread(
            target=self._run, name=f"agent-hooks-{uuid4()}", daemon=True
        )
        self._thread.start()

    @property
    def records(self) -> tuple[InterceptionRecord, ...]:
        with self._condition:
            return copy.deepcopy(tuple(self._records))

    @property
    def records_dropped(self) -> int:
        with self._condition:
            return self._dropped

    @property
    def available(self) -> bool:
        with self._condition:
            return not (self._poisoned or self._stopping)

    @property
    def denied(self) -> bool:
        with self._condition:
            return self._denied or self._poisoned

    def deny(self) -> None:
        with self._condition:
            self._denied = True
            self._condition.notify_all()

    def _poison(self) -> None:
        self._poisoned = True
        self._condition.notify_all()

    def emit(
        self,
        binding: int,
        factory: ContextFactory,
        validate: Callable[[object], object] = normalize,
        *,
        audit: bool = False,
    ) -> object:
        deadline = time.monotonic() + self._deadline
        caller = threading.get_ident()
        with self._condition:
            if self._denied and not audit:
                raise AdapterBlocked()
            if (
                self._stopping
                or self._poisoned
                or caller == self._thread.ident
                or caller in self._callers
                or len(self._callers) >= 9
            ):
                self._poison()
                raise AdapterBlocked()
            self._callers.add(caller)
            self._condition.notify_all()
            try:
                while self._job is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or self._poisoned or self._stopping:
                        self._poison()
                        raise AdapterBlocked()
                    self._condition.wait(remaining)
                if self._poisoned or self._stopping or (self._denied and not audit):
                    raise AdapterBlocked()
                job = _Job(binding, factory, validate, audit)
                self._job = job
                self._condition.notify_all()
            except BaseException:
                self._callers.remove(caller)
                raise
        try:
            if not job.done.wait(max(0, deadline - time.monotonic())):
                with self._condition:
                    self._poison()
                    loop = self._loop
                    if loop is not None and not loop.is_closed():
                        loop.call_soon_threadsafe(self._cancel_job, job)
                raise AdapterBlocked()
            with self._condition:
                if self._poisoned or time.monotonic() >= deadline:
                    self._poison()
                    raise AdapterBlocked()
                if self._denied and not audit:
                    raise AdapterBlocked()
            if job.error is not None:
                raise AdapterBlocked() from job.error
            return job.target
        finally:
            with self._condition:
                self._callers.remove(caller)
                self._condition.notify_all()

    def _cancel_job(self, job: _Job) -> None:
        if self._job is job and self._task is not None:
            self._task.cancel()

    async def _emit(
        self, job: _Job, builder: AgentContextBuilder, emitter: InterceptionEmitter
    ) -> object:
        # AgentContext and EmitOutcome.target are the published SDK's Any boundary.
        # Factories supply detached validated content; the host validates write-back.
        try:
            context = object_value(job.factory(builder))
            if job.audit:
                # Shutdown is a notification, not permission for a future action.
                # Preserve its actual verdict, but never apply its target or seek approval.
                await emitter.emit_unchecked(context)
                return None
            outcome = await emitter.emit(context)
            return job.validate(outcome.target)
        finally:
            records = emitter.take_records()
            with self._condition:
                for record in records:
                    if len(self._records) == self._records.maxlen:
                        self._dropped += 1
                    self._records.append(record)

    def _loop_error(self, loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        self._cleanup_failed = True
        self.deny()
        error = context.get("exception")
        logger.error(
            "adapter_loop_task_failed",
            extra={"trace_id": self._trace_id, "error_type": type(error).__name__},
        )

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(self._loop_error)
        states: dict[int, tuple[AgentContextBuilder, InterceptionEmitter]] = {}
        with self._condition:
            self._loop = loop
        try:
            for key, binding in self._bindings.items():
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
                        agent_id=binding.identity, framework="crewai", session_id=str(uuid4())
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
                        if self._poisoned:
                            raise AdapterBlocked()
                    builder, emitter = states[job.binding]
                    self._task = loop.create_task(self._emit(job, builder, emitter))
                    job.target = loop.run_until_complete(self._task)
                except BaseException as error:
                    job.error = error
                    if not job.audit:
                        self.deny()
                finally:
                    self._task = None
                    with self._condition:
                        self._job = None
                        job.done.set()
                        self._condition.notify_all()
        except BaseException as error:
            logger.error(
                "adapter_worker_failed",
                extra={"trace_id": self._trace_id, "error_type": type(error).__name__},
            )
            with self._condition:
                self._poison()
                if self._job is not None:
                    self._job.error = error
                    self._job.done.set()
                    self._job = None
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    results = loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                    if any(
                        isinstance(result, BaseException)
                        and not isinstance(result, asyncio.CancelledError)
                        for result in results
                    ):
                        self._cleanup_failed = True
                        logger.error(
                            "adapter_background_task_cleanup_failed",
                            extra={"trace_id": self._trace_id},
                        )
                loop.run_until_complete(loop.shutdown_asyncgens())
            except BaseException as error:
                self._cleanup_failed = True
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
                    self._condition.notify_all()

    def close(self) -> None:
        """Join cooperatively; explicitly report retained nonpreemptible work."""
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=min(1.0, self._deadline))
        if self._thread.is_alive() or self._cleanup_failed:
            logger.error("adapter_cleanup_incomplete", extra={"trace_id": self._trace_id})
            raise AdapterCleanupError()
