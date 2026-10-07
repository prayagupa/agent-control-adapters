"""Shared execution admission, effect budgets and bounded native-task ownership."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar, cast
from uuid import uuid4

from agent_hooks import AgentContext, EmitOutcome, InterceptionRecord
from agent_hooks.exceptions import InterceptionBlocked

from ._worker import Binding, ContextFactory, Worker
from .bindings import PreparedControls
from .contracts import RunRequest
from .errors import AdapterBlocked, AdapterCleanupError
from .json import normalize

_NATIVE_RUN_TIMEOUT = 120.0
_NATIVE_CLEANUP_TIMEOUT = 2.0
_MAX_MODEL_CALLS = 10
_MAX_PENDING = 64
_MAX_EVENTS = 256
_GRAPH_RECURSION_LIMIT = 30
_NATIVE_TOOL_CONCURRENCY = 1
_T = TypeVar("_T")
_LOG = logging.getLogger(__name__)


def prompt_value(request: RunRequest | None) -> str:
    """Accept only a bounded plain prompt, never caller-owned runtime metadata."""
    if not isinstance(request, RunRequest):
        raise ValueError("RunRequest({'prompt': nonempty_text}) is required")
    inputs = request.inputs
    if inputs is None or set(inputs) != {"prompt"}:
        raise ValueError("Exactly one prompt field is required")
    prompt = inputs["prompt"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 65_536:
        raise ValueError("prompt must contain between 1 and 65536 nonempty characters")
    return prompt


class RunAdmission:
    """Atomically consume one execution attempt, including invalid admitted requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._used = False

    def admit(self, request: RunRequest | None) -> str:
        """Capture a prompt once; a failed validation still consumes this instance."""
        with self._lock:
            if self._used:
                raise RuntimeError("Adapter instances are single-use")
            self._used = True
        return prompt_value(request)


@dataclass(frozen=True)
class NativeExecutionLimits:
    """Fixed first-profile native recursion and tool-concurrency configuration."""

    recursion: int = _GRAPH_RECURSION_LIMIT
    tool_concurrency: int = _NATIVE_TOOL_CONCURRENCY


def validate_topology(*, actors: int, nodes: int, edges: int) -> None:
    """Reject oversized native membership and graphs before policy I/O or effects."""
    if any(type(value) is not int for value in (actors, nodes, edges)):
        raise TypeError("Native topology counts must be integers")
    if not 1 <= actors <= 16 or not 1 <= nodes <= 16 or not 0 <= edges <= 64:
        raise ValueError("Native topology exceeds 16 actors/nodes or 64 edges")


@dataclass(frozen=True)
class BudgetSnapshot:
    """Bounded counters, not an application-visible authority token."""

    model_calls: int
    pending: int
    native_events: int


class RunBudget:
    """Own the atomic check-and-update for model, tool, correlation and event limits."""

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self._lock = threading.RLock()
        self._error: AdapterBlocked | None = None
        self._model_calls = 0
        self._events = 0
        self._pending: dict[tuple[int, str], Literal["model", "ready_tool", "active_tool"]] = {}

    def _block(self, message: str) -> None:
        if self._error is None:
            self._error = AdapterBlocked(message)
            _LOG.warning(
                "adapter_budget_denied",
                extra={"trace_id": self.trace_id, "error_category": "resource_or_correlation"},
            )
        raise self._error

    def _ensure(self) -> None:
        if self._error is not None:
            raise self._error

    def _key(self, actor: int, correlation: str) -> tuple[int, str]:
        if type(actor) is not int or not 0 <= actor < 16:
            self._block("Unknown native actor binding")
        if not isinstance(cast("object", correlation), str) or not 1 <= len(correlation) <= 128:
            self._block("Invalid native correlation")
        return actor, correlation

    def _claim(
        self, key: tuple[int, str], state: Literal["model", "ready_tool", "active_tool"]
    ) -> None:
        if key in self._pending:
            self._block("Duplicate native correlation")
        if len(self._pending) >= _MAX_PENDING:
            self._block("Pending correlation limit exceeded")
        self._pending[key] = state

    def event(self) -> None:
        """Admit one non-audit native event; the 257th poisons the entire run."""
        with self._lock:
            self._ensure()
            if self._events >= _MAX_EVENTS:
                self._block("Native event limit exceeded")
            self._events += 1

    def before_model(self, actor: int, correlation: str) -> None:
        """Reserve a model call before its policy/model effect."""
        with self._lock:
            self._ensure()
            key = self._key(actor, correlation)
            if self._model_calls >= _MAX_MODEL_CALLS:
                self._block("Model call limit exceeded")
            self._claim(key, "model")
            self._model_calls += 1

    def after_model(self, actor: int, correlation: str, tool_call_ids: Sequence[str] = ()) -> None:
        """Complete an actual model call and allow at most one pending native tool."""
        with self._lock:
            self._ensure()
            key = self._key(actor, correlation)
            if self._pending.get(key) != "model":
                self._block("Unknown or duplicate model completion")
            if isinstance(tool_call_ids, str | bytes) or len(tool_call_ids) > 1:
                self._block("Only one tool call per model output is supported")
            tool_key = self._key(actor, tool_call_ids[0]) if tool_call_ids else None
            if tool_key is not None and tool_key in self._pending:
                self._block("Duplicate native tool correlation")
            del self._pending[key]
            if tool_key is not None:
                self._claim(tool_key, "ready_tool")

    def before_tool(self, actor: int, correlation: str) -> None:
        """Authorize correlation once before any native tool body is entered."""
        with self._lock:
            self._ensure()
            key = self._key(actor, correlation)
            if self._pending.get(key) != "ready_tool":
                self._block("Unknown or repeated native tool call")
            self._pending[key] = "active_tool"

    def after_tool(self, actor: int, correlation: str) -> None:
        """Remove one actual completed tool call; fabricated/duplicate posts deny."""
        with self._lock:
            self._ensure()
            key = self._key(actor, correlation)
            if self._pending.get(key) != "active_tool":
                self._block("Unknown or duplicate tool completion")
            del self._pending[key]

    @property
    def snapshot(self) -> BudgetSnapshot:
        """Read a consistent immutable counter snapshot."""
        with self._lock:
            return BudgetSnapshot(self._model_calls, len(self._pending), self._events)

    def close(self) -> None:
        """Release terminal correlation state without clearing sticky denial."""
        with self._lock:
            self._pending.clear()


class NativeTaskOwner:
    """Own exactly one native task and contain cancellation-resistant work explicitly."""

    def __init__(
        self, on_failure: Callable[[BaseException], None], *, trace_id: str | None = None
    ) -> None:
        self.trace_id = trace_id or str(uuid4())
        self._on_failure = on_failure
        self._used = False
        self._native_task: asyncio.Task[object] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._cleanup_error: AdapterCleanupError | None = None
        self._late_error: BaseException | None = None

    @property
    def cleanup_error(self) -> AdapterCleanupError | None:
        """Report an owned task that did not exit within the fixed drain budget."""
        return self._cleanup_error

    @property
    def pending(self) -> bool:
        """Keep surviving work strongly owned until its real completion."""
        return self._native_task is not None and not self._native_task.done()

    def _observe(self, task: asyncio.Task[object]) -> None:
        if not task.cancelled():
            self._late_error = task.exception()
        if self._native_task is task:
            self._native_task = None

    async def _drain(self, task: asyncio.Task[object]) -> None:
        if not task.done():
            task.cancel()
            _, pending = await asyncio.wait((task,), timeout=_NATIVE_CLEANUP_TIMEOUT)
            if pending:
                self._cleanup_error = AdapterCleanupError(
                    "Native execution did not stop within the cleanup deadline"
                )
                _LOG.error(
                    "adapter_native_cleanup_incomplete",
                    extra={"trace_id": self.trace_id, "error_category": "native_task"},
                )

    async def _finish_drain(self, task: asyncio.Task[object]) -> None:
        cleanup = asyncio.create_task(self._drain(task), name="agent-controls-native-drain")
        self._cleanup_task = cleanup
        deadline = asyncio.get_running_loop().time() + _NATIVE_CLEANUP_TIMEOUT
        while not cleanup.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                self._cleanup_error = AdapterCleanupError(
                    "Native execution did not stop within the cleanup deadline"
                )
                return
            try:
                await asyncio.wait((cleanup,), timeout=remaining)
            except asyncio.CancelledError:
                # A second cancellation must neither orphan the drain nor replace the primary.
                continue
        cleanup.result()
        self._cleanup_task = None

    async def run(self, execute: Callable[[], Awaitable[_T]]) -> _T:
        """Execute on this loop; timeout/cancellation poisons before bounded child drain."""
        if self._used:
            raise RuntimeError("Native task owners are single-use")
        self._used = True

        async def invoke() -> _T:
            return await execute()

        task = asyncio.create_task(invoke(), name="agent-controls-native-run")
        self._native_task = task
        task.add_done_callback(self._observe)
        try:
            done, _ = await asyncio.wait((task,), timeout=_NATIVE_RUN_TIMEOUT)
            if not done:
                raise AdapterBlocked("Native execution deadline exceeded")
            return task.result()
        except BaseException as primary:
            self._on_failure(primary)
            await self._finish_drain(task)
            raise


class PolicySession:
    """Own one policy lane, actual actor membership and sticky native execution scope."""

    def __init__(self, framework: str, controls: PreparedControls) -> None:
        self.framework = framework
        self.controls = controls
        self.trace_id = str(uuid4())
        self.budget = RunBudget(self.trace_id)
        self._lock = threading.RLock()
        self._active = False
        self._started = False
        self._worker: Worker | None = None
        self._error: BaseException | None = None
        self._policy_error = False
        self._cleanup_error: AdapterCleanupError | None = None

    def begin(self) -> None:
        """Start the policy worker only after run admission and native revalidation."""
        with self._lock:
            if self._started:
                raise RuntimeError("Policy sessions are single-use")
            self._started = True
            self._worker = Worker(
                {
                    index: Binding(actor.identity, actor.interceptors, actor.session_id)
                    for index, actor in enumerate(self.controls.actors)
                },
                self.controls.resolver,
                self.controls.timeout,
                self.controls.emit_timeout,
                self.controls.max_records,
                framework=self.framework,
                trace_id=self.trace_id,
            )
            self._active = True

    @property
    def records(self) -> tuple[InterceptionRecord, ...]:
        """Return only actual bounded SDK audit, including records after failure."""
        worker = self._worker
        return worker.records if worker is not None else ()

    @property
    def records_dropped(self) -> int:
        """Expose exact worker audit eviction, not an estimated count."""
        worker = self._worker
        return worker.records_dropped if worker is not None else 0

    @property
    def cleanup_error(self) -> AdapterCleanupError | None:
        """Expose policy-lane cleanup separately from a native primary exception."""
        return self._cleanup_error

    @property
    def last_error(self) -> BaseException | None:
        """Retain original errors privately for the outer application boundary."""
        worker = self._worker
        return (
            worker.last_error
            if worker is not None and worker.last_error is not None
            else self._error
        )

    @property
    def policy_failed(self) -> bool:
        """Distinguish control failures from unrelated native SDK exceptions."""
        with self._lock:
            return self._policy_error

    def binding(self, native: object) -> int:
        """Resolve a strong actual-object membership match, never a model-supplied ID."""
        for index, actor in enumerate(self.controls.actors):
            if native is actor.native:
                return index
        error = AdapterBlocked("Unknown native actor")
        self.fail(error, policy=True)
        raise error

    def fail(self, error: BaseException, *, policy: bool = False, poison: bool = True) -> None:
        """Atomically poison before drain; do not turn ordinary native errors into denies."""
        with self._lock:
            if self._error is None:
                self._error = error
            self._policy_error |= policy
            worker = self._worker
        if worker is not None and poison:
            worker.poison(error)

    def _emission_failure(self, worker: Worker, error: BaseException) -> None:
        self.fail(
            error,
            policy=not isinstance(error, asyncio.CancelledError),
            poison=isinstance(error, asyncio.CancelledError)
            or not isinstance(worker.last_error, InterceptionBlocked),
        )

    def _lane(self, binding: int, *, audit: bool = False) -> Worker:
        with self._lock:
            worker = self._worker
            if (
                worker is None
                or not self._active
                or type(binding) is not int
                or not 0 <= binding < len(self.controls.actors)
                or (self._error is not None and not audit)
            ):
                raise AdapterBlocked("Governed execution scope is not active")
            return worker

    def _event(self, *, audit: bool = False) -> None:
        if not audit:
            try:
                self.budget.event()
            except AdapterBlocked as error:
                self.fail(error, policy=True)
                raise

    def emit(
        self,
        binding: int,
        factory: ContextFactory,
        validate: Callable[[object], object] = normalize,
        *,
        audit: bool = False,
    ) -> object:
        """Emit on the bounded lane from a native synchronous callback."""
        worker = self._lane(binding, audit=audit)
        self._event(audit=audit)
        try:
            result = worker.emit(binding, factory, validate, audit=audit)
        except BaseException as error:
            self._emission_failure(worker, error)
            raise
        self._lane(binding, audit=audit)
        return result

    async def aemit(
        self,
        binding: int,
        factory: ContextFactory,
        validate: Callable[[object], object] = normalize,
        *,
        audit: bool = False,
    ) -> object:
        """Emit without blocking the native event loop or using its default executor."""
        worker = self._lane(binding, audit=audit)
        self._event(audit=audit)
        try:
            result = await worker.aemit(binding, factory, validate, audit=audit)
        except BaseException as error:
            self._emission_failure(worker, error)
            raise
        self._lane(binding, audit=audit)
        return result

    async def aemit_outcome(self, binding: int, context: AgentContext) -> EmitOutcome:
        """Delegate an upstream-owned context without allocating a second SDK sequence."""
        worker = self._lane(binding)
        self._event()
        try:
            result = await worker.aemit_outcome(binding, context)
        except BaseException as error:
            self._emission_failure(worker, error)
            raise
        self._lane(binding)
        return result

    async def aemit_unchecked(self, binding: int, context: AgentContext) -> InterceptionRecord:
        """Record upstream shutdown audit even after a sticky policy denial."""
        worker = self._lane(binding, audit=True)
        return await worker.aemit_unchecked(binding, context)

    def close(self, primary: BaseException | None = None) -> None:
        """Close only the SDK lane, preserving an earlier primary exception."""
        with self._lock:
            self._active = False
            worker = self._worker
        self.budget.close()
        if worker is not None:
            try:
                worker.close()
            except AdapterCleanupError as error:
                self._cleanup_error = error
                if primary is None:
                    raise
                primary.add_note("Governance worker cleanup incomplete.")

    async def aclose(self, primary: BaseException | None = None) -> None:
        """Request bounded policy cleanup without joining a thread on the native loop."""
        with self._lock:
            self._active = False
            worker = self._worker
        self.budget.close()
        if worker is not None:
            try:
                await worker.aclose()
            except AdapterCleanupError as error:
                self._cleanup_error = error
                if primary is None:
                    raise
                primary.add_note("Governance worker cleanup incomplete.")
