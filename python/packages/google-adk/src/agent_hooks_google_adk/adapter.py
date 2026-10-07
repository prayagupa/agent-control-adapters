"""Single-use native Google ADK execution and owned ephemeral-resource cleanup."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from uuid import uuid4

from agent_hooks import InterceptionRecord
from agent_hooks_adapters import AdapterBlocked, AdapterCleanupError, RunRequest, RunResult
from agent_hooks_adapters.bindings import PreparedControls
from agent_hooks_adapters.runtime import NativeTaskOwner, PolicySession, RunAdmission
from google.adk.agents.run_config import RunConfig
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from ._plugin import BLOCK_MESSAGE, ControlsPlugin
from ._profile import Profile, revalidate

_LOG = logging.getLogger(__name__)


class GoogleADKAdapter:
    """Own only its private runner/session/plugin/task, never caller models or clients."""

    def __init__(self, native: Profile, controls: PreparedControls) -> None:
        self._native = native
        self._session = PolicySession("google-adk", controls)
        self._admission = RunAdmission()
        self._abort_signal: asyncio.Event | None = None
        self._owner = NativeTaskOwner(self._fail_native, trace_id=self._session.trace_id)
        self._native_error: BaseException | None = None
        self._resource_error: AdapterCleanupError | None = None
        self._shutdown_error: BaseException | None = None

    def _fail_native(self, error: BaseException) -> None:
        self._session.fail(error)
        if self._abort_signal is not None:
            self._abort_signal.set()

    @property
    def records(self) -> tuple[InterceptionRecord, ...]:
        """Return detached actual SDK records, including records after denial."""
        return self._session.records

    @property
    def records_dropped(self) -> int:
        """Return the exact shared worker audit eviction count."""
        return self._session.records_dropped

    @property
    def cleanup_error(self) -> AdapterCleanupError | None:
        """Expose incomplete native, ephemeral-resource or policy cleanup."""
        return self._owner.cleanup_error or self._resource_error or self._session.cleanup_error

    def run(self, request: RunRequest | None = None) -> RunResult:
        """Refuse before admission; correct to arun without losing the single-use instance."""
        raise RuntimeError(
            "Google ADK requires await adapter.arun(request) on the caller's event loop."
        )

    async def _cleanup(
        self,
        operations: tuple[Callable[[], Awaitable[object]], ...],
        primary: BaseException | None,
    ) -> None:
        for operation in operations:
            try:
                await operation()
            except Exception as error:
                _LOG.error(
                    "adapter_owned_cleanup_failed",
                    extra={"trace_id": self._session.trace_id, "error_type": type(error).__name__},
                )
                if self._resource_error is None:
                    self._resource_error = AdapterCleanupError("Google ADK owned cleanup failed")
                    self._resource_error.__cause__ = error
                if primary is not None:
                    primary.add_note("Google ADK owned cleanup incomplete.")
        if self._resource_error is not None and primary is None:
            raise self._resource_error

    async def arun(self, request: RunRequest | None = None) -> RunResult:
        """Run once on the caller loop with a fixed native wall/deadline and bounded drain."""
        prompt = self._admission.admit(request)
        revalidate(self._native)

        async def execute() -> str:
            user_id, session_id, app_name = (f"adk_{uuid4().hex}" for _ in range(3))
            sessions = InMemorySessionService()
            self._abort_signal = asyncio.Event()
            plugin = ControlsPlugin(
                self._native,
                self._session,
                user_id=user_id,
                session_id=session_id,
                abort_signal=self._abort_signal,
            )
            app = App(name=app_name, root_agent=self._native.root, plugins=[plugin])
            runner = Runner(app=app, session_service=sessions, plugin_close_timeout=2)
            stream = runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                run_config=RunConfig(max_llm_calls=11),
                abort_signal=self._abort_signal,
            )
            primary: BaseException | None = None
            created = False
            try:
                self._session.begin()
                await sessions.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                )
                created = True
                async for _ in stream:
                    pass
                if self._session.policy_failed:
                    raise AdapterBlocked(BLOCK_MESSAGE)
                if plugin.output is None:
                    raise ValueError("Google ADK did not produce governed final text")
                return plugin.output
            except BaseException as error:
                primary = error
                self._native_error = error
                raise
            finally:

                async def shutdown() -> None:
                    try:
                        await plugin.shutdown("completed" if primary is None else "failed")
                    except Exception as error:
                        self._shutdown_error = error
                        _LOG.error(
                            "adapter_shutdown_audit_unavailable",
                            extra={
                                "trace_id": self._session.trace_id,
                                "error_type": type(error).__name__,
                            },
                        )
                        if primary is None:
                            raise
                        primary.add_note(
                            "Shutdown audit unavailable after governance lane failure."
                        )

                async def delete() -> None:
                    if created:
                        await sessions.delete_session(
                            app_name=app_name,
                            user_id=user_id,
                            session_id=session_id,
                        )

                try:
                    try:
                        await self._cleanup((stream.aclose,), primary)
                    finally:
                        try:
                            await shutdown()
                        finally:
                            await self._cleanup((delete, runner.close), primary)
                finally:
                    await self._session.aclose(primary or self._resource_error)

        try:
            output = await self._owner.run(execute)
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._session.policy_failed:
                raise AdapterBlocked(BLOCK_MESSAGE) from self._session.last_error
            raise
        return RunResult(output, self.records, self.records_dropped)
