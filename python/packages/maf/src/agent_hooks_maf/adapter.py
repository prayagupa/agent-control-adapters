"""Single-use ACS scope using the upstream native middleware and codec ownership."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import cast
from uuid import uuid4

from agent_framework import AgentResponse, create_agent_hooks_middleware_from_emitter
from agent_hooks import (
    AgentContext,
    AgentContextBuilder,
    EmitOutcome,
    InterceptionEmitter,
    InterceptionRecord,
)
from agent_hooks_adapters import AdapterBlocked, AdapterCleanupError, RunRequest, RunResult
from agent_hooks_adapters.bindings import PreparedControls
from agent_hooks_adapters.codecs import (
    envelope,
    input_text,
    model_text,
    output_text,
    text_messages,
    unchanged,
    wire_equal,
)
from agent_hooks_adapters.json import JSON, normalize, object_value
from agent_hooks_adapters.runtime import NativeTaskOwner, PolicySession, RunAdmission

from ._profile import Profile, revalidate

_LOG = logging.getLogger(__name__)
_BLOCK_MESSAGE = "MAF execution blocked by governance."


class BoundedInterceptionEmitter(InterceptionEmitter):
    """Native emitter subtype delegating actual outcomes to the bounded policy lane."""

    def __init__(self, session: PolicySession, native: Profile) -> None:
        super().__init__()
        self._session = session
        self._native = native
        self._pending_model: str | None = None

    def _before(self, context: AgentContext) -> dict[str, JSON]:
        revalidate(self._native)
        normalized = object_value(context)
        point = normalized["interception_point"]
        target = normalized["target"]
        budget = self._session.budget
        if point == "pre_model_call":
            if self._pending_model is not None:
                raise ValueError("Overlapping native model calls are unsupported")
            self._pending_model = str(uuid4())
            budget.before_model(0, self._pending_model)
        elif point == "post_model_call":
            if self._pending_model is None:
                raise ValueError("Unexpected native model completion")
            response = envelope(target, {"content", "tool_calls", "finish_reason"})
            if response["content"] is not None and not isinstance(response["content"], str):
                raise ValueError("Only plain text native model output is supported")
            calls = response["tool_calls"]
            if not isinstance(calls, list):
                raise ValueError("Unsupported native tool call output")
            if response["content"] is None and not calls:
                raise ValueError("Native model output requires text or a tool call")
            ids: list[str] = []
            for call in calls:
                projected = envelope(call, {"id", "name", "args"})
                identifier = projected["id"]
                if not isinstance(identifier, str):
                    raise ValueError("Unknown native tool output")
                self._validate_tool_arguments(projected["name"], projected["args"])
                ids.append(identifier)
            budget.after_model(0, self._pending_model, ids)
            self._pending_model = None
        elif point in ("pre_tool_call", "post_tool_call"):
            call = object_value(normalized.get("tool_call"))
            identifier = call.get("id")
            if not isinstance(identifier, str):
                raise ValueError("Invalid native tool correlation")
            if point == "pre_tool_call":
                budget.before_tool(0, identifier)
            else:
                budget.after_tool(0, identifier)
        return normalized

    def _validate_tool_arguments(self, name: JSON, value: object) -> None:
        arguments = object_value(value)
        tool = next((tool for tool in self._native.tools if tool.name == name), None)
        if tool is None:
            raise ValueError("Unknown native tool")
        if tool.input_model is not None:
            parsed = tool.input_model.model_validate(arguments, strict=True)
            if not wire_equal(parsed.model_dump(mode="json", exclude_unset=True), arguments):
                raise ValueError("Native tool arguments cannot be dropped or coerced")
        elif arguments:
            raise ValueError("Unexpected arguments for an argument-free tool")

    def _validate_target(self, context: dict[str, JSON], target: object) -> None:
        original = context["target"]
        point = context["interception_point"]
        if point == "agent_startup":
            unchanged(target, original)
        elif point == "input":
            input_text(target)
        elif point == "output":
            original_output = envelope(original, {"content"})
            if isinstance(original_output["content"], str):
                output_text(target)
            else:
                unchanged(target, original)
        elif point == "pre_model_call":
            if not isinstance(original, list):
                raise ValueError("Unsupported native model messages")
            messages = [object_value(message) for message in original]
            text_messages(target, messages)
        elif point == "post_model_call":
            model_text(target, object_value(original))
        elif point == "pre_tool_call":
            call = object_value(context.get("tool_call"))
            self._validate_tool_arguments(call.get("name"), target)
        elif point == "post_tool_call":
            normalize(target)
        else:
            raise ValueError("Unknown native interception point")

    async def emit(self, ctx: AgentContext) -> EmitOutcome:
        """Return the actual SDK outcome; native codecs alone perform write-back."""
        try:
            context = self._before(ctx)
            outcome = await self._session.aemit_outcome(0, ctx)
            self._validate_target(context, outcome.target)
            return outcome
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self._session.policy_failed:
                self._session.fail(error, policy=True)
            raise AdapterBlocked(_BLOCK_MESSAGE) from None

    async def emit_unchecked(self, ctx: AgentContext) -> InterceptionRecord:
        """Preserve the upstream caller-loop builder's actual shutdown sequence."""
        return await self._session.aemit_unchecked(0, ctx)


class MAFAdapter:
    """ACS-only native run scope; caller transports, clients and loops stay open."""

    def __init__(self, native: Profile, controls: PreparedControls) -> None:
        self._native = native
        self._session = PolicySession("maf", controls)
        self._admission = RunAdmission()
        self._owner = NativeTaskOwner(self._session.fail, trace_id=self._session.trace_id)

    @property
    def records(self) -> tuple[InterceptionRecord, ...]:
        """Expose only the actual bounded SDK audit snapshot."""
        return self._session.records

    @property
    def records_dropped(self) -> int:
        """Expose actual SDK record eviction."""
        return self._session.records_dropped

    @property
    def cleanup_error(self) -> AdapterCleanupError | None:
        """Keep native and policy cleanup diagnostics separate from the primary error."""
        return self._owner.cleanup_error or self._session.cleanup_error

    def run(self, request: RunRequest | None = None) -> RunResult:
        """Refuse sync execution before admission; use arun on the existing loop."""
        raise RuntimeError("MAF requires await adapter.arun(request) on the caller's event loop.")

    async def arun(self, request: RunRequest | None = None) -> RunResult:
        """Execute one bounded native run with a host-owned SDK lifecycle."""
        prompt = self._admission.admit(request)
        revalidate(self._native)

        async def execute() -> str:
            self._session.begin()
            actor = self._session.controls.actors[0]
            builder = AgentContextBuilder(
                agent_id=actor.identity,
                framework="maf",
                session_id=actor.session_id,
                agent_name=actor.label,
            )
            emitter = BoundedInterceptionEmitter(self._session, self._native)
            bundle = create_agent_hooks_middleware_from_emitter(emitter, builder)
            primary: BaseException | None = None
            try:
                await emitter.emit(
                    builder.agent_startup(
                        tools_registered=[tool.name for tool in self._native.tools]
                    )
                )
                response: object = await cast(
                    "Awaitable[object]",
                    self._native.agent.run(
                        prompt, middleware=[bundle], options={"allow_multiple_tool_calls": False}
                    ),
                )
                if self._session.policy_failed:
                    raise AdapterBlocked(_BLOCK_MESSAGE)
                if not isinstance(response, AgentResponse):
                    raise ValueError("MAF returned an unsupported native result")
                return response.text
            except asyncio.CancelledError as error:
                primary = error
                raise
            except Exception as error:
                primary = error
                if self._session.policy_failed:
                    raise AdapterBlocked(_BLOCK_MESSAGE) from self._session.last_error
                raise
            finally:
                try:
                    await emitter.emit_unchecked(
                        builder.agent_shutdown(reason="completed" if primary is None else "failed")
                    )
                except Exception as shutdown_error:
                    _LOG.error(
                        "adapter_shutdown_failed",
                        extra={
                            "trace_id": self._session.trace_id,
                            "error_type": type(shutdown_error).__name__,
                        },
                    )
                    if primary is None:
                        raise
                finally:
                    await self._session.aclose(primary)

        output = await self._owner.run(execute)
        return RunResult(output, self.records, self.records_dropped)
