"""Coupled graph-adapter vectors using real native SDKs and real ACS/Rego."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import pytest
from agent_adapter_conformance import INVALID_PROMPTS, assert_governed_success, make_rego_manifest
from agent_hooks import AgentContext, Decision, Interceptor, Transform, Verdict
from agent_hooks_adapters import AgentPolicy, AsyncAdapter, RunRequest, RunResult
from agent_hooks_adapters.errors import AdapterBlocked
from agent_hooks_langgraph._inspection import attribute, native_builder, native_nodes
from agent_hooks_langgraph._scope import GraphAdapter, NativeScope
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langgraph_adapter_test_helpers import POINTS, NativePolicy, NativeTransport

Mode = Literal["sync", "async"]


@dataclass
class FamilyCase:
    """Transport/effects plus a visible framework-specific builder supplied by each suite."""

    constructor: Callable[..., AsyncAdapter]
    runtime: object
    model: NativeTransport
    effects: list[str]
    controller: Callable[[], object]
    graph: Callable[[], object]

    def adapter(
        self,
        *,
        policy: Interceptor | None = None,
        path: Path | None = None,
        policies: tuple[AgentPolicy, ...] = (),
        max_records: int = 1000,
    ) -> AsyncAdapter:
        return self.constructor(
            runtime=self.runtime,
            interceptors=[policy if policy is not None else NativePolicy()]
            if path is None
            else None,
            policy_file=path,
            agents=None,
            agent_policies=policies,
            resolver=None,
            timeout=5,
            emit_timeout=30,
            max_records=max_records,
        )

    def scope(self) -> NativeScope:
        getter = attribute(self.controller(), "scope")
        if not callable(getter):
            raise TypeError("Native controller has no scope")
        value: object = getter()
        if not isinstance(value, NativeScope):
            raise TypeError("Native scope must be actual graph-owned scope")
        return value


def execute(adapter: AsyncAdapter, mode: Mode, request: RunRequest | None = None) -> RunResult:
    request = request or RunRequest({"prompt": "native request"})
    return adapter.run(request) if mode == "sync" else asyncio.run(adapter.arun(request))


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_real_native_all_points(family_case: FamilyCase, mode: Mode) -> None:
    adapter = family_case.adapter()
    result = execute(adapter, mode)
    assert_governed_success(result, points=POINTS)
    assert result.output == "native final"
    assert family_case.effects == ["original"]
    assert adapter.cleanup_error is None
    scope = family_case.scope()
    assert not scope.active
    assert scope.ledger.pending == scope.session.budget.snapshot.pending == 0
    assert result.records == adapter.records


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("point", POINTS)
def test_native_deny_each_point(family_case: FamilyCase, mode: Mode, point: str) -> None:
    adapter = family_case.adapter(policy=NativePolicy(deny=point))
    if point == "agent_shutdown":
        result = execute(adapter, mode)
        assert result.output == "native final"
    else:
        with pytest.raises(AdapterBlocked):
            execute(adapter, mode)
    assert any(
        record.interception_point.value == point and record.verdict.decision is Decision.DENY
        for record in adapter.records
    )
    if point in ("agent_startup", "input", "pre_model_call", "post_model_call", "pre_tool_call"):
        assert family_case.effects == []
    if point in ("agent_startup", "input", "pre_model_call"):
        assert family_case.model.calls == 0
    if point == "pre_tool_call":
        assert not any(
            record.interception_point.value == "post_tool_call" for record in adapter.records
        )
    assert family_case.scope().ledger.pending == 0


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("point", POINTS)
def test_real_acs_deny_each_point(
    family_case: FamilyCase, mode: Mode, point: str, tmp_path: Path
) -> None:
    manifest = make_rego_manifest(
        tmp_path,
        'verdict := {"decision": "deny", "reason": "actual-rego-deny"} if {'
        f' input.snapshot.interception_point == "{point}"'
        " }\n",
    )
    adapter = family_case.adapter(path=manifest)
    if point == "agent_shutdown":
        execute(adapter, mode)
    else:
        with pytest.raises(AdapterBlocked):
            execute(adapter, mode)
    assert any(
        record.interception_point.value == point and record.verdict.decision is Decision.DENY
        for record in adapter.records
    )
    if point in ("agent_startup", "input", "pre_model_call", "post_model_call", "pre_tool_call"):
        assert family_case.effects == []


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_real_acs_success_and_actor_order(
    family_case: FamilyCase, mode: Mode, tmp_path: Path
) -> None:
    adapter = family_case.adapter(path=make_rego_manifest(tmp_path))
    result = execute(adapter, mode)
    assert_governed_success(result, points=POINTS)
    assert family_case.effects == ["original"]


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_real_acs_engine_invalid_verdict(
    family_case: FamilyCase, mode: Mode, tmp_path: Path
) -> None:
    manifest = make_rego_manifest(
        tmp_path,
        'verdict := {"decision": "not-a-real-decision"} if { '
        'input.snapshot.interception_point == "pre_model_call" }\n',
    )
    adapter = family_case.adapter(path=manifest)
    with pytest.raises(AdapterBlocked):
        execute(adapter, mode)
    assert family_case.effects == []
    assert family_case.model.calls == 0
    assert any(record.verdict.decision is Decision.DENY for record in adapter.records)


@pytest.mark.parametrize("admitted_request", INVALID_PROMPTS)
def test_invalid_admission_is_single_use(
    family_case: FamilyCase, admitted_request: RunRequest | None
) -> None:
    adapter = family_case.adapter()
    with pytest.raises(ValueError):
        adapter.run(admitted_request)
    with pytest.raises(RuntimeError, match="single-use"):
        asyncio.run(adapter.arun(RunRequest({"prompt": "correction"})))
    assert family_case.model.calls == 0
    assert family_case.effects == []
    assert adapter.records == ()


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_actual_transforms_preserve_ids_usage_and_roles(
    family_case: FamilyCase, mode: Mode
) -> None:
    adapter = family_case.adapter(policy=NativePolicy(transform=True))
    result = execute(adapter, mode)
    assert result.output == "transformed output"
    assert family_case.effects == ["transformed arguments"]
    message = next(item for item in family_case.model.seen[1] if isinstance(item, AIMessage))
    assert message.id == "response-tool"
    assert message.content == "transformed model"
    assert message.tool_calls[0]["id"] == "call-1"
    assert message.tool_calls[0]["name"] == "ping"
    assert message.tool_calls[0]["args"] == {"value": "original"}
    assert message.usage_metadata == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    assert message.response_metadata["finish_reason"] == "tool_calls"


class ChangedTarget:
    """Return a real SDK transform at one actual native consumer boundary."""

    def __init__(self, point: str, change: Callable[[object], object]) -> None:
        self.point, self.change = point, change

    def intercept(self, context: AgentContext, /) -> Verdict:
        if context["interception_point"] != self.point:
            return Verdict.allow()
        return Verdict(
            Decision.TRANSFORM,
            transform=Transform("$target", self.change(context["target"])),
        )


_UNSUPPORTED_TRANSFORMS: list[tuple[str, Callable[[object], object]]] = [
    ("agent_startup", lambda _: {"tools_registered": []}),
    ("input", lambda _: {"role": "system", "content": "spoof"}),
    ("output", lambda _: {"content": "output", "extra": "unsupported"}),
    ("pre_model_call", lambda _: []),
    ("post_model_call", lambda _: {"content": None, "tool_calls": [], "finish_reason": "stop"}),
    ("pre_tool_call", lambda _: {"value": "replacement", "extra": "dropped"}),
    ("pre_tool_call", lambda _: {"value": 17}),
]


@pytest.mark.parametrize(("point", "change"), _UNSUPPORTED_TRANSFORMS)
def test_unappliable_transform_aborts(
    family_case: FamilyCase, point: str, change: Callable[[object], object]
) -> None:
    adapter = family_case.adapter(policy=ChangedTarget(point, change))
    with pytest.raises(AdapterBlocked):
        execute(adapter, "async")
    if point not in ("output",):
        assert family_case.effects == []
    assert family_case.scope().ledger.pending == 0


@pytest.mark.parametrize(
    "arguments", [{"value": "x", "unknown": "drop"}, {"value": 17}, {"value": False}]
)
def test_original_arguments_checked_before_native_parser(
    family_case: FamilyCase, arguments: dict[str, object]
) -> None:
    family_case.model.responses[0] = AIMessage(
        content="", tool_calls=[{"id": "call-1", "name": "ping", "args": arguments}]
    )
    adapter = family_case.adapter()
    with pytest.raises(AdapterBlocked):
        execute(adapter, "async")
    assert family_case.effects == []
    assert not any(record.interception_point.value == "pre_tool_call" for record in adapter.records)


@pytest.mark.parametrize("content", [[{"type": "text", "text": "rich"}], None])
def test_rich_null_model_output_rejected(family_case: FamilyCase, content: object) -> None:
    family_case.model.responses[0] = AIMessage.model_construct(content=content)
    adapter = family_case.adapter()
    with pytest.raises(AdapterBlocked if content is not None else TypeError):
        execute(adapter, "async")
    assert family_case.effects == []


def test_one_tool_per_output(family_case: FamilyCase) -> None:
    family_case.model.responses[0] = AIMessage(
        content="",
        tool_calls=[
            {"id": "one", "name": "ping", "args": {"value": "first"}},
            {"id": "two", "name": "ping", "args": {"value": "second"}},
        ],
    )
    adapter = family_case.adapter()
    with pytest.raises(AdapterBlocked):
        execute(adapter, "async")
    assert family_case.effects == []


def test_model_call_budget_exact_threshold(family_case: FamilyCase) -> None:
    family_case.model.responses[:] = [
        AIMessage(
            content="",
            tool_calls=[{"id": f"call-{index}", "name": "ping", "args": {"value": "loop"}}],
        )
        for index in range(11)
    ]
    adapter = family_case.adapter()
    with pytest.raises(AdapterBlocked):
        execute(adapter, "async")
    assert family_case.model.calls == 10
    assert len(family_case.effects) == 10
    assert family_case.scope().session.budget.snapshot.model_calls == 10


def test_actual_binding_mutation_zero_effects(family_case: FamilyCase) -> None:
    adapter = family_case.adapter()
    scope = family_case.scope()
    node = next(iter(native_nodes(scope.proof.graph).values()))

    def foreign_mapper(value: object) -> object:
        return value

    node.mapper = foreign_mapper
    with pytest.raises(ValueError, match="binding changed"):
        execute(adapter, "async")
    assert adapter.records == ()
    assert family_case.model.calls == 0


@pytest.mark.parametrize(
    "field", ["bound", "mapper", "writers", "triggers", "channels", "retry_policy", "timeout"]
)
def test_actual_compiled_fields_revalidated(family_case: FamilyCase, field: str) -> None:
    adapter = family_case.adapter()
    graph = family_case.scope().proof.graph
    node = next(value for name, value in native_nodes(graph).items() if name != "__start__")
    setattr(node, field, object())
    with pytest.raises((ValueError, TypeError)):
        execute(adapter, "sync")
    assert family_case.model.calls == 0
    assert family_case.effects == []


def test_schema_and_topology_mutation_revalidated(family_case: FamilyCase) -> None:
    adapter = family_case.adapter()
    builder = native_builder(family_case.scope().proof.graph)
    builder.edges.add(("model", "foreign"))
    with pytest.raises(ValueError, match="binding changed"):
        execute(adapter, "sync")
    assert adapter.records == ()


def test_records_retained_and_dropped_exact(family_case: FamilyCase) -> None:
    adapter = family_case.adapter(max_records=2)
    result = execute(adapter, "async")
    assert len(result.records) == 2
    # One startup/shutdown per actual actor, plus six actual in-run points and a second model pair.
    count = len(family_case.scope().session.controls.actors) * 2 + 8
    assert result.records_dropped == count - 2
    assert result.records[-1].interception_point.value == "agent_shutdown"


def test_unknown_sdk_error_propagates(
    family_case: FamilyCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = RuntimeError("native transport failure")

    async def fail(
        self: NativeTransport,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        raise original

    monkeypatch.setattr(NativeTransport, "_agenerate", fail)
    adapter = family_case.adapter()
    with pytest.raises(RuntimeError) as raised:
        execute(adapter, "async")
    assert raised.value is original
    assert family_case.effects == []
    assert not any(
        record.interception_point.value == "post_model_call" for record in adapter.records
    )


def test_async_cancel_drains_and_preserves_loop(
    family_case: FamilyCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def check() -> None:
        entered, exited = asyncio.Event(), asyncio.Event()

        async def waiting(
            self: NativeTransport,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: object,
        ) -> ChatResult:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                exited.set()
            raise AssertionError("Unreachable")

        monkeypatch.setattr(NativeTransport, "_agenerate", waiting)
        adapter = family_case.adapter()
        task = asyncio.create_task(adapter.arun(RunRequest({"prompt": "cancel"})))
        await asyncio.wait_for(entered.wait(), 3)
        with pytest.raises(RuntimeError, match="single-use"):
            adapter.run(RunRequest({"prompt": "second"}))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exited.is_set()
        assert not family_case.scope().active
        assert family_case.scope().ledger.pending == 0
        assert adapter.cleanup_error is None
        await asyncio.sleep(0)
        assert not any(
            item.get_name() == "agent-controls-native-run" and not item.done()
            for item in asyncio.all_tasks()
        )

    asyncio.run(check())


def test_native_deadline_drains(family_case: FamilyCase, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_hooks_adapters import runtime as shared_runtime

    monkeypatch.setattr(shared_runtime, "_NATIVE_RUN_TIMEOUT", 0.05)

    async def waiting(
        self: NativeTransport,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        await asyncio.Event().wait()
        raise AssertionError("Unreachable")

    monkeypatch.setattr(NativeTransport, "_agenerate", waiting)
    adapter = family_case.adapter()
    with pytest.raises(AdapterBlocked, match="deadline"):
        execute(adapter, "async")
    assert adapter.cleanup_error is None
    assert family_case.scope().ledger.pending == 0


def test_single_use_sync_async_race(
    family_case: FamilyCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_core.callbacks import CallbackManagerForLLMRun

    entered, release = threading.Event(), threading.Event()
    from agent_hooks_langgraph._inspection import attribute

    class NativeGenerate(Protocol):
        def __call__(
            self,
            model: NativeTransport,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: object,
        ) -> ChatResult: ...

    original = cast("NativeGenerate", attribute(NativeTransport, "_generate"))
    outcomes: list[RunResult | BaseException] = []

    def waiting(
        self: NativeTransport,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        entered.set()
        if not release.wait(3):
            raise TimeoutError("Explicit unit transport deadline")
        return original(self, messages, stop, run_manager, **kwargs)

    monkeypatch.setattr(NativeTransport, "_generate", waiting)
    adapter = family_case.adapter()

    def first() -> None:
        try:
            outcomes.append(adapter.run(RunRequest({"prompt": "first"})))
        except BaseException as error:
            outcomes.append(error)

    thread = threading.Thread(target=first, name="graph-sync-admission-test")
    thread.start()
    try:
        assert entered.wait(3)
        with pytest.raises(RuntimeError, match="single-use"):
            asyncio.run(adapter.arun(RunRequest({"prompt": "second"})))
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], RunResult)


def test_fixed_native_config(family_case: FamilyCase) -> None:
    adapter = family_case.adapter()
    assert isinstance(adapter, GraphAdapter)
    config = attribute(adapter, "_config")
    assert callable(config)
    assert config() == {"recursion_limit": 30, "max_concurrency": 1}
