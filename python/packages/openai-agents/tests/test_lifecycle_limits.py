"""Actual native admission/cancellation/transport ownership and exact resource thresholds."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import httpx2
import pytest
from agent_hooks import ALLOW, AgentContext, Verdict
from agent_hooks_adapters import AdapterBlocked, RunRequest
from agent_hooks_adapters import runtime as shared_runtime
from agent_hooks_adapters.runtime import RunBudget, validate_topology
from agent_hooks_openai_agents._model import GovernedModel
from agent_hooks_openai_agents._profile import profile
from agent_hooks_openai_agents.adapter import OpenAIAgentsAdapter
from agent_hooks_openai_agents.portable import create_backend
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, RunConfig, Runner
from agents.models.interface import ModelTracing
from agents.result import RunResult as NativeRunResult
from openai import AsyncOpenAI
from openai_adapter_test_helpers import Policy, controlled, native_case, scope_of


def test_arun_caller_loop_and_real_caller_transport_open_reusable() -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        requests: list[httpx2.Request] = []

        def completion(request: httpx2.Request) -> httpx2.Response:
            assert request.url.path == "/v1/chat/completions"
            requests.append(request)
            return httpx2.Response(
                200,
                json={
                    "id": "completion-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "unit-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "native text"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                },
            )

        http = httpx2.AsyncClient(
            transport=httpx2.MockTransport(completion), trust_env=False, follow_redirects=False
        )
        client = AsyncOpenAI(
            api_key="local-unit-test",
            base_url="http://127.0.0.1:1/v1",
            http_client=http,
            max_retries=0,
            timeout=1,
        )
        policy_threads: list[int] = []

        def policy_thread(context: AgentContext) -> Verdict:
            policy_threads.append(threading.get_ident())
            return ALLOW

        try:
            model = OpenAIChatCompletionsModel(model="unit-model", openai_client=client)
            agent = Agent[object](
                name="native-agent", model=model, model_settings=ModelSettings(max_tokens=32)
            )
            adapter = create_backend(
                runtime=agent,
                interceptors=[Policy(policy_thread)],
                policy_file=None,
                agents=None,
                agent_policies=(),
                resolver=None,
                timeout=5,
                emit_timeout=30,
                max_records=1000,
            )
            result = await adapter.arun(RunRequest({"prompt": "unit transport"}))
            assert result.output == "native text"
            assert asyncio.get_running_loop() is loop and loop.is_running()
            assert not client.is_closed() and not http.is_closed
            await client.chat.completions.create(
                model="unit-model",
                messages=[{"role": "user", "content": "reuse"}],
                max_tokens=32,
                temperature=0,
            )
            assert len(requests) == 2
            assert agent.model is model and adapter.cleanup_error is None
            assert len(set(policy_threads)) == 1
            assert threading.get_ident() not in policy_threads
        finally:
            await client.close()

    asyncio.run(scenario())


def test_atomic_single_use_arun_race_and_sync_refusal_no_consumption() -> None:
    async def scenario() -> None:
        case = native_case()
        entered, release = asyncio.Event(), asyncio.Event()
        case.model.entered, case.model.release = entered, release
        adapter = controlled(case, policies=[Policy()])
        first = asyncio.create_task(adapter.arun(RunRequest({"prompt": "first"})))
        await asyncio.wait_for(entered.wait(), 2)
        try:
            with pytest.raises(RuntimeError, match="single-use"):
                await adapter.arun(RunRequest({"prompt": "second"}))
            with pytest.raises(RuntimeError, match=r"await adapter.arun"):
                adapter.run(RunRequest({"prompt": "third"}))
            assert len(case.model.seen) == 1
        finally:
            release.set()
            await first

    asyncio.run(scenario())


def test_native_caller_cancel_drains_owned_task_and_policy_thread() -> None:
    async def scenario() -> None:
        before = asyncio.all_tasks()
        case = native_case()
        entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        case.model.entered, case.model.release, case.model.cancelled = entered, release, cancelled
        adapter = controlled(case, policies=[Policy()])
        task = asyncio.create_task(adapter.arun(RunRequest({"prompt": "safe"})))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set() and adapter.cleanup_error is None
        assert scope_of(adapter).tool_tasks == set()
        assert asyncio.all_tasks() == before
        assert not case.model.closed
        assert not any(
            thread.name.startswith("agent-hooks") and thread.is_alive()
            for thread in threading.enumerate()
        )

    asyncio.run(scenario())


def test_native_run_deadline_and_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shared_runtime, "_NATIVE_RUN_TIMEOUT", 0.1)
    monkeypatch.setattr(shared_runtime, "_NATIVE_CLEANUP_TIMEOUT", 0.05)

    async def scenario() -> None:
        case = native_case()
        entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        case.model.entered, case.model.release, case.model.cancelled = entered, release, cancelled
        adapter = controlled(case, policies=[Policy()])
        with pytest.raises(AdapterBlocked):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert entered.is_set() and cancelled.is_set()
        assert adapter.cleanup_error is None and not case.model.closed
        with pytest.raises(RuntimeError, match="single-use"):
            await adapter.arun(RunRequest({"prompt": "later"}))

    asyncio.run(scenario())


def test_mutation_during_policy_wait_stops_native_effect() -> None:
    def mutate(context: AgentContext) -> Verdict:
        if context["interception_point"] == "pre_model_call":
            case.agent.instructions = "changed while policy ran"
        return ALLOW

    case = native_case()

    async def scenario() -> None:
        adapter = controlled(case, policies=[Policy(mutate)])
        with pytest.raises(AdapterBlocked):
            await adapter.arun(RunRequest({"prompt": "safe"}))
        assert case.model.seen == [] and adapter.cleanup_error is None

    asyncio.run(scenario())


def test_actor_node_edge_caps_and_local_tool_cap() -> None:
    validate_topology(actors=16, nodes=16, edges=64)
    for actors, nodes, edges in ((17, 1, 0), (1, 17, 0), (1, 1, 65), (True, 1, 0)):
        with pytest.raises((ValueError, TypeError)):
            validate_topology(actors=actors, nodes=nodes, edges=edges)
    case = native_case(use_tool=True)
    case.agent.tools = [replace(case.tool, name=f"tool-{index}") for index in range(16)]
    assert len(profile(case.agent).tools) == 16
    case.agent.tools.append(replace(case.tool, name="tool-16"))
    with pytest.raises(ValueError, match="16"):
        profile(case.agent)


def test_model_call_budget_exhausted_actual_model_delegate() -> None:
    async def scenario() -> None:
        case = native_case()
        adapter = controlled(case, policies=[Policy()])
        scope = scope_of(adapter)
        scope.session.begin()
        agent = scope.clone()
        assert isinstance(agent.model, GovernedModel)
        primary: BaseException | None = None
        try:
            for _ in range(10):
                await agent.model.get_response(
                    None,
                    "safe",
                    ModelSettings(parallel_tool_calls=False),
                    [],
                    None,
                    [],
                    ModelTracing.DISABLED,
                    previous_response_id=None,
                    conversation_id=None,
                    prompt=None,
                )
            with pytest.raises(AdapterBlocked) as blocked:
                await agent.model.get_response(
                    None,
                    "safe",
                    ModelSettings(parallel_tool_calls=False),
                    [],
                    None,
                    [],
                    ModelTracing.DISABLED,
                    previous_response_id=None,
                    conversation_id=None,
                    prompt=None,
                )
            primary = blocked.value
            assert len(case.model.seen) == 10
        finally:
            await scope.session.aclose(primary)

    asyncio.run(scenario())


def test_pending_correlation_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shared_runtime, "_MAX_MODEL_CALLS", 128)
    budget = RunBudget("unit-pending")
    for index in range(64):
        budget.before_model(0, f"model-{index}")
        budget.after_model(0, f"model-{index}", [f"tool-{index}"])
    assert budget.snapshot.pending == 64
    with pytest.raises(AdapterBlocked, match="Pending correlation limit"):
        budget.before_model(0, "model-64")


def test_native_event_cap() -> None:
    budget = RunBudget("unit-events")
    for _ in range(256):
        budget.event()
    assert budget.snapshot.native_events == 256
    with pytest.raises(AdapterBlocked, match="event limit"):
        budget.event()


def test_fixed_turns_concurrency_and_tracing_actual_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Runner.run
    configs: list[RunConfig] = []

    async def observed(
        agent: Agent[object], input: str, *, max_turns: int, run_config: RunConfig
    ) -> NativeRunResult:
        assert max_turns == 10
        configs.append(run_config)
        return await original(agent, input, max_turns=max_turns, run_config=run_config)

    monkeypatch.setattr(Runner, "run", staticmethod(observed))

    async def scenario() -> None:
        case = native_case(use_tool=True)
        adapter = controlled(case, policies=[Policy()])
        result = await adapter.arun(RunRequest({"prompt": "safe"}))
        assert result.output == "native response"
        assert configs[0].tracing_disabled
        assert not configs[0].trace_include_sensitive_data
        assert configs[0].tool_execution is not None
        assert configs[0].tool_execution.max_function_tool_concurrency == 1
        assert isinstance(adapter, OpenAIAgentsAdapter)
        assert not scope_of(adapter).tool_tasks
        assert vars(asyncio.get_running_loop()).get("_default_executor") is None

    asyncio.run(scenario())
