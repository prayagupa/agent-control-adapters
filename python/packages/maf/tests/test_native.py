"""Actual native middleware activation, executable transforms and whole-run denial."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from agent_adapter_conformance import (
    INVALID_PROMPTS,
    assert_denied_before_effect,
    assert_governed_success,
    make_rego_manifest,
)
from agent_hooks import Decision
from agent_hooks_adapters import AdapterBlocked, RunRequest
from maf_adapter_test_helpers import controlled, native_case


def test_native_bundle_all_points_and_owned_sequence(
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch, use_tool=True)
        adapter = controlled(case, allow_manifest)
        try:
            result = await adapter.arun(RunRequest({"prompt": "Call ping"}))
            assert result.output == "native response"
            assert case.effects == ["native"]
            assert case.model.calls == 2
            assert_governed_success(
                result,
                points=(
                    "agent_startup",
                    "input",
                    "pre_model_call",
                    "post_model_call",
                    "pre_tool_call",
                    "post_tool_call",
                    "output",
                    "agent_shutdown",
                ),
            )
            assert [record.sequence for record in result.records] == list(
                range(len(result.records))
            )
            assert len({record.session_id for record in result.records}) == 1
            assert adapter.cleanup_error is None
            assert case.agent.middleware is None
            assert case.agent.client is not None
            assert not case.transport.is_closed()
            assert asyncio.get_running_loop().is_running()
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "point",
    [
        "agent_startup",
        "input",
        "pre_model_call",
        "post_model_call",
        "pre_tool_call",
        "output",
    ],
)
def test_whole_run_deny_and_original_operator_cause(
    point: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = make_rego_manifest(
        tmp_path,
        f'''verdict := {{
        "decision": "deny", "reason": "private policy detail"
    }} if {{ input.snapshot.interception_point == "{point}" }}''',
    )

    async def scenario() -> None:
        case = native_case(monkeypatch, use_tool=True)
        adapter = controlled(case, path)
        try:
            with pytest.raises(AdapterBlocked) as blocked:
                await adapter.arun(RunRequest({"prompt": "Call ping"}))
            assert blocked.value.__cause__ is not None
            assert "private policy detail" not in str(blocked.value)
            assert any(
                record.interception_point.value == point
                and record.verdict.decision is Decision.DENY
                for record in adapter.records
            )
            if point != "output":
                assert_denied_before_effect(adapter, point=point, effects=len(case.effects))
            assert adapter.cleanup_error is None
            assert not case.transport.is_closed()
            with pytest.raises(RuntimeError, match="single-use"):
                await adapter.arun(RunRequest({"prompt": "later"}))
        finally:
            await case.transport.close()

    asyncio.run(scenario())
    assert "private policy detail" not in caplog.text


def test_post_tool_deny_does_not_claim_rollback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = make_rego_manifest(
        tmp_path,
        """verdict := {"decision": "deny"} if {
        input.snapshot.interception_point == "post_tool_call"
    }""",
    )

    async def scenario() -> None:
        case = native_case(monkeypatch, use_tool=True)
        adapter = controlled(case, path)
        try:
            with pytest.raises(AdapterBlocked):
                await adapter.arun(RunRequest({"prompt": "Call ping"}))
            assert case.effects == ["native"]
            assert case.model.calls == 1
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "point,path,new_value",
    [
        ("input", "$target.content", "changed input"),
        ("pre_model_call", "$target[0].content", "changed request"),
        ("post_model_call", "$target.content", "changed model"),
        ("output", "$target.content", "changed output"),
        ("pre_tool_call", "$target.value", "changed arguments"),
        ("post_tool_call", "$target", "changed result"),
    ],
)
def test_native_transforms_reach_actual_model_tool_or_output(
    point: str,
    path: str,
    new_value: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = make_rego_manifest(
        tmp_path,
        f'''verdict := {{
        "decision": "transform", "transform": {{"path": "{path}", "value": "{new_value}"}}
    }} if {{ input.snapshot.interception_point == "{point}" }}''',
    )

    async def scenario() -> None:
        use_tool = point in ("pre_tool_call", "post_tool_call")
        case = native_case(monkeypatch, use_tool=use_tool)
        adapter = controlled(case, manifest)
        try:
            result = await adapter.arun(RunRequest({"prompt": "original prompt"}))
            seen = json.dumps(
                [[message.to_dict() for message in messages] for messages in case.model.seen]
            )
            if point in ("input", "pre_model_call", "post_tool_call"):
                assert new_value in seen
            elif point == "pre_tool_call":
                assert case.effects == [new_value]
            else:
                assert result.output == new_value
            assert any(record.verdict.decision is Decision.TRANSFORM for record in result.records)
        finally:
            await case.transport.close()

    asyncio.run(scenario())


def test_unsupported_transform_fails_closed_before_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = make_rego_manifest(
        tmp_path,
        """verdict := {
        "decision": "transform", "transform": {"path": "$target.role", "value": "system"}
    } if { input.snapshot.interception_point == "input" }""",
    )

    async def scenario() -> None:
        case = native_case(monkeypatch)
        adapter = controlled(case, path)
        try:
            with pytest.raises(AdapterBlocked):
                await adapter.arun(RunRequest({"prompt": "prompt"}))
            assert case.model.calls == 0
        finally:
            await case.transport.close()

    asyncio.run(scenario())


def test_engine_error_is_a_real_fail_closed_native_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = make_rego_manifest(tmp_path, "not valid rego")

    async def scenario() -> None:
        case = native_case(monkeypatch)
        adapter = controlled(case, path)
        try:
            with pytest.raises(AdapterBlocked):
                await adapter.arun(RunRequest({"prompt": "prompt"}))
            assert case.model.calls == 0
            assert adapter.records
            assert adapter.records[0].verdict.decision is Decision.DENY
        finally:
            await case.transport.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("request_value", INVALID_PROMPTS)
def test_invalid_request_consumes_adapter_without_native_work(
    request_value: RunRequest | None,
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch)
        adapter = controlled(case, allow_manifest)
        try:
            with pytest.raises(ValueError):
                await adapter.arun(request_value)
            assert case.model.calls == 0
            assert adapter.records == ()
            with pytest.raises(RuntimeError, match="single-use"):
                await adapter.arun(RunRequest({"prompt": "valid"}))
        finally:
            await case.transport.close()

    asyncio.run(scenario())


def test_sync_refusal_does_not_consume_admission(
    monkeypatch: pytest.MonkeyPatch,
    allow_manifest: Path,
) -> None:
    async def scenario() -> None:
        case = native_case(monkeypatch)
        adapter = controlled(case, allow_manifest)
        try:
            with pytest.raises(RuntimeError, match="await"):
                adapter.run(RunRequest({"prompt": "valid"}))
            assert (await adapter.arun(RunRequest({"prompt": "valid"}))).output == "native response"
        finally:
            await case.transport.close()

    asyncio.run(scenario())
