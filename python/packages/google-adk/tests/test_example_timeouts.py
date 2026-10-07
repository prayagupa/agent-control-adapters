"""Exercise real example setup and native controls, stubbing only HTTP model transport."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import httpx
import pytest
from agent_control_adapters.json import JSON, object_value

_EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "google-adk"


@pytest.mark.parametrize("folder", ["agent_hooks", "acs"])
@pytest.mark.parametrize("script", ["llm_hooks.py", "tool_hooks.py"])
def test_example_effective_http_timeout_and_preserved_controls(
    folder: str,
    script: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[httpx.Request] = []
    tool_case = script == "tool_hooks.py"

    async def model_transport(
        _transport: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        assert str(request.url) == "http://localhost:11434/v1/chat/completions"
        requests.append(request)
        timeouts = object_value(request.extensions["timeout"])
        assert timeouts == {"connect": 60, "read": 60, "write": 60, "pool": 60}
        body = object_value(json.loads(await request.aread()))
        assert body["model"] == "qwen2.5:latest"
        assert body["temperature"] == 0
        assert body["max_tokens"] == 128
        message: dict[str, JSON] = {"role": "assistant"}
        if tool_case:
            assert body["tool_choice"] == {
                "type": "function",
                "function": {"name": "delete_file"},
            }
            message.update(
                content=None,
                tool_calls=[
                    {
                        "id": "unit-transport-call",
                        "type": "function",
                        "function": {
                            "name": "delete_file",
                            "arguments": json.dumps({"path": "demo.txt"}),
                        },
                    }
                ],
            )
        else:
            message["content"] = "  Unit model-transport boundary text.  "
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "unit-transport-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen2.5:latest",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_case else "stop",
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", model_transport)
    runpy.run_path(str(_EXAMPLES / folder / script), run_name="__main__")
    rows = [
        object_value(json.loads(line))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    summary = next(row for row in rows if "status" in row)
    assert len(requests) == summary["model_requests"] == 1
    assert summary["status"] == ("blocked" if tool_case else "allowed")
    assert any(row.get("point") == "agent_shutdown" for row in rows)
    if tool_case:
        assert summary["tool_effects"] == 0
        assert any(
            row.get("point") == "pre_tool_call" and row.get("decision") == "deny" for row in rows
        )
        assert not any(row.get("point") == "post_tool_call" for row in rows)
    else:
        for point in ("pre_model_call", "post_model_call"):
            assert any(
                row.get("point") == point and row.get("decision") == "transform" for row in rows
            )
