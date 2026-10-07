"""Metadata-only example diagnostics with explicit bounded synthetic request capture."""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import UTC, datetime
from typing import TextIO, cast

import httpx2
from agent_control_adapters import RunResult
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult

_REQUEST_BYTES = 4096
_LOG_BYTES = 65536
_MAX_RECORDS = 32
_SYNTHETIC_MODE = "local-toy-v1"


class _Emitter:
    def __init__(self) -> None:
        self.bytes = 0
        self.limited = False
        self.lock = threading.Lock()

    def write(self, value: dict[str, object], output: TextIO) -> None:
        # Callers supply fixed metadata or a request capped before JSON parsing.
        encoded = json.dumps(value, allow_nan=False) + "\n"
        size = len(encoded.encode())
        with self.lock:
            if self.limited:
                return
            if self.bytes + size > _LOG_BYTES - 256:
                encoded = '{"event":"telemetry_limit","capture_incomplete":true}\n'
                size = len(encoded)
                self.limited = True
            self.bytes += size
            output.write(encoded)
            output.flush()


_DEFAULT_EMITTER = _Emitter()


def _write(value: dict[str, object], output: TextIO | None = None) -> None:
    _DEFAULT_EMITTER.write(value, output if output is not None else sys.stderr)


def record_native_result(result: ChatResult) -> None:
    """Observe counts only; never serialize native/provider messages or metadata."""
    _write(
        {
            "event": "native_model_response",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "generations": len(result.generations),
            "tool_calls": sum(
                len(item.message.tool_calls)
                for item in result.generations
                if isinstance(item.message, AIMessage)
            ),
        }
    )


def record_run_result(result: RunResult, effects: list[str]) -> None:
    """Retain actual audit/effect counts without logging output or effect values."""
    _write(
        {
            "event": "native_run_result",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "tool_effects": len(effects),
            "records_dropped": result.records_dropped,
            "records_total": len(result.records),
            "records_truncated": len(result.records) > _MAX_RECORDS,
            "records": [
                {
                    "point": record.interception_point.value,
                    "decision": record.verdict.decision.value,
                }
                for record in result.records[:_MAX_RECORDS]
            ],
        }
    )


class ExampleTelemetry:
    """Never read response streams; default to payload-free bounded metadata."""

    def __init__(self, output: TextIO | None = None, *, synthetic_data: str | None = None) -> None:
        if synthetic_data not in (None, _SYNTHETIC_MODE):
            raise ValueError("Only explicit local-toy-v1 synthetic diagnostics are supported")
        self._output = output
        self._synthetic = synthetic_data is not None
        self._emitter = _Emitter() if output is not None else _DEFAULT_EMITTER
        self._count = 0
        self._pending: dict[httpx2.Request, tuple[int, float]] = {}
        if self._synthetic:
            self._emit(
                {
                    "event": "synthetic_diagnostics_warning",
                    "warning": "SYNTHETIC LOCAL TOY DATA ONLY; NOT FOR CUSTOMER DATA",
                    "response_payload_capture": False,
                }
            )

    def _emit(self, value: dict[str, object]) -> None:
        self._emitter.write(value, self._output if self._output is not None else sys.stderr)

    def summary(self, payload: dict[str, object]) -> dict[str, object]:
        """Strip output/effect/error strings by default; bound explicit toy results."""
        value = {
            key: payload[key]
            for key in ("status", "tool_effects", "records_dropped", "records", "cleanup_error")
            if key in payload
        }
        output = payload.get("output")
        effects = payload.get("tool_effect_values")
        if isinstance(output, str):
            value["output_characters"] = len(output)
        records = value.get("records")
        if isinstance(records, list):
            items = cast("list[object]", records)
            value["records_total"] = len(items)
            value["records"] = items[:_MAX_RECORDS]
            value["records_truncated"] = len(items) > _MAX_RECORDS
        cause = payload.get("sdk_cause")
        if isinstance(cause, dict) and cast("dict[str, object]", cause).get("type") == (
            "InterceptionBlocked"
        ):
            value["sdk_cause"] = {"type": "InterceptionBlocked"}
        if payload.get("native_error") == "AdapterBlocked":
            value["native_error"] = "AdapterBlocked"
        if self._synthetic:
            omitted = False
            if isinstance(output, str):
                if len(output) <= 1024:
                    value["output"] = output
                else:
                    omitted = True
            if isinstance(effects, list):
                items = cast("list[object]", effects)
                if len(items) <= 4 and all(
                    isinstance(item, str) and len(item) <= 1024 for item in items
                ):
                    value["tool_effect_values"] = effects
                else:
                    omitted = True
            value["synthetic_payload_omitted"] = omitted
        return value

    async def request(self, request: httpx2.Request) -> None:
        self._count += 1
        if self._count > 10:
            raise RuntimeError("Example transport request bound exceeded")
        started = time.monotonic()
        value: dict[str, object] = {
            "request_index": self._count,
            "event": "request_start",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "monotonic_seconds": started,
        }
        if self._synthetic:
            if request.url != httpx2.URL("http://localhost:11434/v1/chat/completions"):
                raise ValueError("Synthetic diagnostics require the fixed credential-free loopback")
            body = request.content
            value["request_bytes"] = len(body)
            if len(body) <= _REQUEST_BYTES:
                payload: object = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("Synthetic native request must be a JSON object")
                value["model_request"] = payload
            else:
                value["raw_request_omitted"] = True
        self._pending[request] = (self._count, started)
        self._emit(value)

    def failed(self, request: httpx2.Request, error: httpx2.HTTPError) -> None:
        index, started = self._pending.pop(request)
        self._emit(
            {
                "event": "request_exception_end",
                "request_index": index,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "exception_type": "ReadTimeout"
                if isinstance(error, httpx2.ReadTimeout)
                else "HTTPError",
            }
        )

    async def response(self, response: httpx2.Response) -> None:
        index, started = self._pending.pop(response.request)
        self._emit(
            {
                "event": "response_headers",
                "request_index": index,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "status_code": response.status_code,
                "response_payload_capture": False,
            }
        )

    def transport(self, inner: httpx2.AsyncBaseTransport) -> httpx2.AsyncBaseTransport:
        """Delegate unchanged I/O and rethrow the original specific HTTP exception."""
        return _ObservedTransport(self, inner)


class _ObservedTransport(httpx2.AsyncBaseTransport):
    def __init__(self, telemetry: ExampleTelemetry, inner: httpx2.AsyncBaseTransport) -> None:
        self._telemetry = telemetry
        self._inner = inner

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        try:
            return await self._inner.handle_async_request(request)
        except httpx2.HTTPError as error:
            self._telemetry.failed(request, error)
            raise

    async def aclose(self) -> None:
        await self._inner.aclose()
