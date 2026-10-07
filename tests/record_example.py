"""Record real example execution without putting test instrumentation in the tutorial."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import patch

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"

import httpx
from agent_control_crewai import AdapterBlocked, CrewAIAdapter
from agent_hooks import InterceptionRecord
from crewai.crews.crew_output import CrewOutput

ROOT: Path = Path(__file__).resolve().parents[1]
EXAMPLES: tuple[str, ...] = (
    "agent_hooks.llm_hooks",
    "agent_hooks.tool_hooks",
    "agent_hooks.human_approval",
    "agent_hooks.multi_agent_llm_hooks",
    "acs.tool_hooks",
    "acs.multi_agent_llm_hooks",
)


@dataclass
class ExampleExecution:
    """Test-owned state populated only by the actual adapter execution."""

    records: tuple[InterceptionRecord, ...] = ()
    output: CrewOutput | None = None
    status: str | None = None
    runs: int = 0


def json_object(value: object) -> dict[str, object]:
    """Validate external JSON before recording model provenance."""
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    result = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in result):
        raise ValueError("Expected string JSON keys")
    return cast(dict[str, object], result)


def model_provenance(model: str) -> dict[str, object]:
    """Read the actual installed artifact and runtime without proxies or redirects."""
    with httpx.Client(timeout=5, trust_env=False, follow_redirects=False) as client:
        tags = client.get("http://localhost:11434/api/tags")
        tags.raise_for_status()
        models = json_object(tags.json()).get("models")
        if not isinstance(models, list):
            raise ValueError("Ollama returned an invalid model list")
        digest: str | None = None
        for value in cast(list[object], models):
            entry = json_object(value)
            if entry.get("name") == model:
                artifact = entry.get("digest")
                if isinstance(artifact, str) and len(artifact) == 64:
                    digest = artifact
        if digest is None:
            raise ValueError(f"Install the chosen local model before testing: {model}")
        response = client.get("http://localhost:11434/api/version")
        response.raise_for_status()
        runtime = json_object(response.json()).get("version")
        if not isinstance(runtime, str) or not runtime:
            raise ValueError("Ollama returned an invalid runtime version")
    return {"model": model, "digest": digest, "runtime": runtime}


def record_example(example: str, script: Path | None = None) -> dict[str, object]:
    """Observe real HTTP requests and SDK records; never replace execution or output."""
    if example not in EXAMPLES:
        raise ValueError("Unknown example")
    model = os.getenv("LOCAL_MODEL", "qwen2.5:latest")
    if ":" not in model:
        model += ":latest"
    provenance = model_provenance(model)
    started_at = datetime.now(UTC).isoformat()
    input_mode = "interactive-terminal" if sys.stdin.isatty() else "piped-test-input"
    requests: list[dict[str, object]] = []
    execution = ExampleExecution()
    original_request = httpx.HTTPTransport.handle_request
    original_kickoff = CrewAIAdapter.kickoff

    def observe_request(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            if request.url.host != "localhost" or request.url.port != 11434:
                raise RuntimeError("An example attempted a non-local model request")
            parameters = json_object(json.loads(request.content))
            requests.append(
                {
                    "model": parameters.get("model"),
                    "temperature": parameters.get("temperature"),
                    "max_tokens": parameters.get("max_tokens"),
                }
            )

        return original_request(transport, request)

    def observe_kickoff(
        adapter: CrewAIAdapter, inputs: dict[str, object] | None = None
    ) -> CrewOutput:
        execution.runs += 1
        try:
            output = original_kickoff(adapter, inputs)
        except AdapterBlocked:
            execution.status = "blocked"
            raise
        else:
            execution.output = output
            execution.status = "allowed"
            return output
        finally:
            execution.records = adapter.records
            if adapter.records_dropped:
                raise RuntimeError("Example audit evidence was dropped")

    with (
        patch.object(httpx.HTTPTransport, "handle_request", observe_request),
        patch.object(CrewAIAdapter, "kickoff", observe_kickoff),
    ):
        if script is None:
            original_path = sys.path.copy()
            sys.path.insert(0, str(ROOT))
            try:
                runpy.run_module(f"examples.crewai.{example}", run_name="__main__", alter_sys=True)
            finally:
                sys.path[:] = original_path
        else:
            runpy.run_path(str(script), run_name="__main__")
    if (
        execution.runs != 1
        or execution.status is None
        or not execution.records
        or not 1 <= len(requests) <= 6
    ):
        raise RuntimeError("Missing or invalid real example execution evidence")
    if model_provenance(model) != provenance:
        raise RuntimeError("The selected model artifact or runtime changed during the run")
    if any(
        request != {"model": model, "temperature": 0, "max_tokens": 256} for request in requests
    ):
        raise RuntimeError("Observed model parameters differ from the example configuration")
    return {
        "example": example,
        "status": execution.status,
        "output": execution.output.raw if execution.output is not None else None,
        "model": {
            **provenance,
            "endpoint": "http://localhost:11434/v1",
            "started_at": started_at,
            "temperature": 0,
            "max_tokens": 256,
            "requests": len(requests),
        },
        "decisions": [
            {
                "point": record.interception_point.value,
                "decision": record.verdict.decision.value,
                "reason": record.verdict.reason,
                "resolved_by": record.resolved_by,
                "session_id": record.session_id,
                "sequence": record.sequence,
                "verdicts": [
                    {
                        "name": verdict.name,
                        "decision": verdict.decision.value,
                        "reason": verdict.reason,
                    }
                    for verdict in record.verdicts
                ],
            }
            for record in execution.records
        ],
        "details": {
            "input_mode": input_mode,
            "task_outputs": [task.raw for task in execution.output.tasks_output]
            if execution.output is not None
            else [],
        },
    }


def main() -> None:
    """Run one actual example and print its machine-readable evidence receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=EXAMPLES)
    parser.add_argument("--script", type=Path)
    arguments = parser.parse_args()
    example: object = arguments.example
    script: object = arguments.script
    if not isinstance(example, str) or (script is not None and not isinstance(script, Path)):
        raise ValueError("Invalid example arguments")
    result = record_example(example, script)
    print(f"RESULT_JSON: {json.dumps(result, ensure_ascii=True)}")


if __name__ == "__main__":
    main()
