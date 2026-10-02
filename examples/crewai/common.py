"""Local-only model setup and small, shared example helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import cast
from urllib.parse import urlsplit

import httpx
from agent_hooks import AgentContext, Decision, InterceptionRecord, Verdict
from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import Field


def json_object(value: object) -> dict[str, object]:
    """Validate an external JSON object before reading it."""
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise ValueError("Expected string JSON keys")
    return cast(dict[str, object], raw)


def local_base_url() -> str:
    """Reject remote endpoints, credentials, redirects and proxy routing."""
    value = os.getenv("LOCAL_MODEL_BASE_URL", "http://localhost:11434/v1")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path.rstrip("/") != "/v1"
        or parsed.query
        or parsed.fragment
        or parsed.port == 0
        or any(ord(char) <= 32 or ord(char) >= 127 for char in value)
    ):
        raise ValueError("LOCAL_MODEL_BASE_URL must be a loopback /v1 endpoint")
    return value.rstrip("/")


@dataclass
class ModelRun:
    """Metadata captured from the actual local runtime, not a claimed model identity."""

    model: str
    digest: str
    runtime: str
    endpoint: str
    started_at: str
    temperature: float = 0
    max_tokens: int = 256
    requests: int = 0


class LocalTextLLM(BaseLLM):
    """CrewAI's public custom-LLM interface backed by real, local HTTP requests.

    Text-only on purpose. The example owns the client instead of relying on
    provider internals to configure and close separate sync/async transports.
    """

    client: OpenAI = Field(exclude=True)

    def call(
        self,
        messages: object,
        tools: object = None,
        callbacks: object = None,
        available_functions: object = None,
        from_task: object = None,
        from_agent: object = None,
        response_model: object = None,
    ) -> str:
        """Generate a real response; reject unsupported tools and structured output."""
        if tools or available_functions or response_model:
            raise ValueError("This demonstration client supports text-only tasks")
        raw_messages: list[object]
        if isinstance(messages, str):
            raw_messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            raw_messages = cast(list[object], messages)
        else:
            raise ValueError("Expected a string or message list")
        request: list[ChatCompletionMessageParam] = []
        for raw in raw_messages:
            message = json_object(raw)
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("Example messages must contain textual content")
            match message.get("role"):
                case "user":
                    request.append({"role": "user", "content": content})
                case "system":
                    request.append({"role": "system", "content": content})
                case "assistant":
                    request.append({"role": "assistant", "content": content})
                case _:
                    raise ValueError("Unsupported example message role")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=request,
            temperature=0,
            max_tokens=256,
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise ValueError("Local model returned no textual completion")
        return content


@contextmanager
def local_llm() -> Generator[tuple[BaseLLM, ModelRun]]:
    """Own and close a no-retry, no-proxy local HTTP transport."""
    base_url = local_base_url()
    root = base_url.removesuffix("/v1")
    model = os.getenv("LOCAL_MODEL", "qwen2.5:latest")
    if ":" not in model:
        model += ":latest"
    with httpx.Client(
        timeout=httpx.Timeout(60, connect=3),
        trust_env=False,
        follow_redirects=False,
    ) as transport:
        tags = transport.get(f"{root}/api/tags", timeout=5)
        tags.raise_for_status()
        available = json_object(tags.json()).get("models")
        if not isinstance(available, list):
            raise ValueError("Ollama returned an invalid model list")
        digest: str | None = None
        for item in cast(list[object], available):
            entry = json_object(item)
            if entry.get("name") == model:
                value = entry.get("digest")
                if isinstance(value, str) and value:
                    digest = value
        if digest is None:
            raise ValueError(f"Install the chosen local model before running: {model}")
        version = transport.get(f"{root}/api/version", timeout=5)
        version.raise_for_status()
        runtime = json_object(version.json()).get("version")
        if not isinstance(runtime, str):
            raise ValueError("Ollama returned an invalid runtime version")
        run = ModelRun(model, digest, runtime, base_url, datetime.now(UTC).isoformat())

        def count_request(request: httpx.Request) -> None:
            if request.url.path.endswith("/chat/completions"):
                if run.requests >= 6:
                    raise RuntimeError("Example model-request budget exhausted")
                run.requests += 1

        transport.event_hooks["request"].append(count_request)
        with OpenAI(
            api_key="ollama",
            base_url=base_url,
            timeout=60,
            max_retries=0,
            http_client=transport,
        ) as client:
            yield LocalTextLLM(model=model, temperature=0, client=client), run


def make_agent(llm: BaseLLM, role: str, goal: str) -> Agent:
    """Create a bounded agent without delegation, persistence or external tools."""
    return Agent(
        role=role,
        goal=goal,
        backstory="You answer the assigned task concisely and accurately.",
        llm=llm,
        allow_delegation=False,
        respect_context_window=False,
        max_iter=3,
        max_retry_limit=0,
        verbose=False,
    )


def make_crew(agents: list[Agent], tasks: list[Task]) -> Crew:
    """Use only the adapter's explicitly supported sequential execution profile."""
    return Crew(
        agents=[agent for agent in agents],
        tasks=tasks,
        process=Process.sequential,
        memory=False,
        planning=False,
        cache=False,
        verbose=False,
    )


class NoTools:
    """A shared policy: these text-only examples must never invoke tools."""

    def intercept(self, context: AgentContext) -> Verdict:
        if context["interception_point"] == "pre_tool_call":
            return Verdict.deny(reason="example:no_tools")
        return Verdict(decision=Decision.ALLOW)


@dataclass
class AgentTextLimit:
    """An agent-specific output control; hits show which agent actually ran it."""

    max_characters: int
    seen_agent_ids: set[str] = field(default_factory=set[str])

    def intercept(self, context: AgentContext) -> Verdict:
        envelope = json_object(context["agent"])
        identifier = envelope.get("id")
        if not isinstance(identifier, str):
            return Verdict.deny(reason="example:missing_agent")
        self.seen_agent_ids.add(identifier)
        if context["interception_point"] == "post_model_call":
            response = json_object(context["target"])
            content = response.get("content")
            if not isinstance(content, str) or len(content) > self.max_characters:
                return Verdict.deny(reason="example:text_limit")
        return Verdict(decision=Decision.ALLOW)


def report(
    *,
    example: str,
    status: str,
    output: str | None,
    run: ModelRun,
    records: Sequence[InterceptionRecord],
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    """Print genuine output plus a machine-readable governance receipt."""
    result: dict[str, object] = {
        "example": example,
        "status": status,
        "output": output,
        "model": asdict(run),
        "decisions": [
            {
                "point": record.interception_point.value,
                "decision": record.verdict.decision.value,
                "reason": record.verdict.reason,
                "resolved_by": record.resolved_by,
                "session_id": record.session_id,
                "sequence": record.sequence,
            }
            for record in records
        ],
        "details": details or {},
    }
    if run.requests == 0:
        raise RuntimeError("The example did not make a real local model request")
    print(f"RESULT_JSON: {json.dumps(result, ensure_ascii=True)}")
    return result
