"""Portable contracts validate owned JSON without importing a native backend."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from textwrap import dedent

import pytest
from agent_hooks import ALLOW, AgentContext, InterceptionRecord, Verdict
from agent_hooks_adapters import (
    Adapter,
    AdapterBlocked,
    AdapterCleanupError,
    AgentPolicy,
    Backend,
    RunRequest,
    RunResult,
    UnsupportedBackendError,
    create_adapter,
    with_agent_hooks,
)


class AllowPolicy:
    """Provide a deterministic interceptor without a native runtime."""

    def intercept(self, context: AgentContext) -> Verdict:
        return ALLOW


class ContractAdapter:
    """Exercise structural compatibility with the portable run protocol."""

    @property
    def records(self) -> tuple[InterceptionRecord, ...]:
        """Expose the contract's audit snapshot."""
        return ()

    @property
    def records_dropped(self) -> int:
        """Expose the contract's audit eviction count."""
        return 0

    @property
    def cleanup_error(self) -> AdapterCleanupError | None:
        """Expose the contract's cleanup diagnostic."""
        return None

    def run(self, request: RunRequest | None = None) -> RunResult:
        return RunResult(
            output="completed", records=self.records, records_dropped=self.records_dropped
        )


@pytest.mark.parametrize("inputs", [None, {}])
def test_optional_inputs_preserve_none_and_empty_object(
    inputs: dict[str, object] | None,
) -> None:
    assert RunRequest(inputs).inputs == inputs
    assert RunRequest().inputs is None


def test_request_owns_nested_inputs_and_each_read_is_detached() -> None:
    shared: dict[str, object] = {"items": ["original"]}
    source: dict[str, object] = {"question": "original", "first": shared, "second": shared}
    request = RunRequest(source)
    source["question"] = "changed"
    shared["items"] = ["changed"]
    expected = {
        "question": "original",
        "first": {"items": ["original"]},
        "second": {"items": ["original"]},
    }
    snapshot = request.inputs
    assert snapshot == expected
    assert snapshot is not None
    assert snapshot["first"] is not snapshot["second"]
    first = snapshot["first"]
    assert isinstance(first, dict)
    first["items"] = ["tampered"]
    snapshot["question"] = "tampered"
    assert request.inputs == expected
    assert request.inputs is not snapshot


def test_request_inputs_cannot_be_reassigned() -> None:
    request = RunRequest({"question": "original"})
    with pytest.raises(AttributeError):
        request.inputs = {"question": "changed"}  # pyright: ignore[reportAttributeAccessIssue]
    assert request.inputs == {"question": "original"}


@pytest.mark.parametrize(
    "value",
    [None, True, False, 0, -(2**53 - 1), 2**53 - 1, 1.25, -0.0, "text", [], {}],
)
def test_request_accepts_strict_interoperable_json(value: object) -> None:
    assert RunRequest({"value": value}).inputs == {"value": value}


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
        2**53,
        -(2**53),
        object(),
        {1: "not a string key"},
        ("tuple",),
        {"set"},
        b"bytes",
        "\ud800",
    ],
    ids=[
        "nan",
        "positive-infinity",
        "negative-infinity",
        "large-positive-integer",
        "large-negative-integer",
        "object",
        "non-string-key",
        "tuple",
        "set",
        "bytes",
        "unpaired-surrogate",
    ],
)
def test_request_rejects_non_json_values(value: object) -> None:
    with pytest.raises(ValueError):
        RunRequest({"value": value})


@pytest.mark.parametrize("value", [[], "text", 1, True, object()])
def test_request_rejects_non_object_inputs(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        RunRequest(value)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("kind", ["list", "dict"])
def test_request_rejects_cycles(kind: str) -> None:
    if kind == "list":
        sequence: list[object] = []
        sequence.append(sequence)
        value: object = sequence
    else:
        mapping: dict[str, object] = {}
        mapping["self"] = mapping
        value = mapping
    with pytest.raises(ValueError, match=r"[Cc]ycl"):
        RunRequest({"value": value})


def test_request_enforces_exact_depth_limit() -> None:
    value: object = None
    for _ in range(31):
        value = [value]
    assert RunRequest({"value": value}).inputs == {"value": value}
    with pytest.raises(ValueError, match="limit"):
        RunRequest({"value": [value]})


def test_request_enforces_exact_node_limit_including_keys() -> None:
    accepted: dict[str, object] = {"value": [None] * 9_997}
    assert RunRequest(accepted).inputs == accepted
    with pytest.raises(ValueError, match="limit"):
        RunRequest({"value": [None] * 9_998})


@pytest.mark.parametrize(
    "text",
    ["x" * 1_048_575, "\u00e9" * 524_287 + "x"],
    ids=["ascii", "utf8"],
)
def test_request_enforces_exact_utf8_byte_limit_including_keys(text: str) -> None:
    assert len(("x" + text).encode("utf-8")) == 1_048_576
    assert RunRequest({"x": text}).inputs == {"x": text}
    with pytest.raises(ValueError, match="limit"):
        RunRequest({"x": text + "x"})


@pytest.mark.parametrize(
    "agent_id", ["x", "application-reviewer", "writer.v1", "writer_v1", "123", "a" * 128]
)
def test_agent_policy_retains_logical_label_and_ordered_tuple(agent_id: str) -> None:
    first, second = AllowPolicy(), AllowPolicy()
    policy = AgentPolicy(agent_id=agent_id, interceptors=(first, second))
    assert policy.agent_id == agent_id
    assert policy.interceptors == (first, second)


@pytest.mark.parametrize(
    "agent_id",
    ["", " ", "\t", "\n", "bad\x00id", "a" * 129, "\u00e9", "with space", "bad/id", "bad:id"],
)
def test_agent_policy_rejects_invalid_logical_labels(agent_id: str) -> None:
    with pytest.raises(ValueError):
        AgentPolicy(agent_id=agent_id, interceptors=(AllowPolicy(),))


@pytest.mark.parametrize("agent_id", [None, 1, object()])
def test_agent_policy_rejects_non_string_labels(agent_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        AgentPolicy(
            agent_id=agent_id,  # pyright: ignore[reportArgumentType]
            interceptors=(AllowPolicy(),),
        )


def test_agent_policy_requires_nonempty_tuple() -> None:
    with pytest.raises(ValueError):
        AgentPolicy(agent_id="reviewer", interceptors=())
    with pytest.raises(TypeError):
        AgentPolicy(
            agent_id="reviewer",
            interceptors=[AllowPolicy()],  # pyright: ignore[reportArgumentType]
        )


@pytest.mark.parametrize(
    "field, value",
    [("output", "changed"), ("records", ()), ("records_dropped", 2)],
)
def test_run_result_is_frozen(field: str, value: object) -> None:
    result = RunResult(output="completed", records=(), records_dropped=0)
    with pytest.raises(FrozenInstanceError):
        setattr(result, field, value)
    assert result.output == "completed"
    assert result.records == ()
    assert result.records_dropped == 0


def test_adapter_protocol_accepts_a_structural_implementation() -> None:
    adapter: Adapter = ContractAdapter()
    result = adapter.run(RunRequest({"question": "hello"}))
    assert isinstance(result, RunResult)
    assert result.output == "completed"


def test_common_errors_have_portable_builtin_bases() -> None:
    assert issubclass(AdapterBlocked, RuntimeError)
    assert issubclass(AdapterCleanupError, RuntimeError)
    assert issubclass(UnsupportedBackendError, ValueError)


def test_backend_enum_contains_only_the_implemented_backend() -> None:
    assert tuple(Backend) == (Backend.CREWAI,)
    assert Backend.CREWAI.value == "crewai"
    assert Backend("crewai") is Backend.CREWAI


def test_original_factory_name_is_a_compatible_alias() -> None:
    assert create_adapter is with_agent_hooks


@pytest.mark.parametrize("value", ["", "CrewAI", "langchain", "openai", "unknown"])
def test_backend_enum_rejects_unknown_values(value: str) -> None:
    with pytest.raises(ValueError):
        Backend(value)


@pytest.mark.parametrize("backend", ["", "crewai", "CrewAI", "langchain", "openai", None, 1])
def test_factory_requires_a_supported_backend_enum(backend: object) -> None:
    with pytest.raises(UnsupportedBackendError, match="Backend enum member"):
        create_adapter(
            backend=backend,  # pyright: ignore[reportArgumentType]
            runtime=object(),
            interceptors=[AllowPolicy()],
        )


def test_core_contracts_and_unsupported_factory_do_not_import_crewai() -> None:
    program = dedent(
        """
        import sys
        from importlib.abc import MetaPathFinder

        class BlockNativeBackend(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in {"crewai", "agent_hooks_crewai"}:
                    raise AssertionError("core imported a native backend: " + fullname)
                return None

        sys.meta_path.insert(0, BlockNativeBackend())
        from agent_hooks import ALLOW
        from agent_hooks_adapters import (
            AgentPolicy, Backend, RunRequest, RunResult, UnsupportedBackendError, create_adapter,
        )

        class Allow:
            def intercept(self, context):
                return ALLOW

        request = RunRequest({"value": [True, None, 1]})
        assert request.inputs == {"value": [True, None, 1]}
        assert RunResult(output="completed", records=(), records_dropped=0).output == "completed"
        assert AgentPolicy(agent_id="reviewer", interceptors=(Allow(),)).agent_id == "reviewer"
        assert Backend("crewai") is Backend.CREWAI
        for backend in ("crewai", "unknown"):
            try:
                create_adapter(backend=backend, runtime=object(), interceptors=[Allow()])
            except UnsupportedBackendError:
                pass
            else:
                raise AssertionError("non-enum backend was accepted")
        assert not any(
            name.split(".", 1)[0] in {"crewai", "agent_hooks_crewai"} for name in sys.modules
        )
        print("core-only contracts verified")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "core-only contracts verified"
