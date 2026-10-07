"""Portable contracts validate owned JSON without importing a native backend."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from textwrap import dedent
from types import ModuleType
from typing import Literal

import agent_hooks_adapters.acs as acs
import agent_hooks_adapters.factory as factory
import pytest
from agent_hooks import ALLOW, AgentContext, InterceptionRecord, Verdict
from agent_hooks_adapters import (
    ACSUnavailableError,
    Adapter,
    AdapterBlocked,
    AdapterCleanupError,
    AgentPolicy,
    AsyncAdapter,
    Backend,
    RunRequest,
    RunResult,
    UnsupportedBackendError,
    create_adapter,
    with_agent_controls,
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


class ContractAsyncAdapter(ContractAdapter):
    """Exercise the async capability without a framework installation."""

    async def arun(self, request: RunRequest | None = None) -> RunResult:
        return self.run(request)


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


def test_backend_enum_contains_only_implemented_backends() -> None:
    expected = {
        Backend.CREWAI: "crewai",
        Backend.GOOGLE_ADK: "google-adk",
        Backend.LANGCHAIN: "langchain",
        Backend.LANGGRAPH: "langgraph",
        Backend.OPENAI_AGENTS: "openai-agents",
        Backend.MAF: "maf",
    }
    assert tuple(Backend) == tuple(expected)
    for backend, value in expected.items():
        assert backend.value == value
        assert Backend(value) is backend


@pytest.mark.parametrize(
    ("backend", "namespace"),
    [
        (Backend.GOOGLE_ADK, "agent_hooks_google_adk"),
        (Backend.LANGCHAIN, "agent_hooks_langchain"),
        (Backend.LANGGRAPH, "agent_hooks_langgraph"),
        (Backend.OPENAI_AGENTS, "agent_hooks_openai_agents"),
        (Backend.MAF, "agent_hooks_maf"),
    ],
)
def test_factory_selects_only_the_requested_async_backend(
    backend: Literal[
        Backend.GOOGLE_ADK,
        Backend.LANGCHAIN,
        Backend.LANGGRAPH,
        Backend.OPENAI_AGENTS,
        Backend.MAF,
    ],
    namespace: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_runtime = object()
    native_agent = object()
    agents = {"writer": native_agent}
    assignments = (AgentPolicy("writer", policy_file=tmp_path / "agent.yaml"),)
    expected = ContractAsyncAdapter()
    seen: list[dict[str, object]] = []
    module = ModuleType(f"{namespace}.portable")

    def create_backend(**options: object) -> ContractAsyncAdapter:
        seen.append(options)
        return expected

    def selected(name: str) -> ModuleType:
        assert name == f"{namespace}.portable"
        return module

    monkeypatch.setattr(module, "create_backend", create_backend, raising=False)
    monkeypatch.setattr(factory, "import_module", selected)
    adapter: AsyncAdapter = with_agent_controls(
        backend=backend,
        runtime=native_runtime,
        policy_file=tmp_path / "controls.yaml",
        agents=agents,
        agent_policies=assignments,
        timeout=1,
        emit_timeout=2,
        max_records=3,
    )
    assert adapter is expected
    assert seen == [
        {
            "runtime": native_runtime,
            "interceptors": None,
            "policy_file": tmp_path / "controls.yaml",
            "agents": agents,
            "agent_policies": assignments,
            "resolver": None,
            "timeout": 1,
            "emit_timeout": 2,
            "max_records": 3,
        }
    ]


@pytest.mark.parametrize(
    ("backend", "namespace"),
    [
        (Backend.CREWAI, "agent_hooks_crewai"),
        (Backend.GOOGLE_ADK, "agent_hooks_google_adk"),
        (Backend.LANGCHAIN, "agent_hooks_langchain"),
        (Backend.LANGGRAPH, "agent_hooks_langgraph"),
        (Backend.OPENAI_AGENTS, "agent_hooks_openai_agents"),
        (Backend.MAF, "agent_hooks_maf"),
    ],
)
@pytest.mark.parametrize("missing", ["backend", "dependency", "internal"])
def test_factory_distinguishes_a_missing_backend_from_a_broken_dependency(
    backend: Backend, namespace: str, missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_name = {
        "backend": namespace,
        "dependency": "absent_native_dependency",
        "internal": f"{namespace}._native",
    }[missing]

    def missing_import(name: str) -> ModuleType:
        assert name == f"{namespace}.portable"
        raise ModuleNotFoundError("selected import failed", name=missing_name)

    monkeypatch.setattr(factory, "import_module", missing_import)
    if missing == "backend":
        with pytest.raises(UnsupportedBackendError, match=f"Install the {backend.value}") as caught:
            with_agent_controls(backend=backend, runtime=object(), policy_file="controls.yaml")
        assert isinstance(caught.value.__cause__, ModuleNotFoundError)
        assert caught.value.__cause__.name == missing_name
    else:
        with pytest.raises(ModuleNotFoundError) as failed:
            with_agent_controls(backend=backend, runtime=object(), policy_file="controls.yaml")
        assert failed.value.name == missing_name


def test_original_factory_name_is_a_compatible_alias() -> None:
    assert create_adapter is with_agent_hooks is with_agent_controls


def test_acs_unavailable_error_has_an_explicit_import_base() -> None:
    assert issubclass(ACSUnavailableError, ImportError)


def test_agent_policy_owns_a_manifest_path_without_loading_it(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"
    policy = AgentPolicy("reviewer", policy_file=path)
    assert policy.interceptors == ()
    assert policy.policy_file == path.absolute()
    assert not path.exists()
    for field in ("agent_id", "interceptors", "policy_file"):
        with pytest.raises(FrozenInstanceError):
            setattr(policy, field, None)


@pytest.mark.parametrize("value", ["", "\x00", "invalid\x00.yaml"])
def test_agent_policy_rejects_invalid_manifest_paths(value: str) -> None:
    with pytest.raises(ValueError, match="path"):
        AgentPolicy("reviewer", policy_file=value)


@pytest.mark.parametrize("value", [False, 1, b"policy.yaml", object()])
def test_agent_policy_rejects_non_path_manifest_sources(value: object) -> None:
    with pytest.raises(TypeError, match="policy_file"):
        AgentPolicy("reviewer", policy_file=value)  # pyright: ignore[reportArgumentType]


def test_agent_policy_rejects_both_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="either"):
        AgentPolicy("reviewer", (AllowPolicy(),), policy_file=tmp_path / "policy.yaml")


def test_policy_path_preserves_lexical_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    policy = AgentPolicy("reviewer", policy_file="nested/../policy.yaml")
    assert policy.policy_file == tmp_path / "nested" / ".." / "policy.yaml"


@pytest.mark.parametrize("interceptors", [None, [], [AllowPolicy()]])
def test_factory_rejects_missing_or_conflicting_shared_sources(
    interceptors: list[AllowPolicy] | None, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        with_agent_controls(
            backend=Backend.CREWAI,
            runtime=object(),
            interceptors=interceptors,
            policy_file=tmp_path / "policy.yaml" if interceptors is not None else None,
        )


def test_missing_acs_distribution_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(acs, "version", missing)
    with pytest.raises(ACSUnavailableError, match=r"agent-hooks-adapters\[acs\]") as caught:
        acs.load_acs_policy(tmp_path / "policy.yaml")
    assert isinstance(caught.value.__cause__, PackageNotFoundError)


def test_incompatible_acs_version_fails_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_import(name: str) -> ModuleType:
        raise AssertionError("An incompatible SDK must not be imported")

    def incompatible(name: str) -> str:
        return "0.4.0a3"

    monkeypatch.setattr(acs, "version", incompatible)
    monkeypatch.setattr(acs, "import_module", no_import)
    with pytest.raises(ACSUnavailableError, match=r"0\.4\.0a4"):
        acs.load_acs_policy(tmp_path / "policy.yaml")


@pytest.mark.parametrize("missing_name", ["agent_control_spec", "agent_control_spec._native"])
def test_optional_import_does_not_hide_nested_failures(
    missing_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(name: str) -> ModuleType:
        raise ModuleNotFoundError("missing native SDK", name=missing_name)

    def compatible(name: str) -> str:
        return acs.ACS_VERSION

    monkeypatch.setattr(acs, "version", compatible)
    monkeypatch.setattr(acs, "import_module", missing)
    if missing_name == "agent_control_spec":
        with pytest.raises(ACSUnavailableError, match=r"\[acs\]"):
            acs.load_acs_policy(tmp_path / "policy.yaml")
    else:
        with pytest.raises(ModuleNotFoundError) as caught:
            acs.load_acs_policy(tmp_path / "policy.yaml")
        assert caught.value.name == missing_name


@pytest.mark.parametrize("value", ["", "CrewAI", "google_adk", "openai", "unknown"])
def test_backend_enum_rejects_unknown_values(value: str) -> None:
    with pytest.raises(ValueError):
        Backend(value)


@pytest.mark.parametrize(
    "backend", [*(backend.value for backend in Backend), "", "CrewAI", "openai", None, 1]
)
def test_factory_requires_a_supported_backend_enum(backend: object) -> None:
    with pytest.raises(UnsupportedBackendError, match="Backend enum member"):
        create_adapter(  # pyright: ignore[reportCallIssue]
            backend=backend,  # pyright: ignore[reportArgumentType]
            runtime=object(),
            interceptors=[AllowPolicy()],
        )


def test_core_contracts_and_unsupported_factory_do_not_import_native_backends() -> None:
    program = dedent(
        """
        import sys
        from importlib.abc import MetaPathFinder

        class BlockNativeBackend(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in {
                    "crewai", "agent_hooks_crewai", "agent_hooks_maf",
                    "agent_hooks_google_adk", "agent_hooks_langchain",
                    "agent_hooks_langgraph", "agent_hooks_openai_agents",
                    "agent_framework", "agent_control_spec", "google",
                    "langchain", "langchain_core", "langgraph", "agents", "openai",
                }:
                    raise AssertionError("core imported a native backend: " + fullname)
                return None

        sys.meta_path.insert(0, BlockNativeBackend())
        from agent_hooks import ALLOW
        from agent_hooks_adapters import (
            AgentPolicy, Backend, RunRequest, RunResult, UnsupportedBackendError,
            create_adapter, with_agent_controls, with_agent_hooks,
        )

        class Allow:
            def intercept(self, context):
                return ALLOW

        request = RunRequest({"value": [True, None, 1]})
        assert request.inputs == {"value": [True, None, 1]}
        assert RunResult(output="completed", records=(), records_dropped=0).output == "completed"
        assert AgentPolicy(agent_id="reviewer", interceptors=(Allow(),)).agent_id == "reviewer"
        assert AgentPolicy("reviewer", policy_file="missing.yaml").interceptors == ()
        assert create_adapter is with_agent_hooks is with_agent_controls
        assert Backend("crewai") is Backend.CREWAI
        for backend in ("crewai", "unknown"):
            try:
                create_adapter(backend=backend, runtime=object(), interceptors=[Allow()])
            except UnsupportedBackendError:
                pass
            else:
                raise AssertionError("non-enum backend was accepted")
        assert not any(
            name.split(".", 1)[0] in {
                "crewai", "agent_hooks_crewai", "agent_control_spec", "agent_hooks_maf",
                "agent_hooks_google_adk", "agent_hooks_langchain", "agent_hooks_langgraph",
                "agent_hooks_openai_agents", "agent_framework", "google", "langchain",
                "langchain_core", "langgraph", "agents", "openai",
            } for name in sys.modules
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
