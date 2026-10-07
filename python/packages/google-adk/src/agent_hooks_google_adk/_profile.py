"""Version-pinned, fail-closed inspection of the native runner and tool tree."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast, get_type_hints

from agent_hooks_adapters.contracts import validate_agent_id
from agent_hooks_adapters.json import JSON, normalize, object_value
from agent_hooks_adapters.runtime import validate_topology
from google.adk.agents import (
    BaseAgent,
    LlmAgent,
    SequentialAgent,  # pyright: ignore[reportDeprecated]
)
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool

_BASE_FIELDS = {
    "name",
    "description",
    "rerun_on_resume",
    "wait_for_output",
    "retry_config",
    "timeout",
    "input_schema",
    "output_schema",
    "state_schema",
    "parent_agent",
    "sub_agents",
    "before_agent_callback",
    "after_agent_callback",
}
_LLM_FIELDS = _BASE_FIELDS | {
    "model",
    "instruction",
    "global_instruction",
    "static_instruction",
    "tools",
    "generate_content_config",
    "mode",
    "parallel_worker",
    "disallow_transfer_to_parent",
    "disallow_transfer_to_peers",
    "include_contents",
    "output_key",
    "planner",
    "code_executor",
    "before_model_callback",
    "after_model_callback",
    "on_model_error_callback",
    "before_tool_callback",
    "after_tool_callback",
    "on_tool_error_callback",
}
_CONFIG_FIELDS = {"temperature", "top_p", "top_k", "max_output_tokens", "seed", "stop_sequences"}
_SCALARS = (str, int, float, bool)


def native_field(native: object, name: str) -> object:
    """Read an upstream Any/Unknown field as object for explicit boundary validation."""
    return getattr(native, name)


@dataclass(frozen=True)
class Parameter:
    """One scalar, keyword-callable native tool parameter."""

    name: str
    annotation: type[str] | type[int] | type[float] | type[bool]
    required: bool
    default: JSON


@dataclass(frozen=True)
class ToolSpec:
    """Strong native tool ownership and a strict original-argument schema."""

    actor: BaseAgent
    native: object
    function: object
    name: str
    parameters: tuple[Parameter, ...]

    def arguments(self, value: object) -> dict[str, JSON]:
        """Reject raw unknown/missing/coerced fields before ADK can sanitize them."""
        args = object_value(value)
        allowed = {parameter.name for parameter in self.parameters}
        if not set(args) <= allowed:
            raise ValueError("Unknown native tool arguments")
        for parameter in self.parameters:
            if parameter.name not in args:
                if parameter.required:
                    raise ValueError("Missing native tool argument")
                continue
            item = args[parameter.name]
            if type(item) is not parameter.annotation:
                raise ValueError("Native tool argument type cannot be coerced")
        return args


@dataclass(frozen=True)
class Profile:
    """Retain actual objects and an immutable supported-profile fingerprint."""

    runner: Runner
    root: BaseAgent
    actors: tuple[BaseAgent, ...]
    tools: tuple[ToolSpec, ...]
    fingerprint: tuple[object, ...]

    @property
    def agents(self) -> Mapping[str, object]:
        """Return trusted native names; prepare_controls captures this mapping."""
        return {actor.name: actor for actor in self.actors}

    def tool(self, actor: BaseAgent, name: str) -> ToolSpec:
        """Resolve only tools enumerated under this actual native actor."""
        for tool in self.tools:
            if tool.actor is actor and tool.name == name:
                return tool
        raise ValueError("Unknown or foreign native tool")


def _tool(actor: BaseAgent, native: object) -> ToolSpec:
    if type(native) is FunctionTool:
        if (
            native.is_long_running
            or native.custom_metadata is not None
            or native.behavior is not None
            or native.response_scheduling is not None
            or native_field(native, "_require_confirmation") is not False
            or native_field(native, "_defers_response") is not False
            or native_field(native, "_context_param_name") != "tool_context"
            or native_field(native, "_ignore_params") != ["tool_context", "input_stream"]
        ):
            raise ValueError("Unsupported native tool behavior")
        function: object = native.func
        name = native.name
    else:
        function = native
        name = getattr(native, "__name__", None)
    if not inspect.isfunction(function) or not inspect.iscoroutinefunction(function):
        raise TypeError("Only async Python functions or exact FunctionTool wrappers are supported")
    validate_agent_id(name)
    if not isinstance(name, str):
        raise TypeError("Native tool requires a name")
    if name in {"transfer_to_agent", "finish_task"}:
        raise ValueError("Delegation tools are unsupported")
    hints: dict[str, object] = get_type_hints(function)
    if type(native) is FunctionTool:
        spec = native_field(native, "_spec")
        if native_field(spec, "signature") != inspect.signature(function):
            raise ValueError("Native tool's cached signature differs from its actual function")
    parameters: list[Parameter] = []
    for parameter in inspect.signature(function).parameters.values():
        annotation = hints.get(parameter.name)
        if (
            parameter.kind
            not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            or parameter.name
            in {
                "tool_context",
                "input_stream",
                "agent_id",
                "session_id",
                "trace_id",
                "framework",
                "interception_point",
                "sequence",
                "policy",
                "correlation_id",
            }
            or not any(annotation is scalar for scalar in _SCALARS)
        ):
            raise ValueError(
                "Native tools require explicit scalar arguments without injected context"
            )
        if annotation is str:
            scalar = str
        elif annotation is bool:
            scalar = bool
        elif annotation is int:
            scalar = int
        else:
            scalar = float
        if parameter.default is not inspect.Parameter.empty:
            if type(parameter.default) is not scalar:
                raise ValueError("Native tool defaults must match their strict scalar types")
            normalize(parameter.default)
        parameters.append(
            Parameter(
                parameter.name,
                scalar,
                parameter.default is inspect.Parameter.empty,
                None
                if parameter.default is inspect.Parameter.empty
                else normalize(parameter.default),
            )
        )
    if len(parameters) > 64:
        raise ValueError("Native tool schema exceeds 64 parameters")
    return ToolSpec(actor, native, function, name, tuple(parameters))


def profile(runtime: object) -> Profile:
    """Reject unsupported ownership, plugins, profiles and topology before manifest I/O."""
    if type(runtime) is not Runner:
        raise TypeError("Google ADK requires an existing exact google.adk.Runner")
    if (
        type(runtime.app) is not App
        or runtime.app.root_agent is not runtime.agent
        or type(runtime.plugin_manager) is not PluginManager
        or runtime.plugin_manager.plugins
        or runtime.app.plugins
        or type(runtime.session_service) is not InMemorySessionService
        or runtime.artifact_service is not None
        or runtime.memory_service is not None
        or runtime.credential_service is not None
        or runtime.auto_create_session
        or runtime.context_cache_config is not None
        or runtime.resumability_config is not None
        or runtime.app.events_compaction_config is not None
        or runtime.app.context_cache_config is not None
        or runtime.app.resumability_config is not None
    ):
        raise ValueError(
            "Plugins, persistence, resume, cache and external runner services are unsupported"
        )
    root = runtime.agent
    if not isinstance(root, BaseAgent) or root.parent_agent is not None:
        raise ValueError("Runner must own a complete native agent tree")
    if isinstance(root, LlmAgent) and root.mode != "chat":
        raise ValueError(
            "Root LlmAgent requires explicit mode='chat'; native default normalization mutates it"
        )
    actors: list[BaseAgent] = []
    tools: list[ToolSpec] = []
    fingerprint: list[object] = [id(runtime), id(runtime.app), id(runtime.session_service)]

    def visit(actor: BaseAgent, parent: BaseAgent | None) -> None:
        if type(actor) not in (LlmAgent, SequentialAgent):  # pyright: ignore[reportDeprecated]
            raise TypeError("Only exact LlmAgent and SequentialAgent nodes are supported")
        if len(actors) >= 16 or any(actor is previous for previous in actors):
            raise ValueError("Native tree exceeds 16 nodes or repeats an actor")
        validate_agent_id(actor.name)
        if (
            any(actor.name == previous.name for previous in actors)
            or actor.parent_agent is not parent
        ):
            raise ValueError("Native names and parent membership must be unique and consistent")
        expected = _LLM_FIELDS if isinstance(actor, LlmAgent) else _BASE_FIELDS
        if set(type(actor).model_fields) != expected:
            raise ValueError(
                "Installed native agent field inventory differs from the pinned profile"
            )
        if (
            actor.rerun_on_resume
            or actor.wait_for_output
            or actor.retry_config is not None
            or actor.timeout is not None
            or native_field(actor, "input_schema") is not None
            or native_field(actor, "output_schema") is not None
            or actor.state_schema is not None
            or actor.before_agent_callback is not None
            or actor.after_agent_callback is not None
        ):
            raise ValueError(
                "Callbacks, retry, state schemas and resumable agent modes are unsupported"
            )
        actors.append(actor)
        normalize([actor.name, actor.description])
        fingerprint.extend((id(actor), actor.name, actor.description, id(parent)))
        if isinstance(actor, LlmAgent):
            if (
                not isinstance(actor.model, BaseLlm)
                or actor.sub_agents
                or not isinstance(actor.instruction, str)
                or not isinstance(actor.global_instruction, str)
                or native_field(actor, "static_instruction") is not None
                or actor.mode not in (None, "chat", "single_turn")
                or actor.parallel_worker not in (None, False)
                or actor.include_contents != "default"
                or actor.output_key is not None
                or actor.planner is not None
                or actor.code_executor is not None
                or any(
                    getattr(actor, field) is not None
                    for field in (
                        "before_model_callback",
                        "after_model_callback",
                        "on_model_error_callback",
                        "before_tool_callback",
                        "after_tool_callback",
                        "on_tool_error_callback",
                    )
                )
            ):
                raise ValueError(
                    "Only explicit text models without delegation or extra callbacks are supported"
                )
            if not actor.model.model or len(actor.model.model) > 1024:
                raise ValueError("Explicit native model identifier must contain 1-1024 characters")
            normalize([actor.instruction, actor.global_instruction])
            config = (
                object_value(
                    actor.generate_content_config.model_dump(mode="json", exclude_none=True)
                )
                if actor.generate_content_config is not None
                else {}
            )
            if not set(config) <= _CONFIG_FIELDS:
                raise ValueError(
                    "Rich, hosted-tool, cached and structured model configuration is unsupported"
                )
            normalize(config)
            raw_tools = native_field(actor, "tools")
            if not isinstance(raw_tools, list):
                raise TypeError("Native tools must be a list")
            native_tools = cast("list[object]", raw_tools)
            if len(native_tools) > 64:
                raise ValueError("Native actor exceeds 64 registered tools")
            actor_tools = [_tool(actor, tool) for tool in native_tools]
            if len({tool.name for tool in actor_tools}) != len(actor_tools):
                raise ValueError("Native tool names must be unique per actor")
            tools.extend(actor_tools)
            fingerprint.extend(
                (
                    id(actor.model),
                    actor.model.model,
                    actor.instruction,
                    actor.global_instruction,
                    config,
                    actor.mode,
                    actor.parallel_worker,
                    actor.disallow_transfer_to_parent,
                    actor.disallow_transfer_to_peers,
                    tuple(
                        (id(tool.native), id(tool.function), tool.name, tool.parameters)
                        for tool in actor_tools
                    ),
                )
            )
        elif not actor.sub_agents:
            raise ValueError("SequentialAgent requires at least one supported child")
        fingerprint.append(tuple(id(child) for child in actor.sub_agents))
        for child in actor.sub_agents:
            visit(child, actor)

    visit(root, None)
    validate_topology(actors=len(actors), nodes=len(actors), edges=len(actors) - 1)
    return Profile(runtime, root, tuple(actors), tuple(tools), tuple(fingerprint))


def revalidate(native: Profile) -> None:
    """Detect caller mutation before run admission effects and every guarded boundary."""
    if profile(native.runner).fingerprint != native.fingerprint:
        raise ValueError("Native Google ADK profile changed after controls were bound")
