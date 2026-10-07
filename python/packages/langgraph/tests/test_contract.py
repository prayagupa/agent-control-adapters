"""Owning LangGraph suite for shared graph-coupled native contract vectors."""

from __future__ import annotations

import pytest
from agent_hooks_langgraph.portable import create_backend
from graph_adapter_contract_vectors import (
    FamilyCase,
)
from graph_adapter_contract_vectors import (
    test_actual_binding_mutation_zero_effects as test_actual_binding_mutation_zero_effects,
)
from graph_adapter_contract_vectors import (
    test_actual_compiled_fields_revalidated as test_actual_compiled_fields_revalidated,
)
from graph_adapter_contract_vectors import (
    test_actual_transforms_preserve_ids_usage_and_roles as test_actual_transforms_preserve_ids_usage_and_roles,  # noqa: E501
)
from graph_adapter_contract_vectors import (
    test_async_cancel_drains_and_preserves_loop as test_async_cancel_drains_and_preserves_loop,
)
from graph_adapter_contract_vectors import (
    test_fixed_native_config as test_fixed_native_config,
)
from graph_adapter_contract_vectors import (
    test_invalid_admission_is_single_use as test_invalid_admission_is_single_use,
)
from graph_adapter_contract_vectors import (
    test_model_call_budget_exact_threshold as test_model_call_budget_exact_threshold,
)
from graph_adapter_contract_vectors import (
    test_native_deadline_drains as test_native_deadline_drains,
)
from graph_adapter_contract_vectors import (
    test_native_deny_each_point as test_native_deny_each_point,
)
from graph_adapter_contract_vectors import (
    test_one_tool_per_output as test_one_tool_per_output,
)
from graph_adapter_contract_vectors import (
    test_original_arguments_checked_before_native_parser as test_original_arguments_checked_before_native_parser,  # noqa: E501
)
from graph_adapter_contract_vectors import (
    test_real_acs_deny_each_point as test_real_acs_deny_each_point,
)
from graph_adapter_contract_vectors import (
    test_real_acs_engine_invalid_verdict as test_real_acs_engine_invalid_verdict,
)
from graph_adapter_contract_vectors import (
    test_real_acs_success_and_actor_order as test_real_acs_success_and_actor_order,
)
from graph_adapter_contract_vectors import (
    test_real_native_all_points as test_real_native_all_points,
)
from graph_adapter_contract_vectors import (
    test_records_retained_and_dropped_exact as test_records_retained_and_dropped_exact,
)
from graph_adapter_contract_vectors import (
    test_rich_null_model_output_rejected as test_rich_null_model_output_rejected,
)
from graph_adapter_contract_vectors import (
    test_schema_and_topology_mutation_revalidated as test_schema_and_topology_mutation_revalidated,
)
from graph_adapter_contract_vectors import (
    test_single_use_sync_async_race as test_single_use_sync_async_race,
)
from graph_adapter_contract_vectors import (
    test_unappliable_transform_aborts as test_unappliable_transform_aborts,
)
from graph_adapter_contract_vectors import (
    test_unknown_sdk_error_propagates as test_unknown_sdk_error_propagates,
)
from langgraph_adapter_test_helpers import graph_case


@pytest.fixture
def family_case() -> FamilyCase:
    case = graph_case()
    return FamilyCase(
        create_backend,
        case.build,
        case.model,
        case.effects,
        lambda: case.controls,
        lambda: case.graph,
    )
