"""Owning LangGraph P3 consumers and actual async runner lifecycle proofs."""

import pytest
from agent_control_langgraph.portable import create_backend
from graph_adapter_contract_vectors import FamilyCase
from graph_adapter_p3_vectors import (
    test_actual_native_resistant_retention_and_repeated_cleanup as test_actual_native_resistant_retention_and_repeated_cleanup,  # noqa: E501
)
from graph_adapter_p3_vectors import (
    test_real_rego_transforms_consumed as test_real_rego_transforms_consumed,
)
from graph_adapter_p3_vectors import (
    test_real_rego_unsupported_metadata_and_routing_fail_closed as test_real_rego_unsupported_metadata_and_routing_fail_closed,  # noqa: E501
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
