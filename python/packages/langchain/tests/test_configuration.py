"""Native create-agent configuration proof rejects mutation before transport effects."""

import pytest
from agent_hooks_adapters import AdapterBlocked, RunRequest
from langchain_adapter_test_helpers import chain_adapter, chain_case
from langgraph_adapter_test_helpers import NativePolicy


@pytest.mark.parametrize("field", ["name", "description", "verbose", "tags", "metadata"])
def test_native_configuration_mutation(field: str) -> None:
    case = chain_case()
    adapter = chain_adapter(case, NativePolicy())
    if field == "name":
        case.model.name = "changed"
    else:
        value: object = {
            "description": "changed",
            "verbose": True,
            "tags": ["changed"],
            "metadata": {"changed": True},
        }[field]
        setattr(case.tool, field, value)
    with pytest.raises((ValueError, AdapterBlocked), match="configuration changed"):
        adapter.run(RunRequest({"prompt": "actual native admission"}))
    assert case.model.calls == 0 and case.effects == []
