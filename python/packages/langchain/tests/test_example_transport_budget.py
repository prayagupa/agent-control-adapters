"""Keep transport and exact-effect checks in the owning LangChain project."""

from pathlib import Path

import pytest
from agent_adapter_conformance import assert_graph_example_transport_budget

ROOT: Path = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("source", ["agent_hooks", "acs"])
@pytest.mark.parametrize("script", ["llm_hooks.py", "tool_hooks.py"])
def test_all_examples_explicit_transport_budget(source: str, script: str) -> None:
    """Retain the original checks without requiring LangChain in LangGraph snapshots."""
    assert_graph_example_transport_budget(ROOT / "examples" / "langchain" / source / script)
