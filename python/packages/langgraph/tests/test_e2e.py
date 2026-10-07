"""Required real native/ACS and isolated-wheel local-Qwen gates; never synthetic output."""

from pathlib import Path

import pytest
from graph_adapter_e2e_helpers import Scenario, SourceName, copied_environment, run_example

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("source", ["agent_hooks", "acs"])
@pytest.mark.parametrize("scenario", ["llm", "tool-allow", "tool-deny"])
def test_real_qwen(source: SourceName, scenario: Scenario, tmp_path: Path) -> None:
    run_example("langgraph", source, scenario, tmp_path / "receipts")


@pytest.mark.parametrize("source", ["agent_hooks", "acs"])
def test_complete_copied_folder(source: SourceName, tmp_path: Path) -> None:
    copied = copied_environment("langgraph", source, tmp_path)
    for scenario in ("llm", "tool-allow", "tool-deny"):
        run_example("langgraph", source, scenario, tmp_path / "receipts", copied=copied)
